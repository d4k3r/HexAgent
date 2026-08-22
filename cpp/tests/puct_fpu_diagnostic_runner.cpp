#include "hex_puct/onnx_evaluator.hpp"
#include "hex_puct/puct.hpp"
#include "hex_puct/student_boundary.hpp"
#ifdef HEX_PUCT_SHARED_BATCH
#include "hex_puct/batched_inference_service.hpp"
#endif

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace hex_puct;
namespace fs = std::filesystem;

namespace {

std::string StringField(const std::string& line, const std::string& key) {
  const std::string marker = "\"" + key + "\":\"";
  const auto begin = line.find(marker);
  if (begin == std::string::npos) throw std::runtime_error("missing string field: " + key);
  const auto start = begin + marker.size();
  const auto end = line.find('"', start);
  if (end == std::string::npos) throw std::runtime_error("unterminated string field: " + key);
  return line.substr(start, end - start);
}

int IntField(const std::string& line, const std::string& key) {
  const std::string marker = "\"" + key + "\":";
  const auto begin = line.find(marker);
  if (begin == std::string::npos) throw std::runtime_error("missing integer field: " + key);
  size_t used = 0;
  const int value = std::stoi(line.substr(begin + marker.size()), &used);
  (void)used;
  return value;
}

std::optional<int> OptionalIntField(const std::string& line, const std::string& key) {
  const std::string marker = "\"" + key + "\":\"";
  const auto begin = line.find(marker);
  if (begin != std::string::npos) {
    const auto start = begin + marker.size();
    const auto end = line.find('"', start);
    return std::stoi(line.substr(start, end - start));
  }
  const std::string raw_marker = "\"" + key + "\":";
  const auto raw = line.find(raw_marker);
  if (raw == std::string::npos) throw std::runtime_error("missing optional field: " + key);
  const auto start = raw + raw_marker.size();
  if (line.compare(start, 4, "null") == 0) return std::nullopt;
  size_t used = 0;
  return std::stoi(line.substr(start), &used);
}

std::vector<int> ArrayField(const std::string& line, const std::string& key) {
  const std::string marker = "\"" + key + "\":[";
  const auto begin = line.find(marker);
  if (begin == std::string::npos) throw std::runtime_error("missing array field: " + key);
  const auto start = begin + marker.size();
  const auto end = line.find(']', start);
  if (end == std::string::npos) throw std::runtime_error("unterminated array field: " + key);
  std::vector<int> out;
  std::stringstream stream(line.substr(start, end - start));
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) out.push_back(std::stoi(token));
  }
  return out;
}

void Array(std::ostream& out, const std::array<int, kBoardArea>& values) {
  out << '[';
  for (int i = 0; i < kBoardArea; ++i) { if (i) out << ','; out << values[i]; }
  out << ']';
}

FpuMode ParseMode(const std::string& mode) {
  if (mode == "zero") return FpuMode::Zero;
  if (mode == "parent_value_reduced") return FpuMode::ParentValueReduced;
  throw std::runtime_error("unsupported fpu mode: " + mode);
}

struct Position { std::string id; std::string line; };

struct Options {
  std::string model, bank, output;
  double c_puct = 1.5;
  std::string fpu_mode = "zero";
  double fpu_reduction = 0.0;
  int visits = 0;
  int limit = -1;
  size_t concurrency = 1;
  size_t max_batch = 1;
  size_t max_queue = 256;
  int wait_us = 200;
};

Options Parse(int argc, char** argv) {
  if (argc < 8) {
    throw std::runtime_error("usage: model.onnx bank.jsonl output.jsonl c_puct fpu_mode fpu_reduction visits [limit] [--concurrency N --max-batch N --max-queue N --wait-us U]");
  }
  Options x;
  x.model = argv[1]; x.bank = argv[2]; x.output = argv[3];
  x.c_puct = std::stod(argv[4]); x.fpu_mode = argv[5];
  x.fpu_reduction = std::stod(argv[6]); x.visits = std::stoi(argv[7]);
  int i = 8;
  if (i < argc && argv[i][0] != '-') x.limit = std::stoi(argv[i++]);
  while (i < argc) {
    const std::string key = argv[i++];
    if (i >= argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[i++];
    if (key == "--concurrency") x.concurrency = static_cast<size_t>(std::stoul(value));
    else if (key == "--max-batch") x.max_batch = static_cast<size_t>(std::stoul(value));
    else if (key == "--max-queue") x.max_queue = static_cast<size_t>(std::stoul(value));
    else if (key == "--wait-us") x.wait_us = std::stoi(value);
    else throw std::runtime_error("unknown option " + key);
  }
  if (x.visits <= 0 || x.concurrency == 0 || x.max_batch == 0 || x.max_queue == 0 || x.wait_us < 0)
    throw std::runtime_error("invalid diagnostic configuration");
  ParseMode(x.fpu_mode);
  if (x.fpu_mode == "zero" && x.fpu_reduction != 0.0) throw std::runtime_error("zero FPU requires reduction=0");
#ifndef HEX_PUCT_SHARED_BATCH
  x.concurrency = 1;
#endif
  return x;
}

std::vector<Position> ReadPositions(const Options& x) {
  std::ifstream input(x.bank);
  if (!input) throw std::runtime_error("unable to open bank");
  std::vector<Position> positions;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    if (x.limit >= 0 && static_cast<int>(positions.size()) >= x.limit) break;
    positions.push_back({StringField(line, "position_id"), line});
  }
  if (positions.empty()) throw std::runtime_error("bank contains no positions");
  return positions;
}

void LoadPartial(const fs::path& path, const std::vector<Position>& positions,
                 std::vector<std::string>& results, std::vector<bool>& done) {
  if (!fs::exists(path)) return;
  std::unordered_map<std::string, size_t> index;
  for (size_t i = 0; i < positions.size(); ++i) index.emplace(positions[i].id, i);
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    try {
      const std::string id = StringField(line, "position_id");
      const auto it = index.find(id);
      if (it == index.end()) throw std::runtime_error("partial result contains position outside requested bank: " + id);
      if (done[it->second]) throw std::runtime_error("duplicate partial position_id: " + id);
      done[it->second] = true; results[it->second] = line;
    } catch (...) {
      // A process killed while appending can leave one unterminated final line.
      // It is deliberately discarded and regenerated; malformed complete lines fail.
      if (input.eof()) break;
      throw;
    }
  }
}

std::string Serialize(const std::string& line, double c_puct, const std::string& mode,
                      double reduction, int visits, const SearchResult& result, double search_seconds) {
  const auto black = ArrayField(line, "black");
  const auto white = ArrayField(line, "white");
  const Color side = StringField(line, "side_to_move") == "B" ? Color::Black : Color::White;
  const auto last = OptionalIntField(line, "last_move");
  const Board board = Board::FromSetup(black, white, side, last);
  int legal = static_cast<int>(board.LegalActions().size());
  int visited = 0;
  for (int n : result.raw_visits) if (n > 0) ++visited;
  double entropy = 0.0;
  for (double p : result.policy) if (p > 0) entropy -= p * std::log(p);
  std::ostringstream output;
  output << std::setprecision(17);
  output << "{\"position_id\":\"" << StringField(line, "position_id")
         << "\",\"source\":\"" << StringField(line, "source")
         << "\",\"game_id\":\"" << StringField(line, "game_id")
         << "\",\"ply\":" << IntField(line, "ply")
         << ",\"side_to_move\":\"" << StringField(line, "side_to_move")
         << "\",\"c_puct\":" << c_puct
         << ",\"fpu_mode\":\"" << mode << "\",\"fpu_reduction\":" << reduction
         << ",\"requested_visits\":" << visits
         << ",\"selected_action\":" << (result.selected_action ? std::to_string(*result.selected_action) : "null")
         << ",\"root_visits\":";
  Array(output, result.raw_visits);
  output << ",\"root_policy\":[";
  for (int i = 0; i < kBoardArea; ++i) { if (i) output << ','; output << result.policy[i]; }
  output << "],\"root_value\":";
  if (result.root_value) output << *result.root_value; else output << "null";
  output << ",\"policy_entropy\":" << entropy
         << ",\"legal_actions\":" << legal << ",\"visited_children\":" << visited
         << ",\"max_depth\":" << result.max_depth << ",\"evaluations\":" << result.evaluations
         << ",\"simulations\":" << visits << ",\"search_seconds\":" << search_seconds << "}";
  return output.str();
}

void WriteTelemetry(const fs::path& output, const Options& x, size_t positions,
                    size_t requests, size_t batches, size_t queue_high_water,
                    const std::vector<size_t>& batch_sizes, size_t evaluations,
                    double elapsed, bool failed, const std::string& failure) {
  std::ofstream out(output.string() + ".telemetry.json", std::ios::trunc);
  if (!out) throw std::runtime_error("unable to write diagnostic telemetry");
  size_t peak = 0; for (size_t n : batch_sizes) peak = std::max(peak, n);
  const double mean = batches ? static_cast<double>(requests) / batches : 0.0;
  out << std::setprecision(17)
      << "{\"schema\":\"hex-puct-fpu-diagnostic-telemetry-v1\",\"backend\":\""
#ifdef HEX_PUCT_SHARED_BATCH
      << "shared_batched"
#else
      << "direct"
#endif
      << "\",\"positions\":" << positions << ",\"requests\":" << requests
      << ",\"batches\":" << batches << ",\"mean_batch\":" << mean
      << ",\"peak_batch\":" << peak << ",\"queue_high_water\":" << queue_high_water
      << ",\"evaluations\":" << evaluations << ",\"simulations\":"
      << (static_cast<uint64_t>(positions) * static_cast<uint64_t>(x.visits))
      << ",\"elapsed_seconds\":" << elapsed << ",\"simulations_per_second\":"
      << (elapsed > 0 ? static_cast<double>(positions) * x.visits / elapsed : 0.0)
      << ",\"positions_per_second\":" << (elapsed > 0 ? positions / elapsed : 0.0)
      << ",\"concurrency\":" << x.concurrency << ",\"max_batch\":" << x.max_batch
      << ",\"max_queue\":" << x.max_queue << ",\"wait_us\":" << x.wait_us
      << ",\"active_searches\":0,\"peak_active_searches\":" << x.concurrency
      << ",\"failed\":" << (failed ? "true" : "false")
      << ",\"failure_reason\":\"" << failure << "\"}\n";
}

#ifdef HEX_PUCT_DIAGNOSTIC_FAKE
class DiagnosticFakeSingle final : public SynchronousModelEvaluator {
 public:
  Evaluation Evaluate(const StudentTensor& input) const override {
    Evaluation e{};
    double sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) sum += input[i] * static_cast<double>((i % 7) + 1);
    for (int a = 0; a < kBoardArea; ++a) e.policy_logits[a] = 0.001 * ((a * 17 + static_cast<int>(sum * 13)) % 101);
    e.value = std::tanh(sum * 0.001);
    return e;
  }
};

class DiagnosticFakeBatch final : public BatchedModelEvaluator {
 public:
  std::vector<Evaluation> EvaluateBatch(const std::vector<StudentTensor>& input) const override {
    std::vector<Evaluation> output; output.reserve(input.size());
    for (const auto& x : input) output.push_back(single_.Evaluate(x));
    return output;
  }
 private:
  DiagnosticFakeSingle single_;
};
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options x = Parse(argc, argv);
    const auto positions = ReadPositions(x);
    const fs::path output_path(x.output);
    std::vector<std::string> results(positions.size());
    std::vector<bool> done(positions.size(), false);
    LoadPartial(output_path, positions, results, done);
    std::ofstream append(output_path, std::ios::app);
    if (!append) throw std::runtime_error("unable to open diagnostic partial output");

    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex, output_mutex;
    const auto started = std::chrono::steady_clock::now();

#ifdef HEX_PUCT_SHARED_BATCH
#ifdef HEX_PUCT_DIAGNOSTIC_FAKE
    DiagnosticFakeBatch model_evaluator;
#else
    OnnxCudaBatchEvaluator model_evaluator(x.model);
#endif
    SharedInferenceService service(model_evaluator,
      {x.max_batch, x.max_queue, std::chrono::microseconds(x.wait_us)});
    EncodedBoardEvaluator evaluator(service);
#else
#ifdef HEX_PUCT_DIAGNOSTIC_FAKE
    DiagnosticFakeSingle model_evaluator;
#else
    OnnxCudaEvaluator model_evaluator(x.model);
#endif
    EncodedBoardEvaluator evaluator(model_evaluator);
#endif

    auto worker = [&]() {
      try {
        while (!failed) {
          const size_t index = next.fetch_add(1);
          if (index >= positions.size()) break;
          if (done[index]) continue;
          const auto black = ArrayField(positions[index].line, "black");
          const auto white = ArrayField(positions[index].line, "white");
          const Color side = StringField(positions[index].line, "side_to_move") == "B" ? Color::Black : Color::White;
          const auto last = OptionalIntField(positions[index].line, "last_move");
          const Board board = Board::FromSetup(black, white, side, last);
          DeterministicPUCT search(evaluator, {x.visits, x.c_puct, ParseMode(x.fpu_mode), x.fpu_reduction});
          const auto search_started = std::chrono::steady_clock::now();
          const auto search_result = search.Search(board);
          const double search_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - search_started).count();
          const std::string serialized = Serialize(positions[index].line, x.c_puct, x.fpu_mode, x.fpu_reduction, x.visits, search_result, search_seconds);
          {
            std::lock_guard lock(output_mutex);
            results[index] = serialized;
            done[index] = true;
            append << serialized << '\n';
            append.flush();
          }
        }
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (!error) error = std::current_exception();
        failed = true;
      }
    };

    std::vector<std::thread> workers;
    for (size_t i = 0; i < x.concurrency; ++i) workers.emplace_back(worker);
    for (auto& thread : workers) thread.join();
#ifdef HEX_PUCT_SHARED_BATCH
    const auto health = service.Health();
    const auto stats = service.Stats();
    service.Shutdown();
#endif
    append.close();
    if (error) std::rethrow_exception(error);
    for (bool value : done) if (!value) throw std::runtime_error("diagnostic did not complete all positions");

    const fs::path canonical = output_path.string() + ".canonical";
    {
      std::ofstream out(canonical, std::ios::trunc);
      if (!out) throw std::runtime_error("unable to open canonical diagnostic output");
      for (const auto& line : results) out << line << '\n';
      out.flush();
    }
    fs::rename(canonical, output_path);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
#ifdef HEX_PUCT_SHARED_BATCH
    WriteTelemetry(output_path, x, positions.size(), stats.requests, stats.batches, stats.queue_high_water,
                   stats.batch_sizes, stats.requests, elapsed, health.failed, health.failure_reason);
    std::cerr << "diagnostic_positions=" << positions.size() << " requests=" << stats.requests
              << " batches=" << stats.batches << " mean_batch="
              << (stats.batches ? static_cast<double>(stats.requests) / stats.batches : 0.0) << "\n";
#else
    size_t evaluations = 0;
    for (const auto& line : results) evaluations += static_cast<size_t>(IntField(line, "evaluations"));
    WriteTelemetry(output_path, x, positions.size(), evaluations, evaluations, 0,
                   std::vector<size_t>(evaluations, 1), evaluations, elapsed, false, "");
    std::cerr << "diagnostic_positions=" << positions.size() << "\n";
#endif
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "puct_fpu_diagnostic_error=" << error.what() << "\n";
    return 1;
  }
}
