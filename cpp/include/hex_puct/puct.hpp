#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace hex_puct {
constexpr int kBoardSize = 11;
constexpr int kBoardArea = kBoardSize * kBoardSize;
enum class Color : unsigned char { Black, White };
Color Opponent(Color color);

class Board {
 public:
  explicit Board(Color side_to_move = Color::Black);
  static Board FromSetup(const std::vector<int>& black, const std::vector<int>& white,
                         Color side_to_move, std::optional<int> last_move = std::nullopt);
  void Play(int action);
  [[nodiscard]] std::vector<int> LegalActions() const;
  [[nodiscard]] std::array<bool, kBoardArea> LegalMask() const;
  [[nodiscard]] std::optional<Color> LiteralWinner() const;
  [[nodiscard]] Color side_to_move() const { return side_to_move_; }
  [[nodiscard]] int Cell(int action) const { return cells_[action]; }  // 0 empty, 1 black, 2 white
  [[nodiscard]] std::optional<int> last_move() const { return last_move_; }
  [[nodiscard]] int Signature() const;

 private:
  void Place(int action, Color color);
  std::array<unsigned char, kBoardArea> cells_{};
  Color side_to_move_;
  std::optional<int> last_move_;
};

struct Evaluation { std::array<double, kBoardArea> policy_logits; double value; };
using Evaluator = std::function<Evaluation(const Board&)>;
struct Edge { int action; double prior; std::unique_ptr<struct Node> child; int visits = 0; double value_sum = 0; double Q() const; };
enum class FpuMode : unsigned char { Zero, ParentValueReduced };
double ComputeFpuQ(FpuMode mode, double parent_value, double reduction, double visited_prior_mass);
struct Node { Color to_play; int visits = 0; std::vector<Edge> edges; double expansion_value = 0.0; };
struct SearchConfig {
  int simulations;
  double c_puct = 1.5;
  FpuMode fpu_mode = FpuMode::Zero;
  double fpu_reduction = 0.0;
};
struct SearchResult {
  std::optional<int> selected_action;
  int root_visits = 0;
  std::array<int, kBoardArea> raw_visits{};
  std::array<double, kBoardArea> policy{};
  std::optional<double> root_value;
  std::array<double, kBoardArea> priors{};
  std::array<double, kBoardArea> raw_value_sums{};
  int evaluations = 0;
  int max_depth = 0;
};

class DeterministicPUCT {
 public:
  DeterministicPUCT(Evaluator evaluator, SearchConfig config);
  SearchResult Search(const Board& source) const;
 private:
  double Expand(Node& node, const Board& board) const;
  Edge& Select(Node& node) const;
  double FpuQ(const Node& node) const;
  Evaluator evaluator_;
  SearchConfig config_;
};

class DeterministicFakeEvaluator {
 public:
  Evaluation operator()(const Board& board) const;
};
}  // namespace hex_puct
