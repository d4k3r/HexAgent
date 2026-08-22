#include "hex_puct/puct.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace hex_puct {

Color Opponent(Color color) { return color == Color::Black ? Color::White : Color::Black; }

Board::Board(Color side_to_move) : side_to_move_(side_to_move) {}

Board Board::FromSetup(const std::vector<int>& black, const std::vector<int>& white,
                       Color side_to_move, std::optional<int> last_move) {
  Board board(side_to_move);
  for (int a : black) board.Place(a, Color::Black);
  for (int a : white) board.Place(a, Color::White);
  if (last_move && (*last_move < 0 || *last_move >= kBoardArea || board.cells_[*last_move] == 0))
    throw std::invalid_argument("invalid last move");
  board.last_move_ = last_move;
  return board;
}

void Board::Place(int action, Color color) {
  if (action < 0 || action >= kBoardArea || cells_[action]) throw std::invalid_argument("invalid setup action");
  cells_[action] = color == Color::Black ? 1 : 2;
}

void Board::Play(int action) {
  if (LiteralWinner() || action < 0 || action >= kBoardArea || cells_[action])
    throw std::invalid_argument("illegal action");
  Place(action, side_to_move_);
  last_move_ = action;
  side_to_move_ = Opponent(side_to_move_);
}

std::vector<int> Board::LegalActions() const {
  if (LiteralWinner()) return {};
  std::vector<int> result;
  for (int a = 0; a < kBoardArea; ++a) if (!cells_[a]) result.push_back(a);
  return result;
}

std::array<bool, kBoardArea> Board::LegalMask() const {
  std::array<bool, kBoardArea> out{};
  if (!LiteralWinner()) for (int a = 0; a < kBoardArea; ++a) out[a] = !cells_[a];
  return out;
}

std::optional<Color> Board::LiteralWinner() const {
  constexpr int dx[6] = {-1, -1, 0, 0, 1, 1};
  constexpr int dy[6] = {0, 1, -1, 1, -1, 0};
  for (Color c : {Color::Black, Color::White}) {
    unsigned char stone = c == Color::Black ? 1 : 2;
    std::queue<int> q;
    std::array<bool, kBoardArea> seen{};
    for (int a = 0; a < kBoardArea; ++a) {
      int x = a % kBoardSize, y = a / kBoardSize;
      if (cells_[a] == stone && ((c == Color::Black && y == 0) || (c == Color::White && x == 0))) {
        q.push(a); seen[a] = true;
      }
    }
    while (!q.empty()) {
      int a = q.front(); q.pop();
      int x = a % kBoardSize, y = a / kBoardSize;
      if ((c == Color::Black && y == kBoardSize - 1) || (c == Color::White && x == kBoardSize - 1)) return c;
      for (int i = 0; i < 6; ++i) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx >= 0 && nx < kBoardSize && ny >= 0 && ny < kBoardSize) {
          int n = ny * kBoardSize + nx;
          if (!seen[n] && cells_[n] == stone) { seen[n] = true; q.push(n); }
        }
      }
    }
  }
  return std::nullopt;
}

int Board::Signature() const {
  int s = side_to_move_ == Color::Black ? 17 : 29;
  for (int a = 0; a < kBoardArea; ++a) s = (s + (a + 1) * cells_[a]) % 1000003;
  return s;
}

double Edge::Q() const { return visits ? value_sum / visits : 0.0; }

double ComputeFpuQ(FpuMode mode, double parent_value, double reduction, double visited_prior_mass) {
  if (mode == FpuMode::Zero) return 0.0;
  const double raw = parent_value - reduction * std::sqrt(std::max(0.0, visited_prior_mass));
  return std::max(-1.0, std::min(1.0, raw));
}

DeterministicPUCT::DeterministicPUCT(Evaluator evaluator, SearchConfig config)
    : evaluator_(std::move(evaluator)), config_(config) {
  if (config.simulations < 0 || !std::isfinite(config.c_puct) || config.c_puct < 0 ||
      !std::isfinite(config.fpu_reduction) || config.fpu_reduction < 0)
    throw std::invalid_argument("invalid config");
  if (config.fpu_mode == FpuMode::Zero && config.fpu_reduction != 0)
    throw std::invalid_argument("zero FPU cannot have a reduction");
}

double DeterministicPUCT::Expand(Node& node, const Board& board) const {
  auto e = evaluator_(board);
  if (!std::isfinite(e.value) || e.value < -1.000001 || e.value > 1.000001)
    throw std::invalid_argument("invalid value");
  node.expansion_value = e.value;
  auto legal = board.LegalActions();
  if (legal.empty()) return e.value;
  double maximum = -INFINITY;
  for (int a : legal) {
    if (!std::isfinite(e.policy_logits[a])) throw std::invalid_argument("invalid legal logit");
    maximum = std::max(maximum, e.policy_logits[a]);
  }
  double total = 0;
  for (int a : legal) total += std::exp(e.policy_logits[a] - maximum);
  if (!std::isfinite(total) || total <= 0) throw std::invalid_argument("invalid priors");
  node.edges.clear(); node.edges.reserve(legal.size());
  for (int a : legal) node.edges.push_back({a, std::exp(e.policy_logits[a] - maximum) / total});
  return e.value;
}

double DeterministicPUCT::FpuQ(const Node& node) const {
  if (config_.fpu_mode == FpuMode::Zero) return 0.0;
  double visited_prior = 0.0;
  for (const auto& edge : node.edges) if (edge.visits > 0) visited_prior += edge.prior;
  return ComputeFpuQ(config_.fpu_mode, node.expansion_value, config_.fpu_reduction, visited_prior);
}

Edge& DeterministicPUCT::Select(Node& node) const {
  double scale = config_.c_puct * std::sqrt(std::max(1, node.visits));
  const double fpu = FpuQ(node);
  Edge* best = nullptr; double best_score = -INFINITY;
  for (auto& e : node.edges) {
    double score = (e.visits ? e.Q() : fpu) + scale * e.prior / (1 + e.visits);
    if (!best || score > best_score || (score == best_score && e.action < best->action)) {
      best = &e; best_score = score;
    }
  }
  return *best;
}

SearchResult DeterministicPUCT::Search(const Board& source) const {
  SearchResult r;
  if (auto winner = source.LiteralWinner()) {
    r.root_value = *winner == source.side_to_move() ? 1.0 : -1.0;
    return r;
  }
  Node root{source.side_to_move()};
  Expand(root, source); r.evaluations = 1;
  for (int i = 0; i < config_.simulations; ++i) {
    Board current = source; Node* node = &root;
    std::vector<Node*> nodes{&root}; std::vector<Edge*> edges;
    while (!node->edges.empty() && !current.LiteralWinner()) {
      Edge& edge = Select(*node); current.Play(edge.action); edges.push_back(&edge);
      if (!edge.child) edge.child = std::make_unique<Node>(Node{current.side_to_move()});
      node = edge.child.get(); nodes.push_back(node);
      if (node->visits == 0 || node->edges.empty()) break;
    }
    r.max_depth = std::max(r.max_depth, static_cast<int>(edges.size()));
    double leaf;
    if (auto winner = current.LiteralWinner()) leaf = *winner == current.side_to_move() ? 1.0 : -1.0;
    else { leaf = Expand(*node, current); ++r.evaluations; }
    for (Node* n : nodes) ++n->visits;
    for (auto it = edges.rbegin(); it != edges.rend(); ++it) { leaf = -leaf; ++(*it)->visits; (*it)->value_sum += leaf; }
  }
  int maximum_visits = -1;
  for (const auto& e : root.edges) {
    r.raw_visits[e.action] = e.visits; r.priors[e.action] = e.prior;
    r.raw_value_sums[e.action] = e.value_sum; r.root_visits += e.visits;
    if (e.visits > maximum_visits) { maximum_visits = e.visits; r.selected_action = e.action; }
  }
  for (int a = 0; a < kBoardArea; ++a) r.policy[a] = r.root_visits ? static_cast<double>(r.raw_visits[a]) / r.root_visits : 0;
  if (r.root_visits) { double sum = 0; for (const auto& e : root.edges) sum += e.value_sum; r.root_value = sum / r.root_visits; }
  return r;
}

Evaluation DeterministicFakeEvaluator::operator()(const Board& board) const {
  Evaluation e{}; int s = board.Signature();
  for (int a = 0; a < kBoardArea; ++a) e.policy_logits[a] = ((a * 17 + s * 31) % 37 - 18) / 8.0;
  e.value = ((s % 13) - 6) / 7.0; return e;
}

}  // namespace hex_puct
