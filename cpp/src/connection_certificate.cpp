#include "hex_puct/connection_certificate.hpp"

#include <algorithm>
#include <bitset>
#include <functional>

namespace hex_puct {
namespace {
constexpr std::array<int, 6> kDx{{0, 1, 1, 0, -1, -1}};
constexpr std::array<int, 6> kDy{{-1, -1, 0, 1, 1, 0}};
constexpr std::array<int, 6> kJumpDx{{1, 2, 1, -1, -2, -1}};
constexpr std::array<int, 6> kJumpDy{{-2, -1, 1, 2, 1, -1}};

bool OnBoard(int x, int y) {
  return x >= 0 && x < kBoardSize && y >= 0 && y < kBoardSize;
}
int Action(int x, int y) { return y * kBoardSize + x; }
int X(int action) { return action % kBoardSize; }
int Y(int action) { return action / kBoardSize; }
bool IsStone(const Board& board, int action, Color player) {
  return action >= 0 && action < kBoardArea &&
      board.Cell(action) == (player == Color::Black ? 1 : 2);
}
bool Empty(const Board& board, int action) {
  return action >= 0 && action < kBoardArea && board.Cell(action) == 0;
}
bool IsStartEdgeStone(int action, Color player) {
  return player == Color::Black ? Y(action) == 0 : X(action) == 0;
}
bool IsGoalEdgeStone(int action, Color player) {
  return player == Color::Black ? Y(action) == kBoardSize - 1 : X(action) == kBoardSize - 1;
}

std::optional<std::array<int, 2>> BridgeCarriers(int from, int to) {
  const int x = X(from), y = Y(from);
  for (int d = 0; d < 6; ++d) {
    if (x + kJumpDx[d] != X(to) || y + kJumpDy[d] != Y(to)) continue;
    const int c0x = x + kDx[d], c0y = y + kDy[d];
    const int c1x = x + kDx[(d + 1) % 6], c1y = y + kDy[(d + 1) % 6];
    if (!OnBoard(c0x, c0y) || !OnBoard(c1x, c1y)) return std::nullopt;
    return std::array<int, 2>{{Action(c0x, c0y), Action(c1x, c1y)}};
  }
  return std::nullopt;
}
bool Adjacent(int a, int b) {
  const int x = X(a), y = Y(a);
  for (int d = 0; d < 6; ++d)
    if (x + kDx[d] == X(b) && y + kDy[d] == Y(b)) return true;
  return false;
}
ConnectionSegment EdgeAdjacent(int edge, int stone) {
  return {ConnectionSegmentKind::EdgeAdjacent, edge, stone};
}
ConnectionSegment EdgeBridge(int edge, int stone, int c0, int c1) {
  return {ConnectionSegmentKind::EdgeBridge, edge, stone, {{c0, c1}}};
}
std::vector<ConnectionSegment> StartLinks(const Board& board, Color player) {
  std::vector<ConnectionSegment> out;
  for (int a = 0; a < kBoardArea; ++a)
    if (IsStone(board, a, player) && IsStartEdgeStone(a, player))
      out.push_back(EdgeAdjacent(kCertificateStartEdge, a));
  // The two empty carriers are the two edge cells that protect a stone one
  // row/column from its goal edge. This is the minimal edge analogue used by
  // the audited KataHex routine, written here in physical coordinates.
  if (player == Color::Black) {
    for (int x = 0; x < kBoardSize - 1; ++x) {
      const int stone = Action(x, 1), c0 = Action(x, 0), c1 = Action(x + 1, 0);
      if (IsStone(board, stone, player) && Empty(board, c0) && Empty(board, c1))
        out.push_back(EdgeBridge(kCertificateStartEdge, stone, c0, c1));
    }
  } else {
    for (int y = 0; y < kBoardSize - 1; ++y) {
      const int stone = Action(1, y), c0 = Action(0, y), c1 = Action(0, y + 1);
      if (IsStone(board, stone, player) && Empty(board, c0) && Empty(board, c1))
        out.push_back(EdgeBridge(kCertificateStartEdge, stone, c0, c1));
    }
  }
  return out;
}
std::vector<ConnectionSegment> GoalLinks(const Board& board, Color player, int from) {
  std::vector<ConnectionSegment> out;
  if (IsGoalEdgeStone(from, player)) out.push_back(EdgeAdjacent(from, kCertificateGoalEdge));
  if (player == Color::Black) {
    const int x = X(from), y = Y(from);
    if (y == kBoardSize - 2 && x >= 1) {
      const int c0 = Action(x, kBoardSize - 1), c1 = Action(x - 1, kBoardSize - 1);
      if (Empty(board, c0) && Empty(board, c1))
        out.push_back(EdgeBridge(from, kCertificateGoalEdge, c0, c1));
    }
  } else {
    const int x = X(from), y = Y(from);
    if (x == kBoardSize - 2 && y >= 1) {
      const int c0 = Action(kBoardSize - 1, y), c1 = Action(kBoardSize - 1, y - 1);
      if (Empty(board, c0) && Empty(board, c1))
        out.push_back(EdgeBridge(from, kCertificateGoalEdge, c0, c1));
    }
  }
  return out;
}
bool SegmentUsesAvailableCarriers(const ConnectionSegment& segment, const std::bitset<kBoardArea>& used) {
  for (int carrier : segment.carriers)
    if (carrier >= 0 && used.test(static_cast<size_t>(carrier))) return false;
  return true;
}
void MarkCarriers(const ConnectionSegment& segment, std::bitset<kBoardArea>& used, bool set) {
  for (int carrier : segment.carriers)
    if (carrier >= 0) used.set(static_cast<size_t>(carrier), set);
}
}  // namespace

bool ValidateElementaryBridgeCertificate(const Board& board, const ConnectionCertificate& certificate) {
  if (certificate.route.empty()) return false;
  int current = kCertificateStartEdge;
  std::bitset<kBoardArea> used_carriers;
  std::bitset<kBoardArea> seen_stones;
  for (const auto& segment : certificate.route) {
    if (segment.endpoint_a != current) return false;
    const int next = segment.endpoint_b;
    const auto both_stones = [&] { return IsStone(board, current, certificate.player) && IsStone(board, next, certificate.player); };
    if (segment.kind == ConnectionSegmentKind::Adjacent) {
      if (!both_stones() || !Adjacent(current, next) || segment.carriers[0] >= 0 || segment.carriers[1] >= 0) return false;
    } else if (segment.kind == ConnectionSegmentKind::Bridge) {
      const auto expected = both_stones() ? BridgeCarriers(current, next) : std::nullopt;
      if (!expected || *expected != segment.carriers || expected->at(0) == expected->at(1) ||
          !Empty(board, expected->at(0)) || !Empty(board, expected->at(1))) return false;
    } else if (segment.kind == ConnectionSegmentKind::EdgeAdjacent) {
      if (!((current == kCertificateStartEdge && IsStartEdgeStone(next, certificate.player)) ||
            (next == kCertificateGoalEdge && IsGoalEdgeStone(current, certificate.player))) ||
          !IsStone(board, current == kCertificateStartEdge ? next : current, certificate.player) ||
          segment.carriers[0] >= 0 || segment.carriers[1] >= 0) return false;
    } else if (segment.kind == ConnectionSegmentKind::EdgeBridge) {
      const int stone = current == kCertificateStartEdge ? next : current;
      if (!IsStone(board, stone, certificate.player) || segment.carriers[0] < 0 || segment.carriers[1] < 0 ||
          segment.carriers[0] == segment.carriers[1] || !Empty(board, segment.carriers[0]) || !Empty(board, segment.carriers[1])) return false;
      const auto valid = current == kCertificateStartEdge ? StartLinks(board, certificate.player) : GoalLinks(board, certificate.player, current);
      if (std::find_if(valid.begin(), valid.end(), [&](const auto& s) { return s.kind == segment.kind && s.endpoint_a == segment.endpoint_a && s.endpoint_b == segment.endpoint_b && s.carriers == segment.carriers; }) == valid.end()) return false;
    } else return false;
    if (!SegmentUsesAvailableCarriers(segment, used_carriers)) return false;
    MarkCarriers(segment, used_carriers, true);
    if (next >= 0) {
      if (seen_stones.test(static_cast<size_t>(next))) return false;
      seen_stones.set(static_cast<size_t>(next));
    }
    current = next;
  }
  return current == kCertificateGoalEdge;
}

std::optional<ConnectionCertificate> FindElementaryBridgeCertificate(const Board& board, Color player) {
  ConnectionCertificate certificate{player, {}};
  std::bitset<kBoardArea> seen_stones, used_carriers;
  std::function<bool(int)> dfs = [&](int current) {
    std::vector<ConnectionSegment> links;
    if (current == kCertificateStartEdge) links = StartLinks(board, player);
    else {
      for (int d = 0; d < 6; ++d) {
        const int nx = X(current) + kDx[d], ny = Y(current) + kDy[d];
        if (OnBoard(nx, ny) && IsStone(board, Action(nx, ny), player))
          links.push_back({ConnectionSegmentKind::Adjacent, current, Action(nx, ny)});
      }
      for (int d = 0; d < 6; ++d) {
        const int nx = X(current) + kJumpDx[d], ny = Y(current) + kJumpDy[d];
        if (!OnBoard(nx, ny)) continue;
        const int target = Action(nx, ny);
        const auto carriers = BridgeCarriers(current, target);
        if (carriers && IsStone(board, target, player) && Empty(board, carriers->at(0)) && Empty(board, carriers->at(1)))
          links.push_back({ConnectionSegmentKind::Bridge, current, target, *carriers});
      }
      auto goals = GoalLinks(board, player, current);
      links.insert(links.end(), goals.begin(), goals.end());
    }
    for (const auto& segment : links) {
      const int next = segment.endpoint_b;
      if (!SegmentUsesAvailableCarriers(segment, used_carriers)) continue;
      if (next >= 0 && seen_stones.test(static_cast<size_t>(next))) continue;
      certificate.route.push_back(segment);
      MarkCarriers(segment, used_carriers, true);
      if (next >= 0) seen_stones.set(static_cast<size_t>(next));
      if (next == kCertificateGoalEdge || dfs(next)) return true;
      if (next >= 0) seen_stones.set(static_cast<size_t>(next), false);
      MarkCarriers(segment, used_carriers, false);
      certificate.route.pop_back();
    }
    return false;
  };
  if (!dfs(kCertificateStartEdge)) return std::nullopt;
  return ValidateElementaryBridgeCertificate(board, certificate) ? std::optional<ConnectionCertificate>(certificate) : std::nullopt;
}

}  // namespace hex_puct
