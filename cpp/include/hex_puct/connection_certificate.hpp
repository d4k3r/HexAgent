#pragma once

#include "hex_puct/puct.hpp"

#include <array>
#include <optional>
#include <vector>

namespace hex_puct {

// These sentinels are route endpoints, never board actions.
constexpr int kCertificateStartEdge = -1;
constexpr int kCertificateGoalEdge = -2;

enum class ConnectionSegmentKind {
  Adjacent,
  Bridge,
  EdgeAdjacent,
  EdgeBridge,
};

// A route is ordered from the player's first physical edge to its opposite
// physical edge. Bridge carriers are -1 for non-bridge segments.
struct ConnectionSegment {
  ConnectionSegmentKind kind;
  int endpoint_a;
  int endpoint_b;
  std::array<int, 2> carriers{{-1, -1}};
};

struct ConnectionCertificate {
  Color player;
  std::vector<ConnectionSegment> route;
};

// Detect only elementary, local virtual connections: stone adjacency, the
// standard two-empty-carrier bridge, and KataHex-equivalent one-bridge edge
// attachments. This is diagnostic-only and never alters Board terminal state.
[[nodiscard]] std::optional<ConnectionCertificate>
FindElementaryBridgeCertificate(const Board& board, Color player);

// Defensive independent validation for serialized/returned certificates.
[[nodiscard]] bool ValidateElementaryBridgeCertificate(
    const Board& board, const ConnectionCertificate& certificate);

}  // namespace hex_puct
