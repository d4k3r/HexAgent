#include "hex_puct/connection_realizer.hpp"

#include <algorithm>
#include <stdexcept>

namespace hex_puct {
namespace {
bool OwnerStone(const Board& board, int action, Color owner) {
  return action >= 0 && action < kBoardArea && board.Cell(action) == (owner == Color::Black ? 1 : 2);
}
bool OpponentStone(const Board& board, int action, Color owner) {
  return action >= 0 && action < kBoardArea && board.Cell(action) == (owner == Color::Black ? 2 : 1);
}
bool BridgeSegment(const ConnectionSegment& segment) {
  return segment.kind == ConnectionSegmentKind::Bridge || segment.kind == ConnectionSegmentKind::EdgeBridge;
}
bool SegmentEndpointsRemainOwned(const Board& board, const ConnectionSegment& segment, Color owner) {
  if (segment.endpoint_a >= 0 && !OwnerStone(board, segment.endpoint_a, owner)) return false;
  if (segment.endpoint_b >= 0 && !OwnerStone(board, segment.endpoint_b, owner)) return false;
  return true;
}
}  // namespace

ElementaryCertificateRealizer::ElementaryCertificateRealizer(const Board& initial_board,
                                                             ConnectionCertificate certificate)
    : certificate_(std::move(certificate)) {
  if (!ValidateElementaryBridgeCertificate(initial_board, certificate_))
    throw std::invalid_argument("realizer requires a valid initial elementary certificate");
}

bool CertificateSegmentLiterallyResolved(const Board& board, const ConnectionCertificate& certificate,
                                         size_t segment_index) {
  if (segment_index >= certificate.route.size()) return false;
  const auto& segment = certificate.route[segment_index];
  if (!SegmentEndpointsRemainOwned(board, segment, certificate.player)) return false;
  if (!BridgeSegment(segment)) return true;
  return OwnerStone(board, segment.carriers[0], certificate.player) ||
      OwnerStone(board, segment.carriers[1], certificate.player);
}

CertificateProgress ElementaryCertificateRealizer::Inspect(const Board& board) const {
  CertificateProgress progress{.valid = true};
  for (size_t i = 0; i < certificate_.route.size(); ++i) {
    const auto& segment = certificate_.route[i];
    if (!SegmentEndpointsRemainOwned(board, segment, certificate_.player)) {
      progress.valid = false;
      progress.failure_reason = "certificate endpoint no longer belongs to owner";
      return progress;
    }
    if (!BridgeSegment(segment)) continue;
    const int a = segment.carriers[0], b = segment.carriers[1];
    if (a < 0 || b < 0 || a == b) {
      progress.valid = false;
      progress.failure_reason = "malformed bridge carriers";
      return progress;
    }
    const bool own_a = OwnerStone(board, a, certificate_.player);
    const bool own_b = OwnerStone(board, b, certificate_.player);
    const bool opp_a = OpponentStone(board, a, certificate_.player);
    const bool opp_b = OpponentStone(board, b, certificate_.player);
    // A single owner carrier literally joins both bridge endpoints (or an
    // endpoint to its physical edge). The other carrier may subsequently be
    // occupied by either colour without breaking that realized connection.
    if (own_a || own_b) continue;
    if (opp_a && opp_b) {
      progress.valid = false;
      progress.failure_reason = "opponent occupied both carriers before response";
      return progress;
    }
    if (opp_a || opp_b) progress.attacked_bridge_segments.push_back(i);
    ++progress.unresolved_bridges;
  }
  progress.literal_route_complete = progress.unresolved_bridges == 0;
  return progress;
}

RealizerDecision ElementaryCertificateRealizer::ChooseMove(
    const Board& board, std::optional<int> opponent_last_move) const {
  const auto progress = Inspect(board);
  if (!progress.valid) return {RealizerMoveKind::Invalid, std::nullopt, std::nullopt, progress.failure_reason};
  if (board.side_to_move() != certificate_.player)
    return {RealizerMoveKind::Invalid, std::nullopt, std::nullopt, "not certificate owner's turn"};
  if (progress.literal_route_complete)
    return {RealizerMoveKind::Complete, std::nullopt, std::nullopt, "all bridge segments are literal"};
  // A current single-carrier attack is mandatory. Prefer the supplied latest
  // move, while still fail-closing to another detected attack if supplied
  // provenance is absent or stale.
  for (size_t index : progress.attacked_bridge_segments) {
    const auto& segment = certificate_.route[index];
    const int a = segment.carriers[0], b = segment.carriers[1];
    const int response = OpponentStone(board, a, certificate_.player) ? b : a;
    if (!board.LegalMask()[response])
      return {RealizerMoveKind::Invalid, std::nullopt, index, "paired response carrier is not legal"};
    if (!opponent_last_move || *opponent_last_move == a || *opponent_last_move == b)
      return {RealizerMoveKind::PairedResponse, response, index, "answer carrier attack"};
  }
  if (!progress.attacked_bridge_segments.empty()) {
    const size_t index = progress.attacked_bridge_segments.front();
    const auto& segment = certificate_.route[index];
    const int response = OpponentStone(board, segment.carriers[0], certificate_.player)
        ? segment.carriers[1] : segment.carriers[0];
    if (!board.LegalMask()[response])
      return {RealizerMoveKind::Invalid, std::nullopt, index, "deferred paired response carrier is not legal"};
    return {RealizerMoveKind::PairedResponse, response, index, "answer earlier carrier attack"};
  }
  for (size_t i = 0; i < certificate_.route.size(); ++i) {
    const auto& segment = certificate_.route[i];
    if (!BridgeSegment(segment)) continue;
    const int a = segment.carriers[0], b = segment.carriers[1];
    const int action = std::min(a, b);
    if (!board.LegalMask()[action]) continue;
    return {RealizerMoveKind::ProactiveResolution, action, i, "proactively realize bridge"};
  }
  return {RealizerMoveKind::Invalid, std::nullopt, std::nullopt, "no legal unresolved bridge carrier"};
}

}  // namespace hex_puct
