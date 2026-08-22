#pragma once

#include "hex_puct/connection_certificate.hpp"

#include <optional>
#include <string>
#include <vector>

namespace hex_puct {

struct CertificateProgress {
  bool valid = false;
  bool literal_route_complete = false;
  int unresolved_bridges = 0;
  std::vector<size_t> attacked_bridge_segments;
  std::string failure_reason;
};

enum class RealizerMoveKind { PairedResponse, ProactiveResolution, Complete, Invalid };

struct RealizerDecision {
  RealizerMoveKind kind = RealizerMoveKind::Invalid;
  std::optional<int> action;
  std::optional<size_t> segment_index;
  std::string reason;
};

// Offline controller for a certificate that was valid on `initial_board`.
// It never changes Board terminal semantics: callers must apply returned moves
// normally and may use progress only as diagnostic evidence.
class ElementaryCertificateRealizer {
 public:
  ElementaryCertificateRealizer(const Board& initial_board, ConnectionCertificate certificate);

  [[nodiscard]] const ConnectionCertificate& certificate() const { return certificate_; }
  [[nodiscard]] CertificateProgress Inspect(const Board& board) const;
  [[nodiscard]] RealizerDecision ChooseMove(const Board& board,
                                             std::optional<int> opponent_last_move = std::nullopt) const;

 private:
  ConnectionCertificate certificate_;
};

// True when this single route segment has become a physical connection under
// the certificate's original, already-validated geometry.
[[nodiscard]] bool CertificateSegmentLiterallyResolved(
    const Board& board, const ConnectionCertificate& certificate, size_t segment_index);

}  // namespace hex_puct
