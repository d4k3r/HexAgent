#include "hex_puct/connection_certificate.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace hex_puct;

namespace {
char Name(Color color) { return color == Color::Black ? 'B' : 'W'; }
const char* Name(ConnectionSegmentKind kind) {
  switch (kind) {
    case ConnectionSegmentKind::Adjacent: return "adjacent";
    case ConnectionSegmentKind::Bridge: return "bridge";
    case ConnectionSegmentKind::EdgeAdjacent: return "edge_adjacent";
    case ConnectionSegmentKind::EdgeBridge: return "edge_bridge";
  }
  return "unknown";
}
void CertificateJson(std::ostream& out, const std::optional<ConnectionCertificate>& certificate) {
  if (!certificate) { out << "null"; return; }
  int bridges = 0;
  out << "{\"player\":\"" << Name(certificate->player) << "\",\"bridge_segments\":";
  for (const auto& segment : certificate->route)
    if (segment.kind == ConnectionSegmentKind::Bridge || segment.kind == ConnectionSegmentKind::EdgeBridge) ++bridges;
  out << bridges << ",\"route\":[";
  for (size_t i = 0; i < certificate->route.size(); ++i) {
    if (i) out << ',';
    const auto& segment = certificate->route[i];
    out << "{\"kind\":\"" << Name(segment.kind) << "\",\"a\":" << segment.endpoint_a
        << ",\"b\":" << segment.endpoint_b << ",\"carriers\":[" << segment.carriers[0]
        << ',' << segment.carriers[1] << "]}";
  }
  out << "]}";
}
}  // namespace

// Input is one line per trace: id|comma-separated physical actions.
int main() {
  try {
    std::string line;
    while (std::getline(std::cin, line)) {
      const auto split = line.find('|');
      if (split == std::string::npos) throw std::runtime_error("expected id|moves input");
      const std::string id = line.substr(0, split);
      Board board;
      std::optional<int> first_black, first_white, first_both, literal_ply;
      std::optional<ConnectionCertificate> black, white;
      std::stringstream moves(line.substr(split + 1));
      std::string token;
      int ply = 0;
      while (std::getline(moves, token, ',')) {
        if (token.empty()) continue;
        if (board.LiteralWinner()) throw std::runtime_error("post-literal-terminal move in " + id);
        board.Play(std::stoi(token));
        ++ply;
        const auto black_now = FindElementaryBridgeCertificate(board, Color::Black);
        const auto white_now = FindElementaryBridgeCertificate(board, Color::White);
        if (!black && black_now) { black = black_now; first_black = ply; }
        if (!white && white_now) { white = white_now; first_white = ply; }
        if (!first_both && black_now && white_now) first_both = ply;
        if (board.LiteralWinner()) literal_ply = ply;
      }
      if (!literal_ply) throw std::runtime_error("no literal terminal in " + id);
      std::cout << "{\"id\":\"" << id << "\",\"literal_terminal_ply\":" << *literal_ply
                << ",\"literal_winner\":\"" << Name(*board.LiteralWinner())
                << "\",\"first_black_certificate_ply\":";
      if (first_black) std::cout << *first_black; else std::cout << "null";
      std::cout << ",\"first_white_certificate_ply\":";
      if (first_white) std::cout << *first_white; else std::cout << "null";
      std::cout << ",\"first_both_certificate_ply\":";
      if (first_both) std::cout << *first_both; else std::cout << "null";
      std::cout << ",\"black_certificate\":"; CertificateJson(std::cout, black);
      std::cout << ",\"white_certificate\":"; CertificateJson(std::cout, white);
      std::cout << "}\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "bridge certificate runner error: " << error.what() << '\n';
    return 1;
  }
}
