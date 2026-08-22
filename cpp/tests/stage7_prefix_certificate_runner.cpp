#include "hex_puct/connection_certificate.hpp"
#include "hex_puct/connection_realizer.hpp"

#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

using namespace hex_puct;
namespace {
char Name(Color colour) { return colour == Color::Black ? 'B' : 'W'; }
int Bridges(const ConnectionCertificate& certificate) {
  int count = 0;
  for (const auto& segment : certificate.route)
    if (segment.kind == ConnectionSegmentKind::Bridge || segment.kind == ConnectionSegmentKind::EdgeBridge) ++count;
  return count;
}
bool RealizerSupported(const Board& board, const ConnectionCertificate& certificate) {
  if (!ValidateElementaryBridgeCertificate(board, certificate)) return false;
  try { ElementaryCertificateRealizer realizer(board, certificate); return true; }
  catch (const std::exception&) { return false; }
}
}

// Input: id|comma-separated complete physical actions. Output reports the
// first supported certificate immediately after an original physical move.
int main() {
  try {
    std::string line;
    while (std::getline(std::cin, line)) {
      const auto split = line.find('|');
      if (split == std::string::npos) throw std::runtime_error("expected id|moves input");
      const std::string id = line.substr(0, split);
      Board board; std::optional<int> certificate_ply; std::optional<Color> owner;
      int bridge_count = 0; bool validated = false, realizer_supported = false, simultaneous = false;
      std::stringstream stream(line.substr(split + 1)); std::string token; int ply = 0;
      while (std::getline(stream, token, ',')) {
        if (token.empty()) continue;
        if (board.LiteralWinner()) throw std::runtime_error("post-literal-terminal move in " + id);
        board.Play(std::stoi(token)); ++ply;
        const auto black = FindElementaryBridgeCertificate(board, Color::Black);
        const auto white = FindElementaryBridgeCertificate(board, Color::White);
        if (!certificate_ply && (black || white)) {
          simultaneous = bool(black && white);
          const auto& certificate = black ? *black : *white;
          certificate_ply = ply; owner = certificate.player; bridge_count = Bridges(certificate);
          validated = ValidateElementaryBridgeCertificate(board, certificate);
          realizer_supported = RealizerSupported(board, certificate);
        }
      }
      const auto winner = board.LiteralWinner();
      if (!winner) throw std::runtime_error("no literal terminal in " + id);
      std::cout << "{\"id\":\"" << id << "\",\"literal_terminal_ply\":" << ply
                << ",\"literal_winner\":\"" << Name(*winner) << "\",\"certificate_ply\":";
      if (certificate_ply) std::cout << *certificate_ply; else std::cout << "null";
      std::cout << ",\"certificate_owner\":";
      if (owner) std::cout << "\"" << Name(*owner) << "\""; else std::cout << "null";
      std::cout << ",\"bridge_count\":" << bridge_count << ",\"validated\":" << (validated ? "true" : "false")
                << ",\"realizer_supported\":" << (realizer_supported ? "true" : "false")
                << ",\"simultaneous\":" << (simultaneous ? "true" : "false") << "}\n";
    }
  } catch (const std::exception& error) { std::cerr << "stage7 prefix certificate error: " << error.what() << '\n'; return 1; }
}
