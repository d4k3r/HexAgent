#include "hex_puct/connection_realizer.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace hex_puct;

namespace {
char Name(Color color) { return color == Color::Black ? 'B' : 'W'; }
const char* Name(RealizerMoveKind kind) {
  switch (kind) {
    case RealizerMoveKind::PairedResponse: return "paired_response";
    case RealizerMoveKind::ProactiveResolution: return "proactive_resolution";
    case RealizerMoveKind::Complete: return "complete";
    case RealizerMoveKind::Invalid: return "invalid";
  }
  return "unknown";
}
bool Owner(const Board& board, int action, Color player) {
  return board.Cell(action) == (player == Color::Black ? 1 : 2);
}
bool Opponent(const Board& board, int action, Color player) {
  return board.Cell(action) == (player == Color::Black ? 2 : 1);
}
bool Bridge(const ConnectionSegment& s) {
  return s.kind == ConnectionSegmentKind::Bridge || s.kind == ConnectionSegmentKind::EdgeBridge;
}
bool CarrierAttack(const Board& board, const ConnectionCertificate& c, int action) {
  for (const auto& s : c.route) if (Bridge(s) && (s.carriers[0] == action || s.carriers[1] == action) &&
      !Owner(board, s.carriers[0], c.player) && !Owner(board, s.carriers[1], c.player)) return true;
  return false;
}
int FallbackAdversaryMove(const Board& board, const ElementaryCertificateRealizer& realizer,
                          const ConnectionCertificate& c) {
  for (const auto& s : c.route) if (Bridge(s)) {
    const bool a_empty = board.LegalMask()[s.carriers[0]], b_empty = board.LegalMask()[s.carriers[1]];
    if (a_empty && b_empty) return std::min(s.carriers[0], s.carriers[1]);
  }
  auto legal = board.LegalActions();
  if (legal.empty()) throw std::runtime_error("no legal adversarial fallback");
  // Among non-immediate attacks, maximize the certificate's remaining
  // unresolved-bridge count; use highest action only as a deterministic tie.
  int best = -1, best_unresolved = -1;
  for (int action : legal) {
    Board probe = board; probe.Play(action);
    const auto progress = realizer.Inspect(probe);
    if (progress.valid && (progress.unresolved_bridges > best_unresolved ||
        (progress.unresolved_bridges == best_unresolved && action > best))) {
      best = action; best_unresolved = progress.unresolved_bridges;
    }
  }
  return best >= 0 ? best : legal.back();
}
void CertificateJson(std::ostream& out, const ConnectionCertificate& c) {
  out << "{\"player\":\"" << Name(c.player) << "\",\"route\":[";
  for (size_t i=0;i<c.route.size();++i) { if(i)out<<','; const auto&s=c.route[i];
    out<<"{\"kind\":"<<static_cast<int>(s.kind)<<",\"a\":"<<s.endpoint_a<<",\"b\":"<<s.endpoint_b
       <<",\"carriers\":["<<s.carriers[0]<<','<<s.carriers[1]<<"]}"; }
  out << "]}";
}
struct AttackEvent { int ply, action; std::optional<int> prescribed_response, actual_owner_move; };
struct HistoricalEvent { std::vector<AttackEvent> attacks; std::optional<int> invalid_ply; std::string invalid_reason; };
HistoricalEvent ReplayHistorical(const std::vector<int>& moves, int certificate_ply, const Board& at_certificate,
                                 const ConnectionCertificate& certificate) {
  Board board=at_certificate; ElementaryCertificateRealizer r(board,certificate); HistoricalEvent e;
  for (size_t index=static_cast<size_t>(certificate_ply); index<moves.size();++index) {
    const int action=moves[index]; const Color turn=board.side_to_move();
    const bool attack = turn != certificate.player && CarrierAttack(board,certificate,action);
    board.Play(action); const int ply=static_cast<int>(index)+1;
    if (attack) {
      AttackEvent event{ply, action};
      auto decision=r.ChooseMove(board,action); if(decision.action)event.prescribed_response=*decision.action;
      e.attacks.push_back(event);
    }
    if (turn==certificate.player && !e.attacks.empty() && !e.attacks.back().actual_owner_move)
      e.attacks.back().actual_owner_move=action;
    const auto progress=r.Inspect(board);
    if (!progress.valid && !e.invalid_ply) { e.invalid_ply=ply; e.invalid_reason=progress.failure_reason; }
    if (board.LiteralWinner()) break;
  }
  return e;
}
struct Counterfactual { bool success=false; int literal_ply=0, owner_moves=0, attacks=0, responses=0, proactive=0, fallback=0; std::string failure; };
Counterfactual ReplayCounterfactual(const std::vector<int>& moves, int certificate_ply, const Board& at_certificate,
                                    const ConnectionCertificate& certificate) {
  Board board=at_certificate; ElementaryCertificateRealizer r(board,certificate); Counterfactual out; std::optional<int> last_opponent;
  for (size_t historical=static_cast<size_t>(certificate_ply); historical < static_cast<size_t>(kBoardArea); ++historical) {
    if (board.LiteralWinner()) { out.success=*board.LiteralWinner()==certificate.player; out.literal_ply=certificate_ply+static_cast<int>(historical)-certificate_ply; return out; }
    int action=-1;
    if (board.side_to_move()!=certificate.player) {
      if (historical<moves.size() && board.LegalMask()[moves[historical]]) action=moves[historical];
      else { action=FallbackAdversaryMove(board,r,certificate); ++out.fallback; }
      if (CarrierAttack(board,certificate,action)) ++out.attacks;
      board.Play(action); last_opponent=action;
    } else {
      const auto decision=r.ChooseMove(board,last_opponent);
      if (!decision.action) { out.failure=decision.reason.empty()?Name(decision.kind):decision.reason; return out; }
      if (decision.kind==RealizerMoveKind::PairedResponse) ++out.responses;
      if (decision.kind==RealizerMoveKind::ProactiveResolution) ++out.proactive;
      board.Play(*decision.action); ++out.owner_moves;
    }
  }
  if (board.LiteralWinner()) { out.success=*board.LiteralWinner()==certificate.player; out.literal_ply=kBoardArea; }
  else out.failure="counterfactual exhausted board limit";
  return out;
}
void Maybe(std::ostream& out, const std::optional<int>& x) { if(x)out<<*x;else out<<"null"; }
void AttacksJson(std::ostream& out, const std::vector<AttackEvent>& attacks) {
  out << '['; for(size_t i=0;i<attacks.size();++i) { if(i)out<<','; const auto& e=attacks[i];
    out<<"{\"ply\":"<<e.ply<<",\"action\":"<<e.action<<",\"prescribed_response\":";Maybe(out,e.prescribed_response);
    out<<",\"actual_owner_move\":";Maybe(out,e.actual_owner_move);out<<'}'; } out<<']';
}
}  // namespace

// Input is one line: id|eventual literal winner (B/W)|comma-separated moves.
int main() {
  try {
    std::string line;
    while(std::getline(std::cin,line)) {
      const auto a=line.find('|'), b=line.find('|',a==std::string::npos?0:a+1);
      if(a==std::string::npos||b==std::string::npos)throw std::runtime_error("expected id|winner|moves");
      const std::string id=line.substr(0,a); const Color winner=line.substr(a+1,b-a-1)=="B"?Color::Black:Color::White;
      std::vector<int> moves; std::stringstream in(line.substr(b+1));std::string token;while(std::getline(in,token,','))if(!token.empty())moves.push_back(std::stoi(token));
      Board board; std::optional<ConnectionCertificate> cert; int certificate_ply=0;
      for(size_t i=0;i<moves.size();++i) { if(board.LiteralWinner())throw std::runtime_error("post terminal historical move"); board.Play(moves[i]);
        if(!cert) { auto candidate=FindElementaryBridgeCertificate(board,winner); if(candidate) {cert=*candidate;certificate_ply=int(i)+1;break;} } }
      if(!cert)throw std::runtime_error("eventual winner has no certificate");
      const Board at_certificate=board; const auto historical=ReplayHistorical(moves,certificate_ply,at_certificate,*cert);
      const auto cf=ReplayCounterfactual(moves,certificate_ply,at_certificate,*cert);
      Board final_board; for(int action:moves)final_board.Play(action);
      std::cout<<"{\"id\":\""<<id<<"\",\"winner\":\""<<Name(winner)<<"\",\"certificate_ply\":"<<certificate_ply
        <<",\"historical_literal_ply\":"<<moves.size()<<",\"counterfactual_success\":"<<(cf.success?"true":"false")
        <<",\"counterfactual_literal_ply\":"<<cf.literal_ply<<",\"new_tail\":"<<(cf.success?cf.literal_ply-certificate_ply:-1)
        <<",\"owner_realization_moves\":"<<cf.owner_moves<<",\"opponent_carrier_attacks\":"<<cf.attacks
        <<",\"successful_paired_responses\":"<<cf.responses<<",\"proactive_resolutions\":"<<cf.proactive<<",\"fallback_moves\":"<<cf.fallback
        <<",\"failure\":\""<<cf.failure<<"\",\"historical_attacks\":";AttacksJson(std::cout,historical.attacks);
      std::cout<<",\"historical_first_invalid_ply\":";Maybe(std::cout,historical.invalid_ply);
      std::cout<<",\"historical_invalid_reason\":\""<<historical.invalid_reason<<"\",\"certificate\":";CertificateJson(std::cout,*cert);std::cout<<"}\n";
    }
  } catch(const std::exception& e) { std::cerr<<"certificate realizer runner error: "<<e.what()<<'\n';return 1; }
}
