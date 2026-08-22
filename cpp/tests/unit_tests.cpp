#include "hex_puct/puct.hpp"
#include "hex_puct/connection_certificate.hpp"
#include "hex_puct/connection_realizer.hpp"
#include "hex_puct/swap_control.hpp"
#include "hex_puct/student_boundary.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace hex_puct;
int main() {
  // FPU is explicitly configurable, while the zero-FPU default remains the
  // historical neutral first-play urgency.  The reduced parent-value form is
  // monotone in visited prior mass and is clamped to the value range.
  assert(ComputeFpuQ(FpuMode::Zero, 0.8, 0.25, 0.9) == 0.0);
  assert(std::abs(ComputeFpuQ(FpuMode::ParentValueReduced, 0.8, 0.25, 0.0) - 0.8) < 1e-12);
  assert(std::abs(ComputeFpuQ(FpuMode::ParentValueReduced, 0.8, 0.25, 1.0) - 0.55) < 1e-12);
  assert(ComputeFpuQ(FpuMode::ParentValueReduced, -0.8, 0.25, 1.0) < -0.99);
  assert(std::abs(ComputeFpuQ(FpuMode::ParentValueReduced, 0.0, 0.25, 0.0)) < 1e-12);
  assert(ComputeFpuQ(FpuMode::ParentValueReduced, 0.2, 2.0, 4.0) == -1.0);
  Board empty;
  assert(empty.LegalActions().size() == 121);
  assert(!FindElementaryBridgeCertificate(empty, Color::Black));
  assert(!FindElementaryBridgeCertificate(empty, Color::White));
  Board win = Board::FromSetup({0,11,22,33,44,55,66,77,88,99}, {1,12,23,34,45,56,67,78,89,100}, Color::Black);
  auto winning_policy = [](const Board&) { Evaluation e{}; e.policy_logits.fill(-10.0); e.policy_logits[110] = 10.0; e.value = -0.5; return e; };
  DeterministicPUCT search(winning_policy, {1});
  auto result = search.Search(win);
  assert(result.root_visits == 1);
  assert(*result.selected_action == 110 && result.raw_visits[110] == 1 && *result.root_value == 1.0);
  Board terminal = Board::FromSetup({0,11,22,33,44,55,66,77,88,99,110}, {}, Color::White);
  auto terminal_result = search.Search(terminal);
  assert(!terminal_result.selected_action && terminal_result.root_visits == 0 && *terminal_result.root_value == -1.0);
  auto a = DeterministicPUCT(DeterministicFakeEvaluator{}, {32}).Search(empty);
  auto b = DeterministicPUCT(DeterministicFakeEvaluator{}, {32}).Search(empty);
  assert(a.raw_visits == b.raw_visits && a.selected_action == b.selected_action);
  assert(a.root_visits == 32);
  auto baseline_explicit = DeterministicPUCT(DeterministicFakeEvaluator{}, SearchConfig{32, 1.5, FpuMode::Zero, 0.0}).Search(empty);
  assert(a.raw_visits == baseline_explicit.raw_visits);
  assert(a.selected_action == baseline_explicit.selected_action);
  assert(a.root_value == baseline_explicit.root_value);
  Board encoded = Board::FromSetup({0, 12}, {1, 23}, Color::Black, 23);
  auto tensor = EncodeStudentInput(encoded);
  assert(tensor[0] == 1 && tensor[12] == 1 && tensor[121 + 1] == 1 && tensor[121 + 23] == 1);
  assert(tensor[2 * 121] == 1 && tensor[3 * 121 + 23] == 1);
  assert(ActionFromRowColumn(10, 10) == 120 && RowColumnFromAction(23) == std::make_pair(2, 1));
  assert(ActionToGtp(120) == "k11");
  Evaluation fixed_output{}; fixed_output.policy_logits.fill(0.0); fixed_output.policy_logits[10] = 8.0; fixed_output.value = 0.25;
  FixedSynchronousModelEvaluator fixed_model(fixed_output);
  auto boundary_result = DeterministicPUCT(EncodedBoardEvaluator(fixed_model), {1}).Search(Board{});
  assert(boundary_result.raw_visits[10] == 1);  // PUCT received raw logits and selected after its legal softmax.
  // Pie rule is an ownership-only event, never a 122nd board/search action.
  Board opening;
  opening.Play(60);
  const auto cells_before = [&] { std::array<int, kBoardArea> x{}; for (int i=0;i<kBoardArea;++i) x[i]=opening.Cell(i); return x; }();
  const auto legal_before = opening.LegalActions();
  UniversityPieRule pie("red_controller", "blue_controller");
  assert(pie.SwapIsLegal(opening));
  assert(pie.ColorOwnedByFirstParticipant() == Color::Black);
  pie.ApplySwap(opening);
  assert(pie.swap_applied());
  assert(pie.ColorOwnedByFirstParticipant() == Color::White);
  assert(pie.ColorOwnedBySecondParticipant() == Color::Black);
  for (int i=0;i<kBoardArea;++i) assert(opening.Cell(i) == cells_before[i]);
  assert(opening.side_to_move() == Color::White);
  assert(opening.LegalActions() == legal_before && opening.LegalActions().size() == 120);
  assert(!pie.SwapIsLegal(opening));
  bool second_swap_rejected = false;
  try { pie.ApplySwap(opening); } catch (const std::invalid_argument&) { second_swap_rejected = true; }
  assert(second_swap_rejected);
  Board wrong_time;
  UniversityPieRule fresh;
  assert(!fresh.SwapIsLegal(wrong_time));
  wrong_time.Play(0); wrong_time.Play(1);
  assert(!fresh.SwapIsLegal(wrong_time));
  // Diagnostic elementary bridge certificates never change literal terminal state.
  Board black_bridge = Board::FromSetup({0,12,23,34,45,56,67,78,89,100,111}, {}, Color::White);
  assert(!black_bridge.LiteralWinner());
  const auto black_certificate = FindElementaryBridgeCertificate(black_bridge, Color::Black);
  assert(black_certificate && ValidateElementaryBridgeCertificate(black_bridge, *black_certificate));
  bool found_black_bridge = false;
  for (const auto& segment : black_certificate->route)
    if (segment.kind == ConnectionSegmentKind::Bridge && segment.endpoint_a == 0 && segment.endpoint_b == 12 &&
        segment.carriers == std::array<int,2>{{1,11}}) found_black_bridge = true;
  assert(found_black_bridge);
  // Either attack on an elementary bridge has exactly one legal paired reply.
  for (const auto [attack, reply] : {std::pair{1,11}, std::pair{11,1}}) {
    Board response_board = Board::FromSetup({0,12,23,34,45,56,67,78,89,100,111}, {}, Color::White);
    ElementaryCertificateRealizer realizer(response_board, *black_certificate);
    response_board.Play(attack);
    const auto decision = realizer.ChooseMove(response_board, attack);
    assert(decision.kind == RealizerMoveKind::PairedResponse && decision.action == reply);
    response_board.Play(*decision.action);
    assert(response_board.LiteralWinner() == Color::Black);
  }
  Board ignored_attack = Board::FromSetup({0,12,23,34,45,56,67,78,89,100,111}, {}, Color::White);
  ElementaryCertificateRealizer ignored_realizer(ignored_attack, *black_certificate);
  ignored_attack.Play(1);       // White attacks carrier 1.
  ignored_attack.Play(3);       // Black deliberately ignores the mandated 11 reply.
  ignored_attack.Play(11);      // White occupies the second carrier.
  const auto invalid_progress = ignored_realizer.Inspect(ignored_attack);
  assert(!invalid_progress.valid && invalid_progress.failure_reason == "opponent occupied both carriers before response");
  Board attacked_bridge = Board::FromSetup({0,12,23,34,45,56,67,78,89,100,111}, {1}, Color::White);
  assert(!FindElementaryBridgeCertificate(attacked_bridge, Color::Black));
  Board white_bridge = Board::FromSetup({}, {0,12,13,14,15,16,17,18,19,20,21}, Color::Black);
  assert(!white_bridge.LiteralWinner());
  const auto white_certificate = FindElementaryBridgeCertificate(white_bridge, Color::White);
  assert(white_certificate && ValidateElementaryBridgeCertificate(white_bridge, *white_certificate));
  // The supported edge analogue has the same paired-response property.
  Board edge_bridge = Board::FromSetup({11,22,33,44,55,66,77,88,99,110}, {}, Color::White);
  const auto edge_certificate = FindElementaryBridgeCertificate(edge_bridge, Color::Black);
  assert(edge_certificate && ValidateElementaryBridgeCertificate(edge_bridge, *edge_certificate));
  size_t edge_index = 0; while (edge_index < edge_certificate->route.size() &&
      edge_certificate->route[edge_index].kind != ConnectionSegmentKind::EdgeBridge) ++edge_index;
  assert(edge_index < edge_certificate->route.size());
  for (const auto [attack, reply] : {std::pair{0,1}, std::pair{1,0}}) {
    Board response_board = Board::FromSetup({11,22,33,44,55,66,77,88,99,110}, {}, Color::White);
    ElementaryCertificateRealizer realizer(response_board, *edge_certificate);
    response_board.Play(attack);
    const auto decision = realizer.ChooseMove(response_board, attack);
    assert(decision.kind == RealizerMoveKind::PairedResponse && decision.action == reply);
    response_board.Play(*decision.action);
    assert(CertificateSegmentLiterallyResolved(response_board, *edge_certificate, edge_index));
    assert(response_board.LiteralWinner() == Color::Black);
  }
  // A bridge certificate cannot silently reuse a carrier, even where each
  // individual jump is geometrically valid.
  Board overlap = Board::FromSetup({1,13,22,33,44,55,66,77,88,99,110}, {}, Color::White);
  ConnectionCertificate reused{Color::Black, {
      {ConnectionSegmentKind::EdgeAdjacent,kCertificateStartEdge,1},
      {ConnectionSegmentKind::Bridge,1,13,{{2,12}}},
      {ConnectionSegmentKind::Bridge,13,22,{{23,12}}},
      {ConnectionSegmentKind::Adjacent,22,33}, {ConnectionSegmentKind::Adjacent,33,44},
      {ConnectionSegmentKind::Adjacent,44,55}, {ConnectionSegmentKind::Adjacent,55,66},
      {ConnectionSegmentKind::Adjacent,66,77}, {ConnectionSegmentKind::Adjacent,77,88},
      {ConnectionSegmentKind::Adjacent,88,99}, {ConnectionSegmentKind::Adjacent,99,110},
      {ConnectionSegmentKind::EdgeAdjacent,110,kCertificateGoalEdge}}};
  assert(!ValidateElementaryBridgeCertificate(overlap, reused));
  Board literal_black = Board::FromSetup({0,11,22,33,44,55,66,77,88,99,110}, {}, Color::White);
  assert(literal_black.LiteralWinner() == Color::Black);
  const auto literal_certificate = FindElementaryBridgeCertificate(literal_black, Color::Black);
  assert(literal_certificate && ValidateElementaryBridgeCertificate(literal_black, *literal_certificate));
  assert(literal_black.LiteralWinner() == Color::Black);  // Detector is non-mutating.
  Board literal_white = Board::FromSetup({}, {0,1,2,3,4,5,6,7,8,9,10}, Color::Black);
  assert(literal_white.LiteralWinner() == Color::White);
  const auto literal_white_certificate = FindElementaryBridgeCertificate(literal_white, Color::White);
  assert(literal_white_certificate && ValidateElementaryBridgeCertificate(literal_white, *literal_white_certificate));
  // A carrier-disjoint multi-bridge route survives sequential adversarial
  // attacks when every attack receives its mandated paired reply.
  std::vector<int> diagonal; for (int i=0;i<kBoardSize;++i) diagonal.push_back(i*12);
  Board multi_bridge = Board::FromSetup(diagonal, {}, Color::White);
  const auto multi_certificate = FindElementaryBridgeCertificate(multi_bridge, Color::Black);
  assert(multi_certificate && ValidateElementaryBridgeCertificate(multi_bridge, *multi_certificate));
  for (size_t i=0;i<multi_certificate->route.size();++i) {
    const auto& segment=multi_certificate->route[i];
    if (segment.kind != ConnectionSegmentKind::Bridge && segment.kind != ConnectionSegmentKind::EdgeBridge) continue;
    for (int attack : segment.carriers) {
      Board direct_attack = Board::FromSetup(diagonal, {}, Color::White);
      ElementaryCertificateRealizer direct_realizer(direct_attack, *multi_certificate);
      direct_attack.Play(attack);
      const auto reply = direct_realizer.ChooseMove(direct_attack, attack);
      assert(reply.kind == RealizerMoveKind::PairedResponse && reply.action);
      direct_attack.Play(*reply.action);
      assert(CertificateSegmentLiterallyResolved(direct_attack, *multi_certificate, i));
    }
  }
  ElementaryCertificateRealizer multi_realizer(multi_bridge, *multi_certificate);
  int attacks = 0, responses = 0;
  while (!multi_bridge.LiteralWinner()) {
    const auto progress = multi_realizer.Inspect(multi_bridge);
    assert(progress.valid);
    if (multi_bridge.side_to_move() == Color::White) {
      assert(!progress.attacked_bridge_segments.empty() || progress.unresolved_bridges > 0);
      size_t index = progress.attacked_bridge_segments.empty() ? 0 : progress.attacked_bridge_segments.front();
      if (progress.attacked_bridge_segments.empty()) while (index < multi_certificate->route.size()) {
        const auto& candidate = multi_certificate->route[index];
        const bool bridge = candidate.kind == ConnectionSegmentKind::Bridge || candidate.kind == ConnectionSegmentKind::EdgeBridge;
        if (bridge && multi_bridge.LegalMask()[candidate.carriers[0]]) break;
        ++index;
      }
      assert(index < multi_certificate->route.size());
      const auto& segment = multi_certificate->route[index];
      const int attack = segment.carriers[0];
      assert(multi_bridge.LegalMask()[attack]); multi_bridge.Play(attack); ++attacks;
    } else {
      const auto decision = multi_realizer.ChooseMove(multi_bridge, multi_bridge.last_move());
      assert(decision.kind == RealizerMoveKind::PairedResponse && decision.action);
      multi_bridge.Play(*decision.action); ++responses;
    }
  }
  assert(multi_bridge.LiteralWinner() == Color::Black && attacks == responses);
  std::cout << "hex_puct_unit_tests passed\n";
}
