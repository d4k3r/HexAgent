#pragma once

// Pie-rule control is deliberately outside the 121-cell physical PUCT API.
#include "hex_puct/puct.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace hex_puct {

enum class SwapChoice { Swap, Keep, Uncertain };

// This is diagnostic data, not a game rule or a threshold policy.  A caller
// may choose how to turn an evaluation into SwapChoice without changing board
// or PUCT semantics.
struct SwapDecision {
  SwapChoice choice = SwapChoice::Uncertain;
  std::string backend;
  std::string evaluation_perspective;
  std::optional<double> scalar_evaluation;
  std::optional<int> search_budget;
  std::optional<int> actual_visits;
  bool converged = false;
  std::string note;
};

class SwapDecisionBackend {
 public:
  virtual ~SwapDecisionBackend() = default;
  virtual SwapDecision EvaluateOneStoneWhiteToMove(const Board& physical_board) const = 0;
};

// University pie rule.  It owns participant-to-physical-colour control only;
// it never mutates a Board.  Therefore swap cannot be encoded as action 121,
// cannot advance physical ply, and cannot transpose/recolour the opening.
class UniversityPieRule {
 public:
  explicit UniversityPieRule(std::string first_participant = "player1",
                             std::string second_participant = "player2")
      : first_(std::move(first_participant)), second_(std::move(second_participant)) {}

  [[nodiscard]] bool swap_applied() const { return swap_applied_; }
  [[nodiscard]] Color ColorOwnedByFirstParticipant() const {
    return swap_applied_ ? Color::White : Color::Black;
  }
  [[nodiscard]] Color ColorOwnedBySecondParticipant() const {
    return Opponent(ColorOwnedByFirstParticipant());
  }
  [[nodiscard]] const std::string& first_participant() const { return first_; }
  [[nodiscard]] const std::string& second_participant() const { return second_; }

  [[nodiscard]] bool SwapIsLegal(const Board& board) const {
    if (swap_applied_ || board.side_to_move() != Color::White || board.LiteralWinner()) return false;
    int occupied = 0;
    int black = 0;
    int white = 0;
    for (int action = 0; action < kBoardArea; ++action) {
      const int cell = board.Cell(action);
      occupied += cell != 0;
      black += cell == 1;
      white += cell == 2;
    }
    return occupied == 1 && black == 1 && white == 0;
  }

  void ApplySwap(const Board& board) {
    if (!SwapIsLegal(board)) throw std::invalid_argument("swap is legal only after one physical black opening");
    swap_applied_ = true;
  }

 private:
  std::string first_;
  std::string second_;
  bool swap_applied_ = false;
};

}  // namespace hex_puct
