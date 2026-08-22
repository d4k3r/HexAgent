#pragma once
#include "hex_puct/puct.hpp"
#include <array>
#include <vector>
#include <string>

namespace hex_puct {
constexpr int kStudentPlanes = 6;
using StudentTensor = std::array<int, kStudentPlanes * kBoardArea>;  // [plane][row][column]
int ActionFromRowColumn(int row, int column);
std::pair<int, int> RowColumnFromAction(int action);
std::string ActionToGtp(int action);
StudentTensor EncodeStudentInput(const Board& board);

// Deliberately synchronous ABI for a future qualified model runtime. Outputs
// are raw physical logits and a side-to-move value; PUCT owns legal softmax.
class SynchronousModelEvaluator {
 public:
  virtual ~SynchronousModelEvaluator() = default;
  virtual Evaluation Evaluate(const StudentTensor& input) const = 0;
};

// Ordered batch boundary. Like the single-position boundary it exposes raw
// physical logits and side-to-move values; it performs neither masking nor
// policy normalization.
class BatchedModelEvaluator {
 public:
  virtual ~BatchedModelEvaluator() = default;
  virtual std::vector<Evaluation> EvaluateBatch(const std::vector<StudentTensor>& input) const = 0;
};

class FixedSynchronousModelEvaluator final : public SynchronousModelEvaluator {
 public:
  explicit FixedSynchronousModelEvaluator(Evaluation evaluation) : evaluation_(std::move(evaluation)) {}
  Evaluation Evaluate(const StudentTensor&) const override { return evaluation_; }
 private:
  Evaluation evaluation_;
};

class EncodedBoardEvaluator {
 public:
  explicit EncodedBoardEvaluator(const SynchronousModelEvaluator& evaluator) : evaluator_(evaluator) {}
  Evaluation operator()(const Board& board) const { return evaluator_.Evaluate(EncodeStudentInput(board)); }
 private:
  const SynchronousModelEvaluator& evaluator_;
};
}  // namespace hex_puct
