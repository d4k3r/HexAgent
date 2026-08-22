#pragma once
#include "hex_puct/student_boundary.hpp"
#include <memory>
namespace hex_puct {
class OnnxCpuEvaluator final : public SynchronousModelEvaluator {
 public:
  explicit OnnxCpuEvaluator(const std::string& model_path);
  ~OnnxCpuEvaluator() override;
  OnnxCpuEvaluator(OnnxCpuEvaluator&&) noexcept;
  Evaluation Evaluate(const StudentTensor& input) const override;
 private: struct Impl; std::unique_ptr<Impl> impl_;
};
class OnnxCudaEvaluator final : public SynchronousModelEvaluator {
 public:
  explicit OnnxCudaEvaluator(const std::string& model_path);
  ~OnnxCudaEvaluator() override;
  Evaluation Evaluate(const StudentTensor& input) const override;
 private: struct Impl; std::unique_ptr<Impl> impl_;
};
class OnnxCudaBatchEvaluator final : public BatchedModelEvaluator {
 public:
  explicit OnnxCudaBatchEvaluator(const std::string& model_path);
  ~OnnxCudaBatchEvaluator() override;
  std::vector<Evaluation> EvaluateBatch(const std::vector<StudentTensor>& input) const override;
 private: struct Impl; std::unique_ptr<Impl> impl_;
};
}
