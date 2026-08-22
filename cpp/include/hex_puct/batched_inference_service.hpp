#pragma once
#include "hex_puct/student_boundary.hpp"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hex_puct {
struct BatchServiceConfig {
  size_t max_batch = 16;
  size_t max_queue = 256;
  std::chrono::microseconds max_wait{200};
};
struct BatchServiceStats {
  size_t batches = 0, requests = 0, queue_high_water = 0;
  std::vector<size_t> batch_sizes;
  std::vector<double> queue_wait_seconds, inference_seconds;
};
struct BatchServiceHealth {
  size_t queue_depth = 0, inflight_requests = 0, outstanding_requests = 0;
  size_t batches = 0, requests = 0;
  bool failed = false;
  std::string failure_reason;
  double seconds_since_last_dispatch = 0.0;
  double seconds_since_last_success = 0.0;
};

// One service thread owns model execution. Calls block for their own response;
// PUCT remains single-threaded within each independently-owned search tree.
class SharedInferenceService final : public SynchronousModelEvaluator {
 public:
  SharedInferenceService(const BatchedModelEvaluator& evaluator, BatchServiceConfig config);
  ~SharedInferenceService() override;
  SharedInferenceService(const SharedInferenceService&) = delete;
  Evaluation Evaluate(const StudentTensor& input) const override;
  void Shutdown();
  [[nodiscard]] BatchServiceStats Stats() const;
  [[nodiscard]] BatchServiceHealth Health() const;
  // Transitions the service permanently to failed and wakes every caller.
  // Public for bounded failure-injection tests; production code never retries.
  void Fail(std::exception_ptr failure);
 private:
  struct Request { StudentTensor input; std::promise<Evaluation> promise; std::chrono::steady_clock::time_point submitted; };
  void Run();
  [[nodiscard]] std::exception_ptr FailureLocked() const;
  const BatchedModelEvaluator& evaluator_;
  BatchServiceConfig config_;
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  mutable std::deque<std::shared_ptr<Request>> queue_;
  mutable std::vector<std::shared_ptr<Request>> inflight_;
  bool stopping_ = false;
  bool failed_ = false;
  std::exception_ptr failure_;
  std::string failure_reason_;
  std::chrono::steady_clock::time_point last_dispatch_ = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point last_success_ = std::chrono::steady_clock::now();
  mutable BatchServiceStats stats_;
  std::thread worker_;
};
}  // namespace hex_puct
