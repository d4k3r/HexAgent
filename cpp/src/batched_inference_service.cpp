#include "hex_puct/batched_inference_service.hpp"
#include <stdexcept>
namespace hex_puct {
SharedInferenceService::SharedInferenceService(const BatchedModelEvaluator& e, BatchServiceConfig c)
    : evaluator_(e), config_(c), worker_(&SharedInferenceService::Run, this) {
  if (!config_.max_batch || !config_.max_queue) throw std::invalid_argument("batch and queue bounds must be nonzero");
}
SharedInferenceService::~SharedInferenceService() { Shutdown(); }
void SharedInferenceService::Shutdown() {
  { std::lock_guard lock(mutex_); if (!stopping_) stopping_ = true; }
  cv_.notify_all(); if (worker_.joinable()) worker_.join();
}
BatchServiceStats SharedInferenceService::Stats() const { std::lock_guard lock(mutex_); return stats_; }
std::exception_ptr SharedInferenceService::FailureLocked() const {
  return failure_ ? failure_ : std::make_exception_ptr(std::runtime_error("inference service failed"));
}
BatchServiceHealth SharedInferenceService::Health() const {
  std::lock_guard lock(mutex_);
  const auto now = std::chrono::steady_clock::now();
  return {queue_.size(), inflight_.size(), queue_.size() + inflight_.size(), stats_.batches, stats_.requests,
          failed_, failure_reason_,
          std::chrono::duration<double>(now - last_dispatch_).count(),
          std::chrono::duration<double>(now - last_success_).count()};
}
void SharedInferenceService::Fail(std::exception_ptr error) {
  std::vector<std::shared_ptr<Request>> failed_requests;
  std::exception_ptr stored;
  {
    std::lock_guard lock(mutex_);
    if (!failed_) {
      failed_ = true; stopping_ = true; failure_ = error;
      try { std::rethrow_exception(error); }
      catch (const std::exception& e) { failure_reason_ = e.what(); }
      catch (...) { failure_reason_ = "unknown inference worker failure"; }
    }
    stored = FailureLocked();
    failed_requests.insert(failed_requests.end(), queue_.begin(), queue_.end()); queue_.clear();
    failed_requests.insert(failed_requests.end(), inflight_.begin(), inflight_.end()); inflight_.clear();
  }
  cv_.notify_all();
  for (const auto& request : failed_requests) {
    try { request->promise.set_exception(stored); } catch (const std::future_error&) {}
  }
}
Evaluation SharedInferenceService::Evaluate(const StudentTensor& input) const {
  auto r = std::make_shared<Request>(); r->input = input; r->submitted=std::chrono::steady_clock::now(); auto future = r->promise.get_future();
  { std::unique_lock lock(mutex_);
    cv_.wait(lock, [&]{ return stopping_ || failed_ || queue_.size() < config_.max_queue; });
    if (failed_) std::rethrow_exception(FailureLocked());
    if (stopping_) throw std::runtime_error("inference service is stopped");
    queue_.push_back(r); stats_.queue_high_water=std::max(stats_.queue_high_water,queue_.size());
  }
  cv_.notify_one(); return future.get();
}
void SharedInferenceService::Run() {
  try { for (;;) {
    std::vector<std::shared_ptr<Request>> requests;
    { std::unique_lock lock(mutex_);
      cv_.wait(lock, [&]{ return stopping_ || !queue_.empty(); });
      if ((stopping_ || failed_) && queue_.empty()) return;
      const auto deadline = std::chrono::steady_clock::now() + config_.max_wait;
      while (!stopping_ && queue_.size() < config_.max_batch && cv_.wait_until(lock, deadline) != std::cv_status::timeout) {}
      const size_t count = std::min(config_.max_batch, queue_.size());
      for (size_t i=0;i<count;++i) { requests.push_back(queue_.front()); queue_.pop_front(); }
      inflight_ = requests;
      ++stats_.batches; stats_.requests += count; stats_.batch_sizes.push_back(count);
      const auto dispatch=std::chrono::steady_clock::now();
      last_dispatch_ = dispatch;
      for(const auto& r:requests) stats_.queue_wait_seconds.push_back(std::chrono::duration<double>(dispatch-r->submitted).count());
    }
    cv_.notify_all();
    std::vector<StudentTensor> input; input.reserve(requests.size());
    for (const auto& r: requests) input.push_back(r->input);
    const auto infer_started=std::chrono::steady_clock::now();
    auto output = evaluator_.EvaluateBatch(input);
    const double infer_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-infer_started).count();
    if (output.size()!=requests.size()) throw std::runtime_error("batch evaluator result count mismatch");
    for (size_t i=0;i<requests.size();++i) requests[i]->promise.set_value(output[i]);
    { std::lock_guard lock(mutex_);
      stats_.inference_seconds.push_back(infer_seconds);
      inflight_.clear(); last_success_=std::chrono::steady_clock::now();
    }
    cv_.notify_all();
  }} catch (...) { Fail(std::current_exception()); }
}
}  // namespace hex_puct
