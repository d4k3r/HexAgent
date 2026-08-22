#include "hex_puct/batched_inference_service.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace hex_puct;
namespace {
StudentTensor Input() { StudentTensor x{}; x[0] = 1.0f; return x; }
Evaluation Output() { Evaluation y{}; y.value = 0.25; return y; }
struct Fake final : BatchedModelEvaluator {
  explicit Fake(int fail_after = -1, int delay_ms = 0) : fail_after_(fail_after), delay_ms_(delay_ms) {}
  std::vector<Evaluation> EvaluateBatch(const std::vector<StudentTensor>& input) const override {
    const int n = calls_.fetch_add(1);
    if (delay_ms_) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
    if (fail_after_ >= 0 && n >= fail_after_) throw std::runtime_error("injected worker failure");
    return std::vector<Evaluation>(input.size(), Output());
  }
  mutable std::atomic<int> calls_{0}; int fail_after_, delay_ms_;
};
template <class F> void Prompt(F&& f) { assert(f.wait_for(std::chrono::seconds(2)) == std::future_status::ready); }
}
int main() {
  { Fake fake; SharedInferenceService service(fake, {4, 32, std::chrono::microseconds(0)});
    std::vector<std::future<Evaluation>> futures;
    for (int i=0;i<12;++i) futures.push_back(std::async(std::launch::async,[&]{return service.Evaluate(Input());}));
    for(auto& f:futures) { Prompt(f); assert(f.get().value == 0.25); }
    service.Shutdown(); assert(!service.Health().failed);
  }
  { Fake fake(0); SharedInferenceService service(fake, {4, 32, std::chrono::microseconds(0)});
    std::vector<std::future<bool>> futures;
    for (int i=0;i<12;++i) futures.push_back(std::async(std::launch::async,[&]{try { service.Evaluate(Input()); return false; } catch (...) { return true; }}));
    for(auto& f:futures) { Prompt(f); assert(f.get()); }
    assert(service.Health().failed); service.Shutdown();
  }
  { Fake fake(-1, 100); SharedInferenceService service(fake, {4, 32, std::chrono::microseconds(0)});
    std::vector<std::future<bool>> futures;
    for (int i=0;i<8;++i) futures.push_back(std::async(std::launch::async,[&]{try { service.Evaluate(Input()); return false; } catch (...) { return true; }}));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    service.Fail(std::make_exception_ptr(std::runtime_error("injected forced stop")));
    for(auto& f:futures) { Prompt(f); assert(f.get()); }
    assert(service.Health().failed); service.Shutdown();
  }
  std::cout << "batched inference liveness tests passed\n";
}
