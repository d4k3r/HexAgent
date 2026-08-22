// Long-lived Stage-6 service soak: separate trees, one shared CUDA owner.
#include "hex_puct/batched_inference_service.hpp"
#include "hex_puct/onnx_evaluator.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <mutex>
#include <thread>
using namespace hex_puct;
int main(int n,char**v) { if(n!=7)return 2; try {
 const int c=std::stoi(v[2]), budget=std::stoi(v[3]), maxb=std::stoi(v[4]), wait=std::stoi(v[5]), seconds=std::stoi(v[6]); if(c<1||budget<1||maxb<1||wait<0||seconds<1)return 2;
 Board root; OnnxCudaEvaluator direct(v[1]); auto ref=DeterministicPUCT(EncodedBoardEvaluator(direct),{budget}).Search(root);
 OnnxCudaBatchEvaluator batch(v[1]); SharedInferenceService service(batch,{size_t(maxb),256,std::chrono::microseconds(wait)}); std::atomic<size_t> searches{0};std::atomic<bool> same{true};std::exception_ptr error;std::mutex error_mu; auto end=std::chrono::steady_clock::now()+std::chrono::seconds(seconds); auto started=std::chrono::steady_clock::now();
 std::vector<std::thread> workers; for(int i=0;i<c;++i) workers.emplace_back([&]{try{while(std::chrono::steady_clock::now()<end){auto r=DeterministicPUCT(EncodedBoardEvaluator(service),{budget}).Search(root);if(r.selected_action!=ref.selected_action||r.raw_visits!=ref.raw_visits)same=false;++searches;}}catch(...){std::lock_guard l(error_mu);error=std::current_exception();}}); for(auto&t:workers)t.join(); service.Shutdown(); if(error)std::rethrow_exception(error);auto s=service.Stats();auto mean=[](const auto&x){return x.empty()?0.:std::accumulate(x.begin(),x.end(),0.)/x.size();};auto q=[](auto x,double f){if(x.empty())return 0.;std::sort(x.begin(),x.end());return double(x[std::min(x.size()-1,size_t(f*(x.size()-1)))]);};size_t full=std::count(s.batch_sizes.begin(),s.batch_sizes.end(),size_t(maxb));double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();const size_t expected=searches*size_t(budget+1);
 std::cout<<"{\"schema\":\"stage7-service-soak-v1\",\"passed\":"<<((same&&s.requests==expected)?"true":"false")<<",\"concurrency\":"<<c<<",\"budget\":"<<budget<<",\"max_batch\":"<<maxb<<",\"wait_us\":"<<wait<<",\"seconds\":"<<elapsed<<",\"searches\":"<<searches<<",\"total_simulations\":"<<searches*size_t(budget)<<",\"expected_requests\":"<<expected<<",\"requests\":"<<s.requests<<",\"batches\":"<<s.batches<<",\"mean_batch\":"<<mean(s.batch_sizes)<<",\"p50_batch\":"<<q(s.batch_sizes,.5)<<",\"p95_batch\":"<<q(s.batch_sizes,.95)<<",\"full_batch_fraction\":"<<(s.batch_sizes.empty()?0.:double(full)/s.batch_sizes.size())<<",\"queue_high_water\":"<<s.queue_high_water<<",\"mean_queue_wait_seconds\":"<<mean(s.queue_wait_seconds)<<",\"p95_queue_wait_seconds\":"<<q(s.queue_wait_seconds,.95)<<",\"mean_inference_seconds\":"<<mean(s.inference_seconds)<<",\"p95_inference_seconds\":"<<q(s.inference_seconds,.95)<<",\"simulations_per_second\":"<<(searches*double(budget)/elapsed)<<",\"neural_evaluations_per_second\":"<<(s.requests/elapsed)<<",\"completed_searches_per_second\":"<<(searches/elapsed)<<",\"exceptions\":0,\"same_as_direct\":"<<(same?"true":"false")<<",\"clean_shutdown\":true}\n";return same&&s.requests==expected?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
