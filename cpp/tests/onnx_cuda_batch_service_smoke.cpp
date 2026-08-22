#include "hex_puct/batched_inference_service.hpp"
#include "hex_puct/onnx_evaluator.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <numeric>
using namespace hex_puct;
int main(int n,char**v){
 if(n!=7)return 2; const int concurrency=std::stoi(v[2]), sims=std::stoi(v[3]), max_batch=std::stoi(v[4]), wait_us=std::stoi(v[5]), ix=std::stoi(v[6]);
 OnnxCudaEvaluator direct(v[1]); OnnxCudaBatchEvaluator batch(v[1]);
 std::vector<Board> roots; roots.emplace_back(Color::Black); roots.push_back(Board::FromSetup({0,12},{1,13},Color::White,13)); roots.push_back(Board::FromSetup({0,12,24,36,48},{1,13,25,37},Color::Black,37)); roots.push_back(Board::FromSetup({0,11,22,33,44,55,66,77},{1,12,23,34,45,56,67},Color::White,67)); roots.push_back(Board::FromSetup({0,12,25,37},{1,13,24,36},Color::Black,36)); roots.push_back(Board::FromSetup({11,23,24,36},{0,12,35,47},Color::White,36)); if(ix<0||ix>=int(roots.size()))return 3; Board root=roots[ix];
 auto reference=DeterministicPUCT(EncodedBoardEvaluator(direct),{sims}).Search(root);
 SharedInferenceService service(batch,{static_cast<size_t>(max_batch),256,std::chrono::microseconds(wait_us)});
 std::vector<SearchResult> results(concurrency); std::vector<std::thread> workers;
 auto started=std::chrono::steady_clock::now();
 for(int i=0;i<concurrency;++i)workers.emplace_back([&,i]{Board b=root;results[i]=DeterministicPUCT(EncodedBoardEvaluator(service),{sims}).Search(b);});
 for(auto& t:workers)t.join(); double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count(); service.Shutdown(); auto stats=service.Stats();
 bool same=true;for(auto&r:results)same&=r.selected_action==reference.selected_action&&r.raw_visits==reference.raw_visits;
 auto avg=[](const auto& a){return a.empty()?0.0:std::accumulate(a.begin(),a.end(),0.0)/a.size();}; auto percentile=[](auto a,double q){if(a.empty())return 0.0;std::sort(a.begin(),a.end());return a[std::min(a.size()-1,static_cast<size_t>(q*(a.size()-1)))];};
 std::cout<<"{\"position_index\":"<<ix<<",\"concurrency\":"<<concurrency<<",\"budget\":"<<sims<<",\"max_batch\":"<<max_batch<<",\"wait_us\":"<<wait_us<<",\"same_as_direct\":"<<(same?"true":"false")<<",\"selected\":"<<(reference.selected_action?std::to_string(*reference.selected_action):"null")<<",\"seconds\":"<<seconds<<",\"simulations_per_second\":"<<(concurrency*sims/seconds)<<",\"batches\":"<<stats.batches<<",\"requests\":"<<stats.requests<<",\"queue_high_water\":"<<stats.queue_high_water<<",\"mean_queue_wait_seconds\":"<<avg(stats.queue_wait_seconds)<<",\"p95_queue_wait_seconds\":"<<percentile(stats.queue_wait_seconds,.95)<<",\"mean_inference_seconds\":"<<avg(stats.inference_seconds)<<",\"p95_inference_seconds\":"<<percentile(stats.inference_seconds,.95)<<",\"batch_sizes\":[";for(size_t i=0;i<stats.batch_sizes.size();++i){if(i)std::cout<<',';std::cout<<stats.batch_sizes[i];}std::cout<<"]}\n";
 return same?0:1;
}
