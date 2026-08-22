#include "hex_puct/onnx_evaluator.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
using namespace hex_puct;
int main(int n,char**v) { if(n!=4)return 2; try { const int batch=std::stoi(v[2]), seconds=std::stoi(v[3]); if(batch<1||batch>512||seconds<1)return 2;
  OnnxCudaBatchEvaluator eval(v[1]); std::vector<StudentTensor> input(batch); for(int b=0;b<batch;++b) { int black=b%121, white=(b*17+1)%121; if(white==black)white=(white+1)%121; input[b]=EncodeStudentInput(Board::FromSetup({black},{white},Color::Black,white)); }
  size_t calls=0, positions=0; double total=0,max=0;std::vector<double> lat; auto until=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
  while(std::chrono::steady_clock::now()<until) { auto t=std::chrono::steady_clock::now(); auto out=eval.EvaluateBatch(input); double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count(); total+=dt;max=std::max(max,dt);lat.push_back(dt); if(out.size()!=input.size())throw std::runtime_error("batch size mismatch"); for(auto&e:out){if(!std::isfinite(e.value))throw std::runtime_error("nonfinite value");for(double x:e.policy_logits)if(!std::isfinite(x))throw std::runtime_error("nonfinite logit");} ++calls;positions+=out.size(); } std::sort(lat.begin(),lat.end());auto q=[&](double x){return lat[std::min(lat.size()-1,size_t(x*(lat.size()-1)))];};
  std::cout<<"{\"schema\":\"stage7-pure-inference-benchmark-v1\",\"passed\":true,\"batch\":"<<batch<<",\"calls\":"<<calls<<",\"positions\":"<<positions<<",\"exceptions\":0,\"mean_batch_latency_seconds\":"<<total/calls<<",\"p50_batch_latency_seconds\":"<<q(.5)<<",\"p95_batch_latency_seconds\":"<<q(.95)<<",\"max_batch_latency_seconds\":"<<max<<",\"positions_per_second\":"<<positions/total<<"}\n"; return 0;
 } catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;} }
