#include "hex_puct/onnx_evaluator.hpp"
#include <cmath>
#include <iostream>
using namespace hex_puct;
int main(int n,char**v){if(n!=2)return 2;OnnxCudaEvaluator gpu(v[1]);OnnxCpuEvaluator cpu(v[1]);auto b=Board::FromSetup({0,12},{1,13},Color::Black,13);auto a=gpu.Evaluate(EncodeStudentInput(b));auto c=cpu.Evaluate(EncodeStudentInput(b));for(double x:a.policy_logits)if(!std::isfinite(x))return 3;if(!std::isfinite(a.value))return 4;double max=0;for(int i=0;i<121;++i)max=std::max(max,std::abs(a.policy_logits[i]-c.policy_logits[i]));std::cout<<"cuda_smoke max_policy_difference="<<max<<" value_difference="<<std::abs(a.value-c.value)<<"\n";}
