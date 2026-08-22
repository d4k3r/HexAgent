#include "hex_puct/onnx_evaluator.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
using namespace hex_puct;
#ifdef HEX_PUCT_USE_CUDA
using QualifiedOnnxEvaluator = OnnxCudaEvaluator;
#else
using QualifiedOnnxEvaluator = OnnxCpuEvaluator;
#endif
struct F{const char*n;std::vector<int>b,w;Color s;int l;};template<class T>void A(const T&x){std::cout<<'[';for(size_t i=0;i<x.size();++i){if(i)std::cout<<',';std::cout<<x[i];}std::cout<<']';}
int main(int ac,char**av){if(ac!=4)return 2;int ix=std::stoi(av[2]),n=std::stoi(av[3]);std::vector<F>x={{"empty_black",{},{},Color::Black,-1},{"early_white",{0,12},{1,13},Color::White,13},{"mid_black",{0,12,24,36,48},{1,13,25,37},Color::Black,37},{"later_white",{0,11,22,33,44,55,66,77},{1,12,23,34,45,56,67},Color::White,67},{"colour_transpose_black",{0,12,25,37},{1,13,24,36},Color::Black,36},{"colour_transpose_white",{11,23,24,36},{0,12,35,47},Color::White,36}};if(ix<0||ix>=int(x.size()))return 3;auto t=std::chrono::steady_clock::now();QualifiedOnnxEvaluator m(av[1]);double init=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();auto&f=x[ix];auto b=Board::FromSetup(f.b,f.w,f.s,f.l<0?std::nullopt:std::optional<int>(f.l));t=std::chrono::steady_clock::now();auto r=DeterministicPUCT(EncodedBoardEvaluator(m),{n}).Search(b);double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::array<double,121>q{};for(int i=0;i<121;++i)q[i]=r.raw_visits[i]?r.raw_value_sums[i]/r.raw_visits[i]:0;std::array<int,121>legal{};auto mask=b.LegalMask();for(int i=0;i<121;++i)legal[i]=mask[i];std::cout<<std::setprecision(17)<<"{\"id\":\""<<f.n<<"\",\"budget\":"<<n<<",\"init_seconds\":"<<init<<",\"search_seconds\":"<<sec<<",\"legal\":";A(legal);std::cout<<",\"selected\":"<<(r.selected_action?std::to_string(*r.selected_action):"null")<<",\"visits\":"<<r.root_visits<<",\"n\":";A(r.raw_visits);std::cout<<",\"w\":";A(r.raw_value_sums);std::cout<<",\"q\":";A(q);std::cout<<",\"p\":";A(r.priors);std::cout<<",\"value\":";if(r.root_value)std::cout<<*r.root_value;else std::cout<<"null";std::cout<<"}\n";}
