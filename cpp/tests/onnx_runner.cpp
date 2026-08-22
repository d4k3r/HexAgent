#include "hex_puct/onnx_evaluator.hpp"
#include <iomanip>
#include <iostream>
using namespace hex_puct;
#ifdef HEX_PUCT_USE_CUDA
using QualifiedOnnxEvaluator = OnnxCudaEvaluator;
#else
using QualifiedOnnxEvaluator = OnnxCpuEvaluator;
#endif
struct F{const char*n;std::vector<int>b,w;Color s;int l;};
int main(int ac,char**av){if(ac!=2)return 2;std::vector<F>x={{"empty_black",{},{},Color::Black,-1},{"empty_white",{},{},Color::White,-1},{"first_move_white",{60},{},Color::White,60},{"edge_components_black",{0,11,22,33},{1,12,23},Color::Black,23},{"midgame_white",{0,12,24,36,48},{1,13,25,37},Color::White,37},{"swap_ownership_black",{0},{1},Color::Black,1},{"colour_transpose_black",{0,12,25,37},{1,13,24,36},Color::Black,36},{"colour_transpose_white",{11,23,24,36},{0,12,35,47},Color::White,36},{"literal_terminal_black",{0,11,22,33,44,55,66,77,88,99,110},{1,12,23,34,45,56,67,78,89,100},Color::White,110},{"late_dense",{0,2,4,6,8,10,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109},{1,3,5,7,9,11,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108},Color::Black,109}};QualifiedOnnxEvaluator e(av[1]);std::cout<<std::setprecision(9)<<"[";for(size_t j=0;j<x.size();++j){if(j)std::cout<<',';auto&f=x[j];auto y=e.Evaluate(EncodeStudentInput(Board::FromSetup(f.b,f.w,f.s,f.l<0?std::nullopt:std::optional<int>(f.l))));std::cout<<"{\"id\":\""<<f.n<<"\",\"policy\":[";for(int i=0;i<121;++i){if(i)std::cout<<',';std::cout<<y.policy_logits[i];}std::cout<<"],\"value\":"<<y.value<<"}";}std::cout<<"]\n";}
