#include "hex_puct/onnx_evaluator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace hex_puct;
int main(int ac,char**av){if(ac!=2)return 2;OnnxCpuEvaluator m(av[1]);EncodedBoardEvaluator e(m);for(auto b:{Board{},Board::FromSetup({0,12},{1,13},Color::Black,13)})for(int n:{1,4,8,32}){auto a=DeterministicPUCT(e,{n}).Search(b);auto z=DeterministicPUCT(e,{n}).Search(b);assert(a.raw_visits==z.raw_visits&&a.selected_action==z.selected_action&&a.root_visits==n);assert(a.selected_action&&b.LegalMask()[*a.selected_action]);for(double p:a.priors)assert(std::isfinite(p));}Board terminal=Board::FromSetup({0,11,22,33,44,55,66,77,88,99,110},{},Color::White,110);assert(!DeterministicPUCT(e,{4}).Search(terminal).selected_action);std::cout<<"onnx neural PUCT smoke passed\n";}
