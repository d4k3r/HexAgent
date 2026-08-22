#include "hex_puct/puct.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace hex_puct;
struct Fixture { std::string name; std::vector<int> black, white; Color side; };
static std::vector<Fixture> Bank() { return {
  {"empty_black", {}, {}, Color::Black},
  {"early_white", {0, 12}, {1, 13}, Color::White},
  {"mid_black", {0,12,24,36,48,60,72,84,96,108}, {1,13,25,37,49,61,73,85,97}, Color::Black},
  {"late_white", {2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92}, {1,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105}, Color::White},
  {"forced_literal_win", {0,11,22,33,44,55,66,77,88,99}, {1,12,23,34,45,56,67,78,89,100}, Color::Black},
  {"transpose_black", {0,12,25,37}, {1,13,24,36}, Color::Black},
  {"colour_transpose_white", {11,13,24,36}, {0,12,35,47}, Color::White},
}; }
template <class T> static void Array(const T& xs) { std::cout << '['; for (size_t i=0;i<xs.size();++i) {if(i)std::cout<<',';std::cout<<xs[i];} std::cout<<']'; }
int main(int argc, char** argv) {
  std::vector<int> budgets{1,4,8,32,128}; bool bench=argc>1 && std::string(argv[1])=="--benchmark";
  std::cout << std::setprecision(17) << "{\"cases\":["; bool first=true;
  for(const auto& f:Bank()) for(int budget:budgets) { if(!first)std::cout<<',';first=false; Board b=Board::FromSetup(f.black,f.white,f.side); std::vector<int> trace; auto eval=[&trace](const Board& p){trace.push_back(p.Signature());return DeterministicFakeEvaluator{}(p);}; auto r=DeterministicPUCT(eval,{budget}); auto result=r.Search(b); std::array<double,kBoardArea> q{}; for(int i=0;i<kBoardArea;++i) q[i]=result.raw_visits[i]?result.raw_value_sums[i]/result.raw_visits[i]:0; std::cout<<"{\"name\":\""<<f.name<<"\",\"budget\":"<<budget<<",\"legal_mask\":"; auto mask=b.LegalMask(); std::array<int,kBoardArea> mi{};for(int i=0;i<kBoardArea;++i)mi[i]=mask[i];Array(mi);std::cout<<",\"trace\":";Array(trace);std::cout<<",\"terminal_winner\":"<<(b.LiteralWinner()?( *b.LiteralWinner()==Color::Black ? "\"black\"":"\"white\""):"null")<<",\"selected_action\":"<<(result.selected_action?std::to_string(*result.selected_action):"null")<<",\"root_visits\":"<<result.root_visits<<",\"raw_visits\":";Array(result.raw_visits);std::cout<<",\"priors\":";Array(result.priors);std::cout<<",\"raw_value_sums\":";Array(result.raw_value_sums);std::cout<<",\"q\":";Array(q);std::cout<<",\"root_value\":"; if(result.root_value) std::cout<<*result.root_value; else std::cout<<"null"; std::cout<<"}"; }
  std::cout<<"]";
  if(bench){Board b; auto start=std::chrono::steady_clock::now(); for(int i=0;i<20;++i) DeterministicPUCT(DeterministicFakeEvaluator{}, {128}).Search(b); auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::cerr<<"cpp_fake_simulations_per_second="<<(20.0*128/sec)<<"\n";} std::cout<<"}\n";
}
