#include "hex_puct/student_boundary.hpp"
#include <iostream>
#include <string>
#include <vector>
using namespace hex_puct;
struct Fixture { std::string name; std::vector<int> black, white; Color side; int last; };
static std::vector<Fixture> Bank() { return {
 {"empty_black",{},{},Color::Black,-1}, {"empty_white",{},{},Color::White,-1},
 {"first_move_white",{60},{},Color::White,60},
 {"edge_components_black",{0,11,22,33},{1,12,23},Color::Black,23},
 {"midgame_white",{0,12,24,36,48},{1,13,25,37},Color::White,37},
 {"swap_ownership_black",{0},{1},Color::Black,1},
 {"colour_transpose_black",{0,12,25,37},{1,13,24,36},Color::Black,36},
 {"colour_transpose_white",{11,23,24,36},{0,12,35,47},Color::White,36},
 {"literal_terminal_black",{0,11,22,33,44,55,66,77,88,99,110},{1,12,23,34,45,56,67,78,89,100},Color::White,110},
 {"late_dense",{0,2,4,6,8,10,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109},{1,3,5,7,9,11,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108},Color::Black,109}
}; }
template<class T> static void Array(const T& xs){std::cout<<'[';for(size_t i=0;i<xs.size();++i){if(i)std::cout<<',';std::cout<<xs[i];}std::cout<<']';}
int main(){std::cout<<"{\"cases\":[";bool first=true;for(const auto& f:Bank()){if(!first)std::cout<<',';first=false;Board b=Board::FromSetup(f.black,f.white,f.side,f.last<0?std::nullopt:std::optional<int>(f.last));auto tensor=EncodeStudentInput(b);auto mask=b.LegalMask();std::array<int,kBoardArea> m{};for(int i=0;i<kBoardArea;++i)m[i]=mask[i];std::cout<<"{\"name\":\""<<f.name<<"\",\"tensor\":";Array(tensor);std::cout<<",\"legal_mask\":";Array(m);std::cout<<",\"terminal_winner\":"<<(b.LiteralWinner()?(*b.LiteralWinner()==Color::Black?"\"black\"":"\"white\""):"null")<<"}";}std::cout<<"],\"actions\":[";for(int a=0;a<kBoardArea;++a){if(a)std::cout<<',';auto [row,col]=RowColumnFromAction(a);std::cout<<"["<<a<<","<<row<<","<<col<<",\""<<ActionToGtp(a)<<"\"]";}std::cout<<"]}\n";}
