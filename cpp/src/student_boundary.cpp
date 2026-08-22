#include "hex_puct/student_boundary.hpp"
#include <queue>
#include <stdexcept>

namespace hex_puct {
int ActionFromRowColumn(int row, int column) { if(row<0||row>=kBoardSize||column<0||column>=kBoardSize) throw std::invalid_argument("coordinate outside board"); return row*kBoardSize+column; }
std::pair<int,int> RowColumnFromAction(int action) { if(action<0||action>=kBoardArea) throw std::invalid_argument("action outside board"); return {action/kBoardSize,action%kBoardSize}; }
std::string ActionToGtp(int action) { auto [row,column]=RowColumnFromAction(action); return std::string(1, static_cast<char>('a'+column))+std::to_string(row+1); }
StudentTensor EncodeStudentInput(const Board& board) {
  StudentTensor result{};
  for(int a=0;a<kBoardArea;++a) { result[a]=board.Cell(a)==1; result[kBoardArea+a]=board.Cell(a)==2; result[2*kBoardArea+a]=board.side_to_move()==Color::Black; result[3*kBoardArea+a]=board.last_move()&&*board.last_move()==a; }
  constexpr int dx[6]={-1,-1,0,0,1,1}; constexpr int dy[6]={0,1,-1,1,-1,0};
  for(Color color:{Color::Black,Color::White}) { int cell=color==Color::Black?1:2; std::array<bool,kBoardArea> start{},end{}; for(int edge=0;edge<2;++edge){std::array<bool,kBoardArea>& out=edge?end:start;std::queue<int> q;for(int a=0;a<kBoardArea;++a){auto [y,x]=RowColumnFromAction(a);bool touches=edge?(color==Color::Black?y==10:x==10):(color==Color::Black?y==0:x==0);if(board.Cell(a)==cell&&touches){out[a]=true;q.push(a);}}while(!q.empty()){int a=q.front();q.pop();auto[y,x]=RowColumnFromAction(a);for(int i=0;i<6;++i){int nx=x+dx[i],ny=y+dy[i];if(nx>=0&&nx<11&&ny>=0&&ny<11){int n=ActionFromRowColumn(ny,nx);if(!out[n]&&board.Cell(n)==cell){out[n]=true;q.push(n);}}}}} for(int a=0;a<kBoardArea;++a) if(board.Cell(a)==cell){result[4*kBoardArea+a]=start[a];result[5*kBoardArea+a]=end[a];} }
  return result;
}
}  // namespace hex_puct
