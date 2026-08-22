// Stage-8C paired candidate/champion evaluator. Literal terminal semantics are
// authoritative; elementary certificates are an optional game-control layer.
#include "hex_puct/batched_inference_service.hpp"
#include "hex_puct/connection_certificate.hpp"
#include "hex_puct/connection_realizer.hpp"
#include "hex_puct/onnx_evaluator.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

using namespace hex_puct;
namespace fs = std::filesystem;

enum class BridgeControllerMode { Off, Shadow, Active };
struct O { std::string candidate,champion,openings,out; int budget=2048,candidate_budget=0,champion_budget=0,concurrency=1,max_batch=96,wait=200; double cp=1.5,candidate_cp=-1,champion_cp=-1,fpu_reduction=0.0,candidate_fpu_reduction=-1,champion_fpu_reduction=-1; FpuMode fpu_mode=FpuMode::Zero,candidate_fpu_mode=FpuMode::Zero,champion_fpu_mode=FpuMode::Zero; bool candidate_fpu_set=false,champion_fpu_set=false; BridgeControllerMode bridge_mode=BridgeControllerMode::Off; };
struct Opening { std::string id; std::vector<int> moves; bool swap=false; };

static char C(Color c){return c==Color::Black?'B':'W';}
static const char* Name(BridgeControllerMode m) { return m==BridgeControllerMode::Off?"off":m==BridgeControllerMode::Shadow?"shadow":"active"; }
static const char* Name(RealizerMoveKind k) { return k==RealizerMoveKind::PairedResponse?"paired_response":k==RealizerMoveKind::ProactiveResolution?"proactive_resolution":k==RealizerMoveKind::Complete?"complete":"invalid"; }
static bool Bridge(const ConnectionSegment& s) { return s.kind==ConnectionSegmentKind::Bridge||s.kind==ConnectionSegmentKind::EdgeBridge; }
static O Parse(int n,char**v){O o;for(int i=1;i<n;i+=2){if(i+1>=n)throw std::runtime_error("missing value");std::string k=v[i],x=v[i+1];if(k=="--candidate")o.candidate=x;else if(k=="--champion")o.champion=x;else if(k=="--openings")o.openings=x;else if(k=="--output")o.out=x;else if(k=="--budget")o.budget=std::stoi(x);else if(k=="--candidate-budget")o.candidate_budget=std::stoi(x);else if(k=="--champion-budget")o.champion_budget=std::stoi(x);else if(k=="--concurrency")o.concurrency=std::stoi(x);else if(k=="--max-batch")o.max_batch=std::stoi(x);else if(k=="--wait-us")o.wait=std::stoi(x);else if(k=="--c-puct")o.cp=std::stod(x);else if(k=="--candidate-c-puct")o.candidate_cp=std::stod(x);else if(k=="--champion-c-puct")o.champion_cp=std::stod(x);else if(k=="--fpu-mode"){if(x=="zero")o.fpu_mode=FpuMode::Zero;else if(x=="parent_value_reduced")o.fpu_mode=FpuMode::ParentValueReduced;else throw std::runtime_error("invalid fpu mode");}else if(k=="--candidate-fpu-mode"){if(x=="zero")o.candidate_fpu_mode=FpuMode::Zero;else if(x=="parent_value_reduced")o.candidate_fpu_mode=FpuMode::ParentValueReduced;else throw std::runtime_error("invalid candidate fpu mode");o.candidate_fpu_set=true;}else if(k=="--champion-fpu-mode"){if(x=="zero")o.champion_fpu_mode=FpuMode::Zero;else if(x=="parent_value_reduced")o.champion_fpu_mode=FpuMode::ParentValueReduced;else throw std::runtime_error("invalid champion fpu mode");o.champion_fpu_set=true;}else if(k=="--fpu-reduction")o.fpu_reduction=std::stod(x);else if(k=="--candidate-fpu-reduction")o.candidate_fpu_reduction=std::stod(x);else if(k=="--champion-fpu-reduction")o.champion_fpu_reduction=std::stod(x);else if(k=="--bridge-controller"){if(x=="off")o.bridge_mode=BridgeControllerMode::Off;else if(x=="shadow")o.bridge_mode=BridgeControllerMode::Shadow;else if(x=="active")o.bridge_mode=BridgeControllerMode::Active;else throw std::runtime_error("invalid bridge controller mode");}else throw std::runtime_error("unknown option "+k);}if(o.candidate.empty()||o.champion.empty()||o.openings.empty()||o.out.empty()||o.budget<1||o.concurrency<1)throw std::runtime_error("invalid options");if(!o.candidate_budget)o.candidate_budget=o.budget;if(!o.champion_budget)o.champion_budget=o.budget;if(o.candidate_cp<0)o.candidate_cp=o.cp;if(o.champion_cp<0)o.champion_cp=o.cp;if(o.candidate_fpu_reduction<0)o.candidate_fpu_reduction=o.fpu_reduction;if(o.champion_fpu_reduction<0)o.champion_fpu_reduction=o.fpu_reduction;if(!o.candidate_fpu_set)o.candidate_fpu_mode=o.fpu_mode;if(!o.champion_fpu_set)o.champion_fpu_mode=o.fpu_mode;if(o.candidate_budget<1||o.champion_budget<1||o.candidate_cp<0||o.champion_cp<0||o.fpu_reduction<0||o.candidate_fpu_reduction<0||o.champion_fpu_reduction<0||(o.fpu_mode==FpuMode::Zero&&o.fpu_reduction!=0)||(o.candidate_fpu_mode==FpuMode::Zero&&o.candidate_fpu_reduction!=0)||(o.champion_fpu_mode==FpuMode::Zero&&o.champion_fpu_reduction!=0))throw std::runtime_error("invalid options");return o;}
static std::vector<Opening> Read(const std::string&p){std::ifstream in(p);std::vector<Opening>r;std::string s;while(std::getline(in,s)){std::stringstream q(s);Opening x;std::string actions,sw;if(!std::getline(q,x.id,'|')||!std::getline(q,actions,'|')||!std::getline(q,sw))throw std::runtime_error("malformed opening line");std::stringstream a(actions);std::string z;while(std::getline(a,z,','))if(!z.empty())x.moves.push_back(std::stoi(z));x.swap=sw=="SWAP";if(x.moves.empty())throw std::runtime_error("opening requires first black move");r.push_back(x);}return r;}
static int Pick(const SearchResult&r){int a=0;for(int i=1;i<kBoardArea;i++)if(r.raw_visits[i]>r.raw_visits[a])a=i;return a;}

struct ControllerEvent { int ply; RealizerMoveKind kind; int prescribed, actual; bool agrees; };
struct ControllerTracker {
  Color owner;
  std::optional<ElementaryCertificateRealizer> realizer;
  bool disabled=false;
  int first_certificate_ply=-1, bridge_count=0, active_turns=0, required_responses=0, ignored_required_responses=0;
  int controller_moves=0, paired_responses=0, proactive_resolutions=0, fail_closed=0;
  std::vector<std::string> failures;
  std::vector<ControllerEvent> events;

  void Observe(const Board& board, int ply, uint64_t& detector_calls, double& detector_seconds) {
    if (realizer) {
      const auto progress=realizer->Inspect(board);
      if (!progress.valid) { ++fail_closed; failures.push_back(progress.failure_reason); realizer.reset(); disabled=true; }
      return;
    }
    if (disabled) return;
    const auto started=std::chrono::steady_clock::now();
    const auto certificate=FindElementaryBridgeCertificate(board,owner);
    detector_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count(); ++detector_calls;
    if (!certificate) return;
    try {
      int bridges=0;for(const auto& s:certificate->route)if(Bridge(s))++bridges;
      realizer.emplace(board,*certificate); first_certificate_ply=ply; bridge_count=bridges;
    } catch(const std::exception& e) { ++fail_closed; failures.push_back(e.what()); disabled=true; }
  }
  std::optional<RealizerDecision> Decision(const Board& board) {
    if (!realizer) return std::nullopt;
    ++active_turns; const auto d=realizer->ChooseMove(board,board.last_move());
    if (d.kind==RealizerMoveKind::Invalid || d.kind==RealizerMoveKind::Complete) {
      ++fail_closed; failures.push_back(d.reason); realizer.reset(); disabled=true; return std::nullopt;
    }
    if (d.kind==RealizerMoveKind::PairedResponse) ++required_responses;
    return d;
  }
  void Event(int ply,const RealizerDecision& d,int actual,bool controlled) {
    const bool agrees=d.action&&*d.action==actual; events.push_back({ply,d.kind,d.action.value_or(-1),actual,agrees});
    if (d.kind==RealizerMoveKind::PairedResponse && !agrees) ++ignored_required_responses;
    if (!controlled) return;
    ++controller_moves;
    if(d.kind==RealizerMoveKind::PairedResponse)++paired_responses;
    if(d.kind==RealizerMoveKind::ProactiveResolution)++proactive_resolutions;
  }
  void Json(std::ostream& z,int final_ply) const {
    z<<"{\"owner\":\""<<C(owner)<<"\",\"first_certificate_ply\":";
    if(first_certificate_ply<0)z<<"null";else z<<first_certificate_ply;
    z<<",\"bridge_count\":"<<bridge_count<<",\"turns_certificate_active\":"<<active_turns
     <<",\"required_responses\":"<<required_responses<<",\"ignored_required_responses\":"<<ignored_required_responses
     <<",\"controller_moves\":"<<controller_moves<<",\"successful_paired_responses\":"<<paired_responses
     <<",\"proactive_resolutions\":"<<proactive_resolutions<<",\"fail_closed_events\":"<<fail_closed
     <<",\"certificate_to_literal_tail\":";
    if(first_certificate_ply<0)z<<"null";else z<<final_ply-first_certificate_ply;
    z<<",\"failures\":[";for(size_t i=0;i<failures.size();++i){if(i)z<<',';z<<"\""<<failures[i]<<"\"";}z<<"],\"events\":[";
    for(size_t i=0;i<events.size();++i){if(i)z<<',';const auto&e=events[i];z<<"{\"ply\":"<<e.ply<<",\"kind\":\""<<Name(e.kind)<<"\",\"prescribed_move\":"<<e.prescribed<<",\"actual_move\":"<<e.actual<<",\"agrees\":"<<(e.agrees?"true":"false")<<'}';}z<<"]}";
  }
};
struct ControllerGameTelemetry { uint64_t detector_calls=0; double detector_seconds=0; ControllerTracker black{Color::Black},white{Color::White}; };

static std::string Game(const Opening&o,bool candidate_black,const O&x,const Evaluator&cand,const Evaluator&champ,std::atomic<uint64_t>& searches,std::atomic<uint64_t>& simulations){
  Board b;for(int a:o.moves)b.Play(a);Color cc=candidate_black?Color::Black:Color::White;if(o.swap)cc=Opponent(cc);std::vector<int> played=o.moves;int searched=0,controlled=0;ControllerGameTelemetry telemetry;
  const auto observe=[&]{if(x.bridge_mode==BridgeControllerMode::Off)return;telemetry.black.Observe(b,int(played.size()),telemetry.detector_calls,telemetry.detector_seconds);telemetry.white.Observe(b,int(played.size()),telemetry.detector_calls,telemetry.detector_seconds);};observe();
  while(!b.LiteralWinner()){
    const Color turn=b.side_to_move();ControllerTracker& tracker=turn==Color::Black?telemetry.black:telemetry.white;std::optional<RealizerDecision> decision;
    if(x.bridge_mode!=BridgeControllerMode::Off)decision=tracker.Decision(b);
    int action=-1;bool controller_selected=false;
    if(x.bridge_mode==BridgeControllerMode::Active&&decision&&decision->action){action=*decision->action;controller_selected=true;++controlled;tracker.Event(int(played.size())+1,*decision,action,true);}
    else {const bool candidate_turn=turn==cc;const Evaluator&e=candidate_turn?cand:champ;const int budget=candidate_turn?x.candidate_budget:x.champion_budget;const double cp=candidate_turn?x.candidate_cp:x.champion_cp;const FpuMode mode=candidate_turn?x.candidate_fpu_mode:x.champion_fpu_mode;const double reduction=candidate_turn?x.candidate_fpu_reduction:x.champion_fpu_reduction;auto r=DeterministicPUCT(e,SearchConfig{budget,cp,mode,reduction}).Search(b);if(!r.selected_action||r.root_visits!=budget)throw std::runtime_error("incomplete search");action=Pick(r);++searched;searches.fetch_add(1);simulations.fetch_add(static_cast<uint64_t>(budget));if(decision&&decision->action)tracker.Event(int(played.size())+1,*decision,action,false);}
    if(!b.LegalMask()[action])throw std::runtime_error("illegal selected action");b.Play(action);played.push_back(action);observe();
  }
  const bool win=*b.LiteralWinner()==cc;std::ostringstream z;z<<"{\"candidate_physical_colour\":\""<<C(cc)<<"\",\"winner\":\""<<C(*b.LiteralWinner())<<"\",\"candidate_score\":"<<(win?1:0)<<",\"searched_moves\":"<<searched<<",\"controller_selected_moves\":"<<controlled<<",\"moves\":[";for(size_t i=0;i<played.size();++i){if(i)z<<',';z<<played[i];}z<<"],\"bridge_controller\":{\"mode\":\""<<Name(x.bridge_mode)<<"\",\"detector_calls\":"<<telemetry.detector_calls<<",\"detector_seconds\":"<<telemetry.detector_seconds<<",\"black\":";telemetry.black.Json(z,int(played.size()));z<<",\"white\":";telemetry.white.Json(z,int(played.size()));z<<"}}";return z.str();}
struct Activity { std::atomic<size_t> completed_pairs=0,active_pairs=0,active_games=0; };
static std::string Escape(std::string s) { for (size_t p=0;(p=s.find('"',p))!=std::string::npos;p+=2)s.replace(p,1,"\\\""); return s; }
static void AtomicText(const fs::path& path,const std::string& text) { const auto partial=path.string()+".partial"; { std::ofstream out(partial); out<<text; } fs::rename(partial,path); }
#ifndef HEX_CANDIDATE_MATCH_FAKE
static void WriteStatus(const fs::path& path,const char* state,const Activity& a,const SharedInferenceService& candidate,const SharedInferenceService& champion,const std::string& reason="") {
  const auto c=candidate.Health(), h=champion.Health(); std::ostringstream z;
  z<<"{\"schema\":\"stage8c-runner-status-v1\",\"status\":\""<<state<<"\",\"completed_pairs\":"<<a.completed_pairs.load()<<",\"active_pairs\":"<<a.active_pairs.load()<<",\"active_games\":"<<a.active_games.load()
   <<",\"candidate\":{\"queue_depth\":"<<c.queue_depth<<",\"inflight_requests\":"<<c.inflight_requests<<",\"outstanding_requests\":"<<c.outstanding_requests<<",\"batches\":"<<c.batches<<",\"requests\":"<<c.requests<<",\"failed\":"<<(c.failed?"true":"false")<<",\"failure_reason\":\""<<Escape(c.failure_reason)<<"\",\"seconds_since_last_success\":"<<c.seconds_since_last_success<<"}"
   <<",\"champion\":{\"queue_depth\":"<<h.queue_depth<<",\"inflight_requests\":"<<h.inflight_requests<<",\"outstanding_requests\":"<<h.outstanding_requests<<",\"batches\":"<<h.batches<<",\"requests\":"<<h.requests<<",\"failed\":"<<(h.failed?"true":"false")<<",\"failure_reason\":\""<<Escape(h.failure_reason)<<"\",\"seconds_since_last_success\":"<<h.seconds_since_last_success<<"}"
   <<",\"reason\":\""<<Escape(reason)<<"\"}\n"; AtomicText(path,z.str());
}
#endif
static void Run(const Opening&o,const O&x,const Evaluator&cand,const Evaluator&champ,std::atomic<uint64_t>& searches,std::atomic<uint64_t>& simulations,Activity& activity){fs::path f=fs::path(x.out)/("pair-"+o.id+".json");if(fs::exists(f))return;activity.active_pairs.fetch_add(1);struct PairGuard { Activity&a; ~PairGuard(){a.active_pairs.fetch_sub(1);} } guard{activity};auto game=[&](bool candidate_black){activity.active_games.fetch_add(1);try{auto result=Game(o,candidate_black,x,cand,champ,searches,simulations);activity.active_games.fetch_sub(1);return result;}catch(...){activity.active_games.fetch_sub(1);throw;}};auto a=std::async(std::launch::async,[&]{return game(true);});auto b=std::async(std::launch::async,[&]{return game(false);});auto ga=a.get(),gb=b.get();std::ofstream q(f.string()+".partial");q<<"{\"schema\":\"stage8c-pair-v1\",\"status\":\"complete\",\"pair_id\":\""<<o.id<<"\",\"opening_moves\":[";for(size_t i=0;i<o.moves.size();++i){if(i)q<<',';q<<o.moves[i];}q<<"],\"swap_decision\":\""<<(o.swap?"SWAP":"KEEP")<<"\",\"game_a\":"<<ga<<",\"game_b\":"<<gb<<"}\n";q.close();fs::rename(f.string()+".partial",f);activity.completed_pairs.fetch_add(1);}
static void Stats(std::ostream&o,const BatchServiceStats&s){size_t peak=0;for(auto n:s.batch_sizes)peak=std::max(peak,n);o<<"{\"requests\":"<<s.requests<<",\"batches\":"<<s.batches<<",\"mean_batch\":"<<(s.batches?double(s.requests)/s.batches:0)<<",\"peak_batch\":"<<peak<<",\"queue_high_water\":"<<s.queue_high_water<<"}";}
int main(int n,char**v){try{
  O x=Parse(n,v);auto started=std::chrono::steady_clock::now();fs::create_directories(x.out);auto os=Read(x.openings);
#ifdef HEX_CANDIDATE_MATCH_FAKE
  DeterministicFakeEvaluator fake;
  Evaluator cand=[&fake](const Board& b){return fake(b);}, champ=cand;
#else
  OnnxCudaBatchEvaluator ce(x.candidate),he(x.champion);
  SharedInferenceService cs(ce,{size_t(x.max_batch),256,std::chrono::microseconds(x.wait)}),hs(he,{size_t(x.max_batch),256,std::chrono::microseconds(x.wait)});
  Evaluator cand=EncodedBoardEvaluator(cs),champ=EncodedBoardEvaluator(hs);
#endif
  std::atomic<size_t>next=0;std::atomic<uint64_t> searches=0,simulations=0;std::atomic<bool> stop=false,monitor_stop=false;Activity activity;std::exception_ptr err;std::mutex mu;
#ifndef HEX_CANDIDATE_MATCH_FAKE
  const fs::path status_path=fs::path(x.out).parent_path()/"runner-status.json";
  WriteStatus(status_path,"running",activity,cs,hs);
  std::thread monitor([&]{
    unsigned ticks=0;
    while(!monitor_stop.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(1)); if(monitor_stop.load()) break;
      const auto c=cs.Health(),h=hs.Health();
      const bool fatal=c.failed||h.failed;
      const bool stalled=(c.outstanding_requests&&c.seconds_since_last_success>120.0)||(h.outstanding_requests&&h.seconds_since_last_success>120.0);
      if(fatal||stalled) {
        const std::string why=fatal?"inference service fatal: "+(c.failed?c.failure_reason:h.failure_reason):"inference service heartbeat stalled with outstanding requests";
        stop.store(true); WriteStatus(status_path,stalled?"watchdog_failed":"service_failed",activity,cs,hs,why);
        std::cerr<<"match watchdog: "<<why<<'\n';
        if(stalled) std::_Exit(70);
      }
      if (++ticks % 30 == 0) {
        WriteStatus(status_path,"running",activity,cs,hs);
        std::cerr<<"match progress: pairs="<<activity.completed_pairs.load()<<" active_pairs="<<activity.active_pairs.load()<<" active_games="<<activity.active_games.load()<<" candidate_q="<<c.queue_depth<<"/"<<c.inflight_requests<<" champion_q="<<h.queue_depth<<"/"<<h.inflight_requests<<'\n';
      }
    }
  });
#endif
  auto w=[&]{try{for(;;){if(stop.load())break;auto i=next++;if(i>=os.size())break;Run(os[i],x,cand,champ,searches,simulations,activity);}}catch(...){stop.store(true);std::lock_guard l(mu);if(!err)err=std::current_exception();}};
  std::vector<std::thread>ts;for(int i=0;i<x.concurrency;i++)ts.emplace_back(w);for(auto&t:ts)t.join();
#ifndef HEX_CANDIDATE_MATCH_FAKE
  monitor_stop.store(true);monitor.join();auto cstats=cs.Stats(),hstats=hs.Stats();cs.Shutdown();hs.Shutdown();
#endif
  if(err) {
#ifndef HEX_CANDIDATE_MATCH_FAKE
    WriteStatus(status_path,"failed",activity,cs,hs,"pair/game worker exception");
#endif
    std::rethrow_exception(err);
  } double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
#ifndef HEX_CANDIDATE_MATCH_FAKE
  WriteStatus(status_path,"complete",activity,cs,hs);
#endif
  std::cout<<"{\"status\":\"complete\",\"pairs\":"<<os.size()<<",\"games\":"<<2*os.size()<<",\"bridge_controller\":\""<<Name(x.bridge_mode)<<"\",\"searched_moves\":"<<searches<<",\"simulations\":"<<simulations<<",\"seconds\":"<<sec<<",\"games_per_second\":"<<(2*os.size()/sec)<<",\"simulations_per_second\":"<<(simulations.load()/sec);
#ifndef HEX_CANDIDATE_MATCH_FAKE
  std::cout<<",\"candidate_service\":";Stats(std::cout,cstats);std::cout<<",\"champion_service\":";Stats(std::cout,hstats);
#else
  std::cout<<",\"fake_evaluator\":true";
#endif
  std::cout<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<"match error: "<<e.what()<<'\n';return 1;}}
