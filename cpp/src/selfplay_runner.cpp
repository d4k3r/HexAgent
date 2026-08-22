// Stage-7 physical, no-swap self-play.  Records are committed by rename only.
#include "hex_puct/batched_inference_service.hpp"
#include "hex_puct/onnx_evaluator.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

using namespace hex_puct;
namespace fs = std::filesystem;
constexpr char kSchema[] = "hex-selfplay-game-v1";
constexpr char kFrozenModelSha[] = "2205816d6477182ff5eea0a7af5f3d2b5b5935d80388e4a53fe2233a3731e12e";
struct Options { std::string model, output; uint64_t base_seed=1, start_id=0; int budget=128, concurrency=1, max_batch=1, wait_us=0, max_games=1, max_seconds=0; double c_puct=1.5; bool argmax_only=false; };
struct Sample { Color to_play; std::array<int,kBoardArea> visits; int selected; };
static uint64_t Mix(uint64_t x) { x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
static char C(Color c) { return c==Color::Black?'B':'W'; }
static uint64_t GameSeed(uint64_t base, uint64_t id) { return Mix(base ^ Mix(id)); }
static int SampleVisit(const std::array<int,kBoardArea>& v, std::mt19937_64& rng) {
  uint64_t total=0; for(int x:v) total+=x; if(!total) throw std::runtime_error("zero root visit total");
  const uint64_t r=rng()%total; uint64_t acc=0; for(int a=0;a<kBoardArea;++a) { acc+=v[a]; if(r<acc) return a; } throw std::runtime_error("sampling overflow");
}
static int Argmax(const std::array<int,kBoardArea>& v) { int a=0; for(int i=1;i<kBoardArea;++i) if(v[i]>v[a]) a=i; return a; }
static void JArray(std::ostream& o, const std::array<int,kBoardArea>& xs) { o<<'['; for(int i=0;i<kBoardArea;++i) { if(i)o<<','; o<<xs[i]; } o<<']'; }
static void IArray(std::ostream& o, const std::vector<int>& xs) { o<<'['; for(size_t i=0;i<xs.size();++i) { if(i)o<<','; o<<xs[i]; } o<<']'; }
static bool Complete(const fs::path& p) { return fs::exists(p); }
static std::string ConfigId(const Options& x) { std::ostringstream o; o<<"stage7-v1-b"<<x.budget<<"-cp"<<x.c_puct<<"-t20-1.0-argmax-noswap"; return o.str(); }
static void WriteGame(const fs::path& final, uint64_t id, uint64_t seed, const Options& x, const std::vector<int>& moves, const std::vector<Sample>& samples, Color winner) {
  const fs::path temp=final.string()+".partial"; std::ofstream o(temp); if(!o) throw std::runtime_error("cannot open output");
  o<<std::setprecision(17)<<"{\"schema\":\""<<kSchema<<"\",\"status\":\"complete\",\"game_id\":"<<id<<",\"base_seed\":"<<x.base_seed<<",\"game_seed\":"<<seed<<",\"seed_scheme\":\"splitmix64(base_seed XOR splitmix64(game_id))\",\"model_sha256\":\""<<kFrozenModelSha<<"\",\"evaluation_engine\":\""
#ifdef HEX_SELFPLAY_FAKE
   <<"fake_semantics_only"
#else
   <<"onnx_cuda_shared_service"
#endif
   <<"\",\"configuration_id\":\""<<ConfigId(x)<<"\",\"search_budget\":"<<x.budget<<",\"c_puct\":"<<x.c_puct<<",\"exploration\":{\"dirichlet\":false,\"temperature_plies\":20,\"temperature\":1,\"after\":\"argmax_lowest_action_tie\",\"argmax_only\":"<<(x.argmax_only?"true":"false")<<"},\"inference\":{\"concurrency\":"<<x.concurrency<<",\"max_batch\":"<<x.max_batch<<",\"wait_us\":"<<x.wait_us<<"},\"initial_state\":{\"side_to_move\":\"B\",\"swap\":false},\"moves\":"; IArray(o,moves); o<<",\"samples\":[";
  for(size_t i=0;i<samples.size();++i) { if(i)o<<','; const auto&s=samples[i]; o<<"{\"ply\":"<<i<<",\"side_to_move\":\""<<C(s.to_play)<<"\",\"root_visits\":"; JArray(o,s.visits); o<<",\"selected_move\":"<<s.selected<<",\"z\":"<<(s.to_play==winner?1:-1)<<'}'; }
  o<<"],\"winner\":\""<<C(winner)<<"\",\"game_length\":"<<moves.size()<<"}\n"; o.close(); fs::rename(temp,final);
}
static void PlayGame(uint64_t id, const Options& x, const Evaluator& evaluator) {
  const uint64_t seed=GameSeed(x.base_seed,id); std::mt19937_64 rng(seed); Board board; std::vector<int> moves; std::vector<Sample> samples;
  while(!board.LiteralWinner()) { const auto result=DeterministicPUCT(evaluator,{x.budget,x.c_puct}).Search(board); if(!result.selected_action || result.root_visits!=x.budget) throw std::runtime_error("incomplete nonterminal search"); const int action=(x.argmax_only || moves.size()>=20)?Argmax(result.raw_visits):SampleVisit(result.raw_visits,rng); if(!board.LegalMask()[action]) throw std::runtime_error("selected illegal root action"); samples.push_back({board.side_to_move(),result.raw_visits,action}); board.Play(action); moves.push_back(action); }
  WriteGame(fs::path(x.output)/"games"/("game-"+std::to_string(id)+".json"),id,seed,x,moves,samples,*board.LiteralWinner());
}
static Options Parse(int n,char**v) { Options x; for(int i=1;i<n;i+=2) { if(i+1>=n) throw std::runtime_error("missing option value"); std::string k=v[i], z=v[i+1]; if(k=="--model")x.model=z; else if(k=="--output")x.output=z; else if(k=="--base-seed")x.base_seed=std::stoull(z); else if(k=="--start-game-id")x.start_id=std::stoull(z); else if(k=="--budget")x.budget=std::stoi(z); else if(k=="--concurrency")x.concurrency=std::stoi(z); else if(k=="--max-batch")x.max_batch=std::stoi(z); else if(k=="--wait-us")x.wait_us=std::stoi(z); else if(k=="--max-games")x.max_games=std::stoi(z); else if(k=="--max-seconds")x.max_seconds=std::stoi(z); else if(k=="--c-puct")x.c_puct=std::stod(z); else if(k=="--argmax-only")x.argmax_only=std::stoi(z)!=0; else throw std::runtime_error("unknown option "+k); } if(x.model.empty()||x.output.empty()||x.budget<1||x.concurrency<1||x.max_batch<1||x.max_games<0) throw std::runtime_error("invalid required options"); return x; }
int main(int n,char**v) { try { Options x=Parse(n,v); fs::create_directories(fs::path(x.output)/"games");
#ifndef HEX_SELFPLAY_FAKE
  OnnxCudaBatchEvaluator batch(x.model); SharedInferenceService service(batch,{size_t(x.max_batch),256,std::chrono::microseconds(x.wait_us)});
#else
  DeterministicFakeEvaluator fake;
#endif
  std::atomic<uint64_t> next{x.start_id}; std::atomic<bool> stop=false; std::mutex err_mu; std::exception_ptr error; const auto deadline=x.max_seconds?std::optional(std::chrono::steady_clock::now()+std::chrono::seconds(x.max_seconds)):std::nullopt;
    auto worker=[&]{ try { while(!stop) { uint64_t id=next.fetch_add(1); if(id>=x.start_id+uint64_t(x.max_games)||(deadline&&std::chrono::steady_clock::now()>=*deadline)) break; fs::path out=fs::path(x.output)/"games"/("game-"+std::to_string(id)+".json"); if(!Complete(out)) {
#ifndef HEX_SELFPLAY_FAKE
      PlayGame(id,x,EncodedBoardEvaluator(service));
#else
      PlayGame(id,x,fake);
#endif
    } } } catch(...) { std::lock_guard l(err_mu); error=std::current_exception(); stop=true; } };
    std::vector<std::thread> threads; for(int i=0;i<x.concurrency;++i) threads.emplace_back(worker); for(auto&t:threads)t.join();
#ifndef HEX_SELFPLAY_FAKE
    service.Shutdown(); if(error) std::rethrow_exception(error); const auto s=service.Stats(); std::cout<<"{\"status\":\"complete\",\"schema\":\"stage7-run-v1\",\"requests\":"<<s.requests<<",\"batches\":"<<s.batches<<",\"mean_batch\":"<<(s.batches?double(s.requests)/s.batches:0)<<",\"output\":\""<<x.output<<"\"}\n";
#else
    if(error) std::rethrow_exception(error); std::cout<<"{\"status\":\"complete\",\"schema\":\"stage7-semantics-run-v1\",\"output\":\""<<x.output<<"\"}\n";
#endif
    return 0;
  } catch(const std::exception&e) { std::cerr<<"selfplay error: "<<e.what()<<'\n'; return 1; } }
