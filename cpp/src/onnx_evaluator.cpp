#include "hex_puct/onnx_evaluator.hpp"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <stdexcept>
#include <string>
namespace hex_puct {
static void RequireCudaAvailable() {
  char** providers=nullptr; int count=0;
  Ort::ThrowOnError(Ort::GetApi().GetAvailableProviders(&providers,&count));
  bool found=false; for(int i=0;i<count;++i) found |= std::string(providers[i]) == "CUDAExecutionProvider";
  Ort::GetApi().ReleaseAvailableProviders(providers,count);
  if(!found) throw std::runtime_error("CUDAExecutionProvider is unavailable");
}
struct OnnxCpuEvaluator::Impl { Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"hex_puct_stage3"}; Ort::SessionOptions opts; Ort::Session session; Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault)}; Impl(const std::string&p):session([&]{opts.SetIntraOpNumThreads(1);opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);return Ort::Session(env,p.c_str(),opts);}()){} };
OnnxCpuEvaluator::OnnxCpuEvaluator(const std::string&p):impl_(std::make_unique<Impl>(p)) { if(impl_->session.GetInputCount()!=1||impl_->session.GetOutputCount()!=2)throw std::runtime_error("ONNX I/O count contract mismatch"); }
OnnxCpuEvaluator::~OnnxCpuEvaluator()=default;OnnxCpuEvaluator::OnnxCpuEvaluator(OnnxCpuEvaluator&&) noexcept=default;
Evaluation OnnxCpuEvaluator::Evaluate(const StudentTensor& in) const { std::array<float,726> data{};for(int i=0;i<726;++i)data[i]=in[i];std::array<int64_t,4> shape{1,6,11,11};auto input=Ort::Value::CreateTensor<float>(impl_->mem,data.data(),data.size(),shape.data(),shape.size()); const char* inname="state";const char* outnames[]={"policy_logits","value"};auto out=impl_->session.Run(Ort::RunOptions{nullptr},&inname,&input,1,outnames,2);auto policy=out[0].GetTensorData<float>();auto value=out[1].GetTensorData<float>();Evaluation e{};for(int i=0;i<121;++i)e.policy_logits[i]=policy[i];e.value=value[0];return e; }
struct OnnxCudaEvaluator::Impl { Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"hex_puct_stage5"}; Ort::SessionOptions opts; Ort::Session session; Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault)}; Impl(const std::string&p):session([&]{opts.SetIntraOpNumThreads(1);OrtCUDAProviderOptions cuda{};cuda.device_id=0;opts.AppendExecutionProvider_CUDA(cuda);return Ort::Session(env,p.c_str(),opts);}()){} };
OnnxCudaEvaluator::OnnxCudaEvaluator(const std::string&p):impl_(std::make_unique<Impl>(p)){RequireCudaAvailable();} OnnxCudaEvaluator::~OnnxCudaEvaluator()=default;
Evaluation OnnxCudaEvaluator::Evaluate(const StudentTensor& in) const { std::array<float,726> d{};for(int i=0;i<726;++i)d[i]=in[i];std::array<int64_t,4>s{1,6,11,11};auto x=Ort::Value::CreateTensor<float>(impl_->mem,d.data(),d.size(),s.data(),4);const char* i="state";const char* o[]={"policy_logits","value"};auto y=impl_->session.Run(Ort::RunOptions{nullptr},&i,&x,1,o,2);Evaluation e{};auto p=y[0].GetTensorData<float>();for(int k=0;k<121;++k)e.policy_logits[k]=p[k];e.value=y[1].GetTensorData<float>()[0];return e; }
struct OnnxCudaBatchEvaluator::Impl { Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"hex_puct_stage6"}; Ort::SessionOptions opts; Ort::Session session; Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault)}; Impl(const std::string&p):session([&]{opts.SetIntraOpNumThreads(1);OrtCUDAProviderOptions cuda{};cuda.device_id=0;opts.AppendExecutionProvider_CUDA(cuda);return Ort::Session(env,p.c_str(),opts);}()){} };
OnnxCudaBatchEvaluator::OnnxCudaBatchEvaluator(const std::string&p):impl_(std::make_unique<Impl>(p)){if(impl_->session.GetInputCount()!=1||impl_->session.GetOutputCount()!=2)throw std::runtime_error("ONNX I/O count contract mismatch"); RequireCudaAvailable();}
OnnxCudaBatchEvaluator::~OnnxCudaBatchEvaluator()=default;
std::vector<Evaluation> OnnxCudaBatchEvaluator::EvaluateBatch(const std::vector<StudentTensor>& in) const { if(in.empty()) return {}; std::vector<float>d(in.size()*726);for(size_t b=0;b<in.size();++b)for(int i=0;i<726;++i)d[b*726+i]=in[b][i];std::array<int64_t,4>s{static_cast<int64_t>(in.size()),6,11,11};auto x=Ort::Value::CreateTensor<float>(impl_->mem,d.data(),d.size(),s.data(),4);const char* i="state";const char* o[]={"policy_logits","value"};auto y=impl_->session.Run(Ort::RunOptions{nullptr},&i,&x,1,o,2);auto p=y[0].GetTensorData<float>();auto v=y[1].GetTensorData<float>();std::vector<Evaluation> result(in.size());for(size_t b=0;b<in.size();++b){for(int k=0;k<121;++k)result[b].policy_logits[k]=p[b*121+k];result[b].value=v[b];}return result; }
}
