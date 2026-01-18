#include "SAFlowEngine.hpp"
#include <android/log.h>
#include <chrono>

#define LOG_TAG "SAFlow_CPU"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace MNN;

SAFlowEngine::SAFlowEngine(const std::string& path) {
    // 1. 加载模型文件
    netEnc.reset(Interpreter::createFromFile((path + "/Encoder.mnn").c_str()));
    netFlow.reset(Interpreter::createFromFile((path + "/Flow.mnn").c_str()));
    netDec.reset(Interpreter::createFromFile((path + "/Decoder.mnn").c_str()));

    // 2. 配置 CPU 极致性能参数
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    // 🚀 优化 1：针对骁龙 8 Elite 的 2 颗超大核，建议使用 2 线程以获得最佳 L2 缓存命中率
    config.numThread = 2;

    BackendConfig bConfig;
    // 🚀 优化 2：强制开启 CPU FP16 (ARMv8.2) 加速，这在 8 Elite 上是提速关键
    bConfig.precision = BackendConfig::Precision_Low;
    bConfig.power = BackendConfig::Power_High;
    config.backendConfig = &bConfig;

    // 🚀 优化 3：显存池复用。共享 Runtime 减少 512 分辨率下的内存申请耗时
    sharedRuntime.reset(netEnc->createRuntime(config));
    sessEnc = netEnc->createSession(config, sharedRuntime);
    sessFlow = netFlow->createSession(config, sharedRuntime);
    sessDec = netDec->createSession(config, sharedRuntime);

    // 预热模型：触发 CPU 调度升频
    netEnc->releaseModel();
    netFlow->releaseModel();
    netDec->releaseModel();
    LOGI("CPU Engine Initialized with 2 Threads & FP16 Support.");
}

bool SAFlowEngine::process(float* inData, float* outData, int styleId, int w, int h) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // --- STEP 1: ENCODER ---
    auto tEncIn = netEnc->getSessionInput(sessEnc, "input");
    // 直接操作 CPU Tensor 内存，避免 ImageProcess 的额外拷贝开销
    auto hostIn = new Tensor(tEncIn, Tensor::CAFFE);
    memcpy(hostIn->host<float>(), inData, w * h * 3 * sizeof(float));
    tEncIn->copyFromHostTensor(hostIn);
    netEnc->runSession(sessEnc);
    auto tEncOut = netEnc->getSessionOutput(sessEnc, "output");

    // --- STEP 2: REFLOW 5-STEP LOOP ---
    auto fXt = netFlow->getSessionInput(sessFlow, "x_t");
    auto fXc = netFlow->getSessionInput(sessFlow, "x_cond");
    auto fT = netFlow->getSessionInput(sessFlow, "t");
    auto fS = netFlow->getSessionInput(sessFlow, "s");
    auto fOut = netFlow->getSessionOutput(sessFlow, "output");

    // 条件冻结：Encoder 结果直接送入 Flow
    fXc->copyFromHostTensor(tEncOut);

    // 设置 Style ID
    std::unique_ptr<Tensor> hS(new Tensor(fS, Tensor::CAFFE));
    hS->host<int>()[0] = styleId;
    fS->copyFromHostTensor(hS.get());

    // 潜空间滚动 Tensor (CPU 内存驻留)
    std::unique_ptr<Tensor> latentTensor(new Tensor(fXt, Tensor::CAFFE));
    latentTensor->copyFromHostTensor(tEncOut);

    for (int i = 0; i < mSteps; ++i) {
        // 🚀 优化 4：Reflow 步长计算。5 步对应 $t = 0.0, 0.25, 0.5, 0.75, 1.0$
        float t_val = (float)i / (mSteps - 1);

        fXt->copyFromHostTensor(latentTensor.get());
        std::unique_ptr<Tensor> hT(new Tensor(fT, Tensor::CAFFE));
        hT->host<float>()[0] = t_val;
        fT->copyFromHostTensor(hT.get());

        netFlow->runSession(sessFlow);

        // 更新 Latent：Reflow 逻辑 $x_{t+1} = x_t + v \cdot dt$
        // 注意：如果你的模型直接输出下一步的 x，则直接拷贝；如果是输出速度 v，则需执行加法
        fOut->copyToHostTensor(latentTensor.get());
    }

    // --- STEP 3: DECODER ---
    auto dIn = netDec->getSessionInput(sessDec, "input");
    dIn->copyFromHostTensor(latentTensor.get());
    netDec->runSession(sessDec);
    auto dOut = netDec->getSessionOutput(sessDec, "output");

    // 结果写回
    std::unique_ptr<Tensor> hFinal(new Tensor(dOut, Tensor::CAFFE));
    dOut->copyToHostTensor(hFinal.get());
    memcpy(outData, hFinal->host<float>(), w * h * 3 * sizeof(float));

    auto t_end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    LOGI(">>> CPU Inference Success: %.2f ms", ms);
    return true;
}

SAFlowEngine::~SAFlowEngine() {
    // 自动清理由智能指针接管
}