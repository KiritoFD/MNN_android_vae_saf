#ifndef SA_FLOW_ENGINE_HPP
#define SA_FLOW_ENGINE_HPP

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/ImageProcess.hpp>
#include <memory>
#include <vector>
#include <string>

class SAFlowEngine {
public:
    SAFlowEngine(const std::string& modelPath);
    ~SAFlowEngine();

    // 执行推理：5步 Reflow 逻辑
    bool process(float* inputPixels, float* outputPixels, int styleId, int width, int height);

private:
    // MNN 核心组件
    std::unique_ptr<MNN::Interpreter> netEnc, netFlow, netDec;
    MNN::Session *sessEnc = nullptr, *sessFlow = nullptr, *sessDec = nullptr;

    // 共享运行时 (内存池优化核心)
    std::shared_ptr<MNN::Runtime> sharedRuntime;

    // 参数配置
    const int mSteps = 5; // Reflow 5步
    const int mLatentSize = 1 * 4 * 64 * 64;

    void initSessions(const std::string& path);
};

#endif