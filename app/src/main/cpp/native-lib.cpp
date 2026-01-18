#include <jni.h>
#include <string>
#include <vector>
#include <fstream>
#include <android/log.h>
#include <android/bitmap.h>
#include <chrono>
#include <algorithm>
#include <memory>

// MNN 核心头文件
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/ImageProcess.hpp>

#define LOG_TAG "SAFlow_CPU_Final"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace MNN;

/**
 * SAFlow 引擎类：深度适配骁龙 8 Elite CPU
 */
class SAFlowEngine {
public:
    std::unique_ptr<Interpreter> netEnc, netFlow, netDec;
    Session *sessEnc = nullptr, *sessFlow = nullptr, *sessDec = nullptr;
    // RuntimeInfo 包含 Runtime 句柄和相关的配置是否生效的信息
    RuntimeInfo mRuntimeInfo;

    // 静态缓存：5步 Reflow 的时间步 t
    const std::vector<float> mStepsT = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};

    SAFlowEngine(const std::string& path) {
        LOGI(">>> Initializing SAFlow CPU Engine for Snapdragon 8 Elite <<<");

        // 1. 设置 CPU 推理配置
        ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;

        // 🚀 核心优化：骁龙 8 Elite 建议使用 2 线程
        config.numThread = 2;

        BackendConfig bConfig;
        // 🚀 核心优化：强制开启 FP16 以适配 Oryon 架构
        bConfig.precision = BackendConfig::Precision_Low;
        bConfig.power = BackendConfig::Power_High;
        config.backendConfig = &bConfig;

        // 2. 加载模型
        netEnc.reset(Interpreter::createFromFile((path + "/Encoder.mnn").c_str()));
        netFlow.reset(Interpreter::createFromFile((path + "/Flow.mnn").c_str()));
        netDec.reset(Interpreter::createFromFile((path + "/Decoder.mnn").c_str()));

        // 3. 内存池优化：修正此处参数类型错误
        // createRuntime 需要 std::vector<ScheduleConfig>，此处使用 {} 进行隐式转换
        mRuntimeInfo = MNN::Interpreter::createRuntime({config});

        // 使用共享的 RuntimeInfo 创建各模型的 Session，实现内存池和线程池共享
        sessEnc = netEnc->createSession(config, mRuntimeInfo);
        sessFlow = netFlow->createSession(config, mRuntimeInfo);
        sessDec = netDec->createSession(config, mRuntimeInfo);

        // 释放原始权重内存，仅保留运行 Session 所需内存
        netEnc->releaseModel();
        netFlow->releaseModel();
        netDec->releaseModel();
        LOGI(">>> CPU Engine Ready (FP16 + 2-Threads) <<<");
    }

    bool run(JNIEnv* env, jobject inBmp, jobject outBmp, int styleId) {
        if (!sessEnc || !sessFlow || !sessDec) return false;

        auto t_start = std::chrono::high_resolution_clock::now();

        // --- STEP 1: ENCODER ---
        auto tEncIn = netEnc->getSessionInput(sessEnc, "input");
        AndroidBitmapInfo info;
        void* pixels;
        AndroidBitmap_getInfo(env, inBmp, &info);
        AndroidBitmap_lockPixels(env, inBmp, &pixels);

        CV::ImageProcess::Config imgConfig;
        imgConfig.sourceFormat = CV::RGBA;
        imgConfig.destFormat = CV::RGB;
        float mean[3] = {127.5f, 127.5f, 127.5f};
        float normal[3] = {0.007843f, 0.007843f, 0.007843f};
        memcpy(imgConfig.mean, mean, sizeof(mean));
        memcpy(imgConfig.normal, normal, sizeof(normal));

        std::unique_ptr<CV::ImageProcess> processer(CV::ImageProcess::create(imgConfig));
        processer->convert((const uint8_t*)pixels, info.width, info.height, 0, tEncIn);
        AndroidBitmap_unlockPixels(env, inBmp);

        netEnc->runSession(sessEnc);
        auto tEncOut = netEnc->getSessionOutput(sessEnc, "output");

        // --- STEP 2: FLOW (5-STEP REFLOW LOOP) ---
        auto fXt = netFlow->getSessionInput(sessFlow, "x_t");
        auto fXc = netFlow->getSessionInput(sessFlow, "x_cond");
        auto fT = netFlow->getSessionInput(sessFlow, "t");
        auto fS = netFlow->getSessionInput(sessFlow, "s");
        auto fOut = netFlow->getSessionOutput(sessFlow, "output");

        fXc->copyFromHostTensor(tEncOut);

        std::unique_ptr<Tensor> hS(new Tensor(fS, Tensor::CAFFE));
        hS->host<int>()[0] = styleId;
        fS->copyFromHostTensor(hS.get());

        std::unique_ptr<Tensor> latent(new Tensor(fXt, Tensor::CAFFE));
        latent->copyFromHostTensor(tEncOut);

        for (int i = 0; i < 5; ++i) {
            fXt->copyFromHostTensor(latent.get());

            std::unique_ptr<Tensor> hT(new Tensor(fT, Tensor::CAFFE));
            hT->host<float>()[0] = mStepsT[i];
            fT->copyFromHostTensor(hT.get());

            netFlow->runSession(sessFlow);
            fOut->copyToHostTensor(latent.get());
        }

        // --- STEP 3: DECODER ---
        auto dIn = netDec->getSessionInput(sessDec, "input");
        dIn->copyFromHostTensor(latent.get());
        netDec->runSession(sessDec);
        auto dOut = netDec->getSessionOutput(sessDec, "output");

        // --- STEP 4: RESULT RENDER ---
        AndroidBitmap_lockPixels(env, outBmp, &pixels);
        std::unique_ptr<Tensor> hFinal(new Tensor(dOut, Tensor::CAFFE));
        dOut->copyToHostTensor(hFinal.get());

        float* data = hFinal->host<float>();
        uint8_t* rgba = (uint8_t*)pixels;
        for (int i = 0; i < 512 * 512; i++) {
            rgba[i * 4 + 0] = (uint8_t)std::clamp(data[i] * 255.0f, 0.0f, 255.0f);
            rgba[i * 4 + 1] = (uint8_t)std::clamp(data[i + 262144] * 255.0f, 0.0f, 255.0f);
            rgba[i * 4 + 2] = (uint8_t)std::clamp(data[i + 524288] * 255.0f, 0.0f, 255.0f);
            rgba[i * 4 + 3] = 255;
        }
        AndroidBitmap_unlockPixels(env, outBmp);

        auto end_time = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end_time - t_start).count();
        LOGI(">>> CPU SUCCESS! Inference Time: %.2f ms", ms);
        return true;
    }
};

static SAFlowEngine* g_engine = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mnn_MainActivity_initEngine(JNIEnv* env, jobject thiz, jstring jCacheDir) {
    const char* path = env->GetStringUTFChars(jCacheDir, nullptr);
    if (g_engine) { delete g_engine; g_engine = nullptr; }
    g_engine = new SAFlowEngine(path);
    env->ReleaseStringUTFChars(jCacheDir, path);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mnn_MainActivity_runStyleTransfer(JNIEnv* env, jobject thiz, jobject src, jobject dst, jint styleId) {
    if (!g_engine) return JNI_FALSE;
    return g_engine->run(env, src, dst, (int)styleId);
}