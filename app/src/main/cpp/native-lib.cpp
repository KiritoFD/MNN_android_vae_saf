#include <jni.h>
#include <string>
#include <vector>
#include <fstream>
#include <android/log.h>
#include <android/bitmap.h>
#include <chrono>
#include <algorithm>
#include <memory>
#include <ctime>

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/ImageProcess.hpp>

#define LOG_TAG "SAFlow_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace MNN;

static std::string g_log_path = "";

// 日志工具：同时输出到 Logcat 和文件
void WriteLog(const char* fmt, ...) {
    char buf[1024];
    va_list args; va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args); va_end(args);

    LOGI("%s", buf);

    if (!g_log_path.empty()) {
        std::ofstream os(g_log_path, std::ios::app);
        if (os.is_open()) {
            time_t now = time(0);
            tm *ltm = localtime(&now);
            os << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "] " << buf << std::endl;
        }
    }
}

class SAFlowEngine {
public:
    std::unique_ptr<Interpreter> netEnc, netFlow, netDec;
    Session *sessEnc = nullptr, *sessFlow = nullptr, *sessDec = nullptr;
    std::shared_ptr<CV::ImageProcess> imgProc;

    SAFlowEngine(const std::string& path) {
        g_log_path = path + "/sa_debug.txt";
        // 每次初始化清空旧日志
        std::ofstream(g_log_path, std::ios::trunc).close();

        WriteLog("=== ENGINE INIT: CPU SAFE MODE ===");
        WriteLog("Model Path: %s", path.c_str());

        // --- CPU 优化配置 ---
        ScheduleConfig config;
        config.type = MNN_FORWARD_CPU; // 强制 CPU
        config.numThread = 4;          // 4线程平衡性能与发热

        BackendConfig bConfig;
        bConfig.precision = BackendConfig::Precision_Low; // 开启 FP16 (ARMv8.2+)
        bConfig.power = BackendConfig::Power_High;        // 倾向使用大核
        bConfig.memory = BackendConfig::Memory_High;      // 空间换时间
        config.backendConfig = &bConfig;

        // 加载 Encoder
        netEnc.reset(Interpreter::createFromFile((path + "/Encoder.mnn").c_str()));
        if (netEnc) {
            sessEnc = netEnc->createSession(config);
            netEnc->releaseModel(); // 释放模型Buffer以节省内存
        } else {
            WriteLog("❌ Failed to load Encoder.mnn");
        }

        // 加载 Flow (注意：这里会读取最新的 Flow.mnn)
        netFlow.reset(Interpreter::createFromFile((path + "/Flow.mnn").c_str()));
        if (netFlow) {
            sessFlow = netFlow->createSession(config);
            netFlow->releaseModel();
        } else {
            WriteLog("❌ Failed to load Flow.mnn");
        }

        // 加载 Decoder
        netDec.reset(Interpreter::createFromFile((path + "/Decoder.mnn").c_str()));
        if (netDec) {
            sessDec = netDec->createSession(config);
            netDec->releaseModel();
        } else {
            WriteLog("❌ Failed to load Decoder.mnn");
        }

        WriteLog(">>> CPU Engine Ready (FP16, 4 Threads) <<<");
    }

    bool run(JNIEnv* env, jobject inBmp, jobject outBmp, int style, int steps) {
        if (!sessEnc || !sessFlow || !sessDec) {
            WriteLog("❌ Sessions not ready");
            return false;
        }

        auto t_all_start = std::chrono::high_resolution_clock::now();

        // --- STEP 1: ENCODER ---
        auto tEncIn = netEnc->getSessionInput(sessEnc, "input");

        // 锁定位图处理
        void* pixels;
        AndroidBitmap_lockPixels(env, inBmp, &pixels);
        if (!imgProc) {
            CV::ImageProcess::Config c;
            c.sourceFormat = CV::RGBA; c.destFormat = CV::RGB;
            // mean=[127.5, ...], normal=[1/127.5, ...]
            float m[3]={127.5f, 127.5f, 127.5f};
            float n[3]={0.007843f, 0.007843f, 0.007843f};
            memcpy(c.mean, m, sizeof(m));
            memcpy(c.normal, n, sizeof(n));
            imgProc.reset(CV::ImageProcess::create(c));
        }
        imgProc->convert((const uint8_t*)pixels, 512, 512, 0, tEncIn);
        AndroidBitmap_unlockPixels(env, inBmp);

        netEnc->runSession(sessEnc);
        auto tEncOut = netEnc->getSessionOutput(sessEnc, "output");

        // --- STEP 2: FLOW LOOP ---
        // 准备 Latent
        int size = 1 * 4 * 64 * 64; // shape: [1, 4, 64, 64]
        std::vector<float> latents(size);

        // Copy Encoder Output -> CPU -> Latents
        std::unique_ptr<Tensor> hostL(new Tensor(tEncOut, Tensor::CAFFE));
        tEncOut->copyToHostTensor(hostL.get());
        memcpy(latents.data(), hostL->host<float>(), size * sizeof(float));

        auto fXt = netFlow->getSessionInput(sessFlow, "x_t");
        auto fXc = netFlow->getSessionInput(sessFlow, "x_cond");
        auto fT = netFlow->getSessionInput(sessFlow, "t");
        auto fS = netFlow->getSessionInput(sessFlow, "s");
        auto fOut = netFlow->getSessionOutput(sessFlow, "output");

        // 设置 Condition (Encoder output)
        fXc->copyFromHostTensor(hostL.get());

        // 设置 Style ID
        std::unique_ptr<Tensor> hS(new Tensor(fS, Tensor::CAFFE));
        hS->host<int>()[0] = style;
        fS->copyFromHostTensor(hS.get());

        // 预分配 Buffer
        std::unique_ptr<Tensor> hXt(new Tensor(fXt, Tensor::CAFFE));
        std::unique_ptr<Tensor> hT(new Tensor(fT, Tensor::CAFFE));
        std::unique_ptr<Tensor> hV(new Tensor(fOut, Tensor::CAFFE));

        // 动态步数控制 (限制在 1~50 之间防止死机)
        int safe_steps = std::max(1, std::min(steps, 50));
        float dt = 1.0f / (float)safe_steps * 0.2f; // 简单缩放时间步长，保持总流形长度大致一致（可选逻辑）
        // 或者保持固定步长 dt=0.05，此时 steps 越多效果越强/变化越大
        // 这里采用原逻辑：步长固定 0.05
        float fixed_dt = 0.05f;

        for (int i = 0; i < safe_steps; i++) {
            // 输入当前的 latents
            memcpy(hXt->host<float>(), latents.data(), size * sizeof(float));
            fXt->copyFromHostTensor(hXt.get());

            // 输入时间 t
            hT->host<float>()[0] = (float)i * fixed_dt;
            fT->copyFromHostTensor(hT.get());

            // 推理
            netFlow->runSession(sessFlow);

            // 获取速度场 v
            fOut->copyToHostTensor(hV.get());
            float* v = hV->host<float>();

            // Euler 积分更新: x = x + v * dt
            for (int j = 0; j < size; j++) {
                latents[j] += v[j] * fixed_dt;
            }
        }

        // --- STEP 3: DECODER ---
        auto dIn = netDec->getSessionInput(sessDec, "input");
        std::unique_ptr<Tensor> hDecIn(new Tensor(dIn, Tensor::CAFFE));
        memcpy(hDecIn->host<float>(), latents.data(), size * sizeof(float));
        dIn->copyFromHostTensor(hDecIn.get());

        netDec->runSession(sessDec);
        auto dOut = netDec->getSessionOutput(sessDec, "output");

        // --- STEP 4: OUTPUT RENDER ---
        AndroidBitmap_lockPixels(env, outBmp, &pixels);
        std::unique_ptr<Tensor> hFinal(new Tensor(dOut, Tensor::CAFFE));
        dOut->copyToHostTensor(hFinal.get());

        float* data = hFinal->host<float>();
        uint8_t* rgba = (uint8_t*)pixels;
        int total_pixels = 512 * 512;

        // 简单的反归一化与排布
        for (int i = 0; i < total_pixels; i++) {
            // Channel 0, 1, 2 分别偏移 0, 262144, 524288
            float r = data[i];
            float g = data[i + total_pixels];
            float b = data[i + total_pixels * 2];

            rgba[i*4+0] = (uint8_t)std::clamp(r * 255.0f, 0.0f, 255.0f);
            rgba[i*4+1] = (uint8_t)std::clamp(g * 255.0f, 0.0f, 255.0f);
            rgba[i*4+2] = (uint8_t)std::clamp(b * 255.0f, 0.0f, 255.0f);
            rgba[i*4+3] = 255; // Alpha
        }
        AndroidBitmap_unlockPixels(env, outBmp);

        auto t_all_end = std::chrono::high_resolution_clock::now();
        float cost = std::chrono::duration<float, std::milli>(t_all_end - t_all_start).count();
        WriteLog("Success: steps=%d, cost=%.2f ms", safe_steps, cost);

        return true;
    }
};

// 全局引擎指针
static SAFlowEngine* g_engine = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mnn_MainActivity_initEngine(JNIEnv* env, jobject thiz, jstring jCacheDir) {
    const char* path = env->GetStringUTFChars(jCacheDir, nullptr);
    if (g_engine) {
        delete g_engine;
        g_engine = nullptr;
    }
    g_engine = new SAFlowEngine(path);
    env->ReleaseStringUTFChars(jCacheDir, path);
    return JNI_TRUE;
}

// 注意：增加了 steps 参数
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mnn_MainActivity_runStyleTransfer(JNIEnv* env, jobject thiz, jobject src, jobject dst, jint styleId, jint steps) {
    if (!g_engine) return JNI_FALSE;
    return g_engine->run(env, src, dst, (int)styleId, (int)steps);
}