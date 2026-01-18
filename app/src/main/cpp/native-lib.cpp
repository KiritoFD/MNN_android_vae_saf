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

#define LOG_TAG "SAFlow_Debug"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace MNN;

static std::string g_log_path = "";

// 增强型日志：带时间戳，确保写入文件
void WriteLog(const char* fmt, ...) {
    char buf[1024];
    va_list args; va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args); va_end(args);

    // 打印到 Logcat
    LOGI("%s", buf);

    // 写入文件
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
        // 清空旧日志
        std::ofstream(g_log_path, std::ios::trunc).close();

        WriteLog("==============================================");
        WriteLog(">>> ENGINE INIT: MNN 3.3 | Snapdragon 8 Elite <<<");
        WriteLog("Model Path: %s", path.c_str());

        ScheduleConfig config;
        config.type = MNN_FORWARD_OPENGL; // 尝试强制 OpenGL
        config.numThread = 1;

        BackendConfig bConfig;
        bConfig.precision = BackendConfig::Precision_Low;
        bConfig.power = BackendConfig::Power_High;
        config.backendConfig = &bConfig;
        config.mode = MNN_GPU_TUNING_WIDE;

        // 加载 Encoder
        netEnc.reset(Interpreter::createFromFile((path + "/Encoder.mnn").c_str()));
        if (!netEnc) WriteLog("❌ Failed to load Encoder.mnn");
        sessEnc = netEnc->createSession(config);
        checkBackend(netEnc, sessEnc, "Encoder");

        // 加载 Flow
        netFlow.reset(Interpreter::createFromFile((path + "/Flow.mnn").c_str()));
        if (!netFlow) WriteLog("❌ Failed to load Flow.mnn");
        sessFlow = netFlow->createSession(config);
        checkBackend(netFlow, sessFlow, "Flow_Iter");

        // 加载 Decoder
        netDec.reset(Interpreter::createFromFile((path + "/Decoder.mnn").c_str()));
        if (!netDec) WriteLog("❌ Failed to load Decoder.mnn");
        sessDec = netDec->createSession(config);
        checkBackend(netDec, sessDec, "Decoder");

        // 释放权重内存
        netEnc->releaseModel();
        netFlow->releaseModel();
        netDec->releaseModel();
        WriteLog(">>> Engine Ready <<<");
    }

    // 详尽监控后端：确定是否 Fallback
    void checkBackend(std::unique_ptr<Interpreter>& net, Session* sess, const char* name) {
        if (!sess) {
            WriteLog("[%s] ❌ Session Creation FAILED!", name);
            return;
        }
        float bType = -1.0f;
        net->getSessionInfo(sess, MNN::Interpreter::BACKEND_INFO, &bType);

        int type = (int)bType;
        const char* typeStr = "UNKNOWN";
        if (type == MNN_FORWARD_CPU) typeStr = "CPU (Fallback! ❌)";
        else if (type == MNN_FORWARD_OPENCL) typeStr = "OPENCL (GPU ✅)";
        else if (type == MNN_FORWARD_OPENGL) typeStr = "OPENGL (GPU ✅)";
        else if (type == MNN_FORWARD_VULKAN) typeStr = "VULKAN (GPU ✅)";

        WriteLog("[%s] Backend Reported Code: %d", name, type);
        WriteLog("[%s] Backend Actual Identification: %s", name, typeStr);
    }

    bool run(JNIEnv* env, jobject inBmp, jobject outBmp, int style) {
        if (!sessEnc || !sessFlow || !sessDec) {
            WriteLog("❌ Cannot run: Sessions not ready");
            return false;
        }

        auto t_all_start = std::chrono::high_resolution_clock::now();

        // --- STEP 1: ENCODER ---
        auto t_step_start = std::chrono::high_resolution_clock::now();
        auto tEncIn = netEnc->getSessionInput(sessEnc, "input");
        void* pixels;
        AndroidBitmap_lockPixels(env, inBmp, &pixels);
        if (!imgProc) {
            CV::ImageProcess::Config c;
            c.sourceFormat = CV::RGBA; c.destFormat = CV::RGB;
            float m[3]={127.5f, 127.5f, 127.5f}, n[3]={0.007843f, 0.007843f, 0.007843f};
            memcpy(c.mean, m, sizeof(m)); memcpy(c.normal, n, sizeof(n));
            imgProc.reset(CV::ImageProcess::create(c));
        }
        imgProc->convert((const uint8_t*)pixels, 512, 512, 0, tEncIn);
        AndroidBitmap_unlockPixels(env, inBmp);

        netEnc->runSession(sessEnc);
        auto tEncOut = netEnc->getSessionOutput(sessEnc, "output");

        auto t_step_end = std::chrono::high_resolution_clock::now();
        WriteLog("[Step 1] Encoder Run Time: %.2f ms", std::chrono::duration<float, std::milli>(t_step_end - t_step_start).count());

        // --- STEP 2: FLOW (Loop) ---
        t_step_start = std::chrono::high_resolution_clock::now();
        int size = 1 * 4 * 64 * 64;
        std::vector<float> latents(size);

        // 监控第一次同步开销
        auto t_sync_start = std::chrono::high_resolution_clock::now();
        std::unique_ptr<Tensor> hostL(new Tensor(tEncOut, Tensor::CAFFE));
        tEncOut->copyToHostTensor(hostL.get());
        auto t_sync_end = std::chrono::high_resolution_clock::now();
        WriteLog("[Step 2] Initial Sync (GPU->CPU) Time: %.2f ms", std::chrono::duration<float, std::milli>(t_sync_end - t_sync_start).count());

        memcpy(latents.data(), hostL->host<float>(), size * sizeof(float));

        auto fXt = netFlow->getSessionInput(sessFlow, "x_t");
        auto fXc = netFlow->getSessionInput(sessFlow, "x_cond");
        auto fT = netFlow->getSessionInput(sessFlow, "t");
        auto fS = netFlow->getSessionInput(sessFlow, "s");
        auto fOut = netFlow->getSessionOutput(sessFlow, "output");

        fXc->copyFromHostTensor(hostL.get());
        std::unique_ptr<Tensor> hS(new Tensor(fS, Tensor::CAFFE));
        hS->host<int>()[0] = style;
        fS->copyFromHostTensor(hS.get());

        std::unique_ptr<Tensor> hXt(new Tensor(fXt, Tensor::CAFFE));
        std::unique_ptr<Tensor> hT(new Tensor(fT, Tensor::CAFFE));
        std::unique_ptr<Tensor> hV(new Tensor(fOut, Tensor::CAFFE));

        float loop_sync_total = 0;
        for (int i = 0; i < 4; i++) {
            memcpy(hXt->host<float>(), latents.data(), size * sizeof(float));
            fXt->copyFromHostTensor(hXt.get());
            hT->host<float>()[0] = (float)i * 0.05f;
            fT->copyFromHostTensor(hT.get());

            netFlow->runSession(sessFlow);

            auto t_lsync_start = std::chrono::high_resolution_clock::now();
            fOut->copyToHostTensor(hV.get());
            auto t_lsync_end = std::chrono::high_resolution_clock::now();
            loop_sync_total += std::chrono::duration<float, std::milli>(t_lsync_end - t_lsync_start).count();

            float* v = hV->host<float>();
            for (int j = 0; j < size; j++) latents[j] += v[j] * 0.05f;

            if(i % 5 == 0) WriteLog("... Flow Iteration %d/20 done", i);
        }
        t_step_end = std::chrono::high_resolution_clock::now();
        WriteLog("[Step 2] Total Flow Loop Time: %.2f ms (Sync Overhead: %.2f ms)",
                 std::chrono::duration<float, std::milli>(t_step_end - t_step_start).count(), loop_sync_total);

        // --- STEP 3: DECODER ---
        t_step_start = std::chrono::high_resolution_clock::now();
        auto dIn = netDec->getSessionInput(sessDec, "input");
        std::unique_ptr<Tensor> hDecIn(new Tensor(dIn, Tensor::CAFFE));
        memcpy(hDecIn->host<float>(), latents.data(), size * sizeof(float));
        dIn->copyFromHostTensor(hDecIn.get());

        netDec->runSession(sessDec);
        auto dOut = netDec->getSessionOutput(sessDec, "output");
        t_step_end = std::chrono::high_resolution_clock::now();
        WriteLog("[Step 3] Decoder Run Time: %.2f ms", std::chrono::duration<float, std::milli>(t_step_end - t_step_start).count());

        // --- STEP 4: OUTPUT ---
        t_step_start = std::chrono::high_resolution_clock::now();
        AndroidBitmap_lockPixels(env, outBmp, &pixels);
        std::unique_ptr<Tensor> hFinal(new Tensor(dOut, Tensor::CAFFE));
        dOut->copyToHostTensor(hFinal.get());

        float* data = hFinal->host<float>();
        uint8_t* rgba = (uint8_t*)pixels;
        for (int i = 0; i < 512*512; i++) {
            rgba[i*4+0] = (uint8_t)std::clamp(data[i]*255.0f, 0.0f, 255.0f);
            rgba[i*4+1] = (uint8_t)std::clamp(data[i+262144]*255.0f, 0.0f, 255.0f);
            rgba[i*4+2] = (uint8_t)std::clamp(data[i+524288]*255.0f, 0.0f, 255.0f);
            rgba[i*4+3] = 255;
        }
        AndroidBitmap_unlockPixels(env, outBmp);
        t_step_end = std::chrono::high_resolution_clock::now();
        WriteLog("[Step 4] Bitmap Render Time: %.2f ms", std::chrono::duration<float, std::milli>(t_step_end - t_step_start).count());

        auto t_all_end = std::chrono::high_resolution_clock::now();
        WriteLog(">>> TOTAL INFERENCE TIME: %.2f ms", std::chrono::duration<float, std::milli>(t_all_end - t_all_start).count());
        WriteLog("==============================================");
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