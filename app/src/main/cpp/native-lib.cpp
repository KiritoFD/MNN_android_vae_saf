#include <jni.h>
#include <string>
#include <vector>
#include <fstream>
#include <cstdarg>
#include <random>
#include <chrono> // 新增：用于精确计时
#include <android/log.h>
#include <android/bitmap.h>

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/ImageProcess.hpp>

#define LOG_TAG "MNN_NATIVE"

static std::string g_LogFilePath = "";

std::string GetTimeStr() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
    return std::string(buf);
}

void WriteLog(const char* level, const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    std::string msg = buffer;

    __android_log_print(strcmp(level, "INFO") == 0 ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, LOG_TAG, "%s", buffer);

    if (!g_LogFilePath.empty()) {
        std::ofstream outfile;
        outfile.open(g_LogFilePath, std::ios_base::app);
        if (outfile.is_open()) {
            outfile << "[" << GetTimeStr() << "] [" << level << "] " << msg << std::endl;
            outfile.close();
        }
    }
}

#define LOGI(...) WriteLog("INFO", __VA_ARGS__)
#define LOGE(...) WriteLog("ERROR", __VA_ARGS__)

using namespace MNN;
using namespace MNN::CV;

void generateNoise(float* buffer, int size) {
    std::mt19937 gen(42);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (int i = 0; i < size; i++) {
        buffer[i] = d(gen);
    }
}

// 简单的计时器类
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    // 返回毫秒
    float elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;
    }
};

class SAFlowPipeline {
public:
    std::shared_ptr<Interpreter> encNet, flowNet, decNet;
    Session *encSess = nullptr, *flowSess = nullptr, *decSess = nullptr;

    Tensor *encInput = nullptr, *encOutput = nullptr;
    Tensor *flowXt = nullptr, *flowXCond = nullptr, *flowT = nullptr, *flowS = nullptr, *flowOutput = nullptr;
    Tensor *decInput = nullptr, *decOutput = nullptr;

    std::shared_ptr<ImageProcess> imgProcessor;

    SAFlowPipeline(const std::string& cacheDir) {
        LOGI("========== 引擎初始化 ==========");
        LOGI("目标设备: OpenCL (GPU)");
        LOGI("精度配置: FP16 (Precision_Low)");
        LOGI("线程数: 4");

        // 加载模型...
        if (!loadModel(cacheDir + "/Encoder.mnn", encNet, encSess, "Encoder")) return;
        encInput = encNet->getSessionInput(encSess, "input");
        encOutput = encNet->getSessionOutput(encSess, "output");

        if (!loadModel(cacheDir + "/Flow.mnn", flowNet, flowSess, "Flow")) return;
        flowXt = flowNet->getSessionInput(flowSess, "x_t");
        flowXCond = flowNet->getSessionInput(flowSess, "x_cond");
        flowT = flowNet->getSessionInput(flowSess, "t");
        flowS = flowNet->getSessionInput(flowSess, "s");
        flowOutput = flowNet->getSessionOutput(flowSess, "output");

        if (!loadModel(cacheDir + "/Decoder.mnn", decNet, decSess, "Decoder")) return;
        decInput = decNet->getSessionInput(decSess, "input");
        decOutput = decNet->getSessionOutput(decSess, "output");

        ImageProcess::Config imgConfig;
        imgConfig.sourceFormat = RGBA;
        imgConfig.destFormat   = RGB;
        float mean[3]   = {127.5f, 127.5f, 127.5f};
        float normal[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
        ::memcpy(imgConfig.mean, mean, sizeof(mean));
        ::memcpy(imgConfig.normal, normal, sizeof(normal));
        imgProcessor = std::shared_ptr<ImageProcess>(ImageProcess::create(imgConfig));

        LOGI("初始化完成。");
    }

    ~SAFlowPipeline() {
        if(encNet) encNet->releaseSession(encSess);
        if(flowNet) flowNet->releaseSession(flowSess);
        if(decNet) decNet->releaseSession(decSess);
    }

    bool isValid() { return encSess && flowSess && decSess; }

    bool run(JNIEnv* env, jobject inputBitmap, jobject outputBitmap, int styleIndex, int steps) {
        if (!isValid()) return false;

        Timer totalTimer;
        Timer stepTimer;

        // --- 1. Encoder ---
        AndroidBitmapInfo info;
        void* pixels;
        AndroidBitmap_getInfo(env, inputBitmap, &info);
        AndroidBitmap_lockPixels(env, inputBitmap, &pixels);

        imgProcessor->convert((const uint8_t*)pixels, 512, 512, 0, encInput);
        AndroidBitmap_unlockPixels(env, inputBitmap);

        encNet->runSession(encSess);
        float t_enc = stepTimer.elapsed();
        stepTimer.reset();

        // --- 2. Flow Loop ---
        int latentSize = 1 * 4 * 64 * 64;
        std::vector<float> latents(latentSize);
        generateNoise(latents.data(), latentSize);

        std::shared_ptr<Tensor> hostCond(new Tensor(flowXCond, Tensor::CAFFE));
        encOutput->copyToHostTensor(hostCond.get());
        flowXCond->copyFromHostTensor(hostCond.get());

        std::shared_ptr<Tensor> hostS(new Tensor(flowS, Tensor::CAFFE));
        hostS->host<int>()[0] = styleIndex;
        flowS->copyFromHostTensor(hostS.get());

        float dt = 1.0f / (float)steps;

        std::shared_ptr<Tensor> hostXt(new Tensor(flowXt, Tensor::CAFFE));
        std::shared_ptr<Tensor> hostT(new Tensor(flowT, Tensor::CAFFE));
        std::shared_ptr<Tensor> hostOut(new Tensor(flowOutput, Tensor::CAFFE));

        // 仅计算 Flow 网络本身的耗时，不包含数据搬运
        float t_flow_net_only = 0;

        for (int i = 0; i < steps; i++) {
            float t_curr = (float)i / steps;

            ::memcpy(hostXt->host<float>(), latents.data(), latentSize * sizeof(float));
            flowXt->copyFromHostTensor(hostXt.get());

            hostT->host<float>()[0] = t_curr;
            flowT->copyFromHostTensor(hostT.get());

            Timer netTimer;
            flowNet->runSession(flowSess);
            t_flow_net_only += netTimer.elapsed();

            flowOutput->copyToHostTensor(hostOut.get());
            float* v_ptr = hostOut->host<float>();

            for (int j = 0; j < latentSize; j++) {
                latents[j] = latents[j] + v_ptr[j] * dt;
            }
        }
        float t_flow_total = stepTimer.elapsed();
        stepTimer.reset();

        // --- 3. Decoder ---
        std::shared_ptr<Tensor> hostDecIn(new Tensor(decInput, Tensor::CAFFE));
        ::memcpy(hostDecIn->host<float>(), latents.data(), latentSize * sizeof(float));
        decInput->copyFromHostTensor(hostDecIn.get());

        decNet->runSession(decSess);
        float t_dec = stepTimer.elapsed();
        stepTimer.reset();

        // --- 4. Post Process ---
        AndroidBitmap_lockPixels(env, outputBitmap, &pixels);

        std::shared_ptr<Tensor> finalOut(Tensor::create<float>({1, 512, 512, 3}, nullptr, Tensor::TENSORFLOW));
        decOutput->copyToHostTensor(finalOut.get());

        float* outData = finalOut->host<float>();
        uint8_t* bmpData = (uint8_t*)pixels;
        int pixelCount = 512 * 512;

        for (int i = 0; i < pixelCount; i++) {
            float r = outData[i * 3 + 0];
            float g = outData[i * 3 + 1];
            float b = outData[i * 3 + 2];

            int R = (int)(r * 255.0f);
            int G = (int)(g * 255.0f);
            int B = (int)(b * 255.0f);

            bmpData[i * 4 + 0] = (uint8_t)(std::max(0, std::min(255, R)));
            bmpData[i * 4 + 1] = (uint8_t)(std::max(0, std::min(255, G)));
            bmpData[i * 4 + 2] = (uint8_t)(std::max(0, std::min(255, B)));
            bmpData[i * 4 + 3] = 255;
        }

        AndroidBitmap_unlockPixels(env, outputBitmap);
        float t_post = stepTimer.elapsed();

        LOGI("--- 性能统计 (Steps=%d) ---", steps);
        LOGI("1. Encoder     : %.2f ms", t_enc);
        LOGI("2. Flow Loop   : %.2f ms (NetOnly: %.2f ms)", t_flow_total, t_flow_net_only);
        LOGI("3. Decoder     : %.2f ms", t_dec);
        LOGI("4. PostProcess : %.2f ms", t_post);
        LOGI("Total Time     : %.2f ms", totalTimer.elapsed());
        LOGI("--------------------------");

        return true;
    }

private:
    bool loadModel(const std::string& path, std::shared_ptr<Interpreter>& net, Session*& sess, const char* name) {
        LOGI("[%s] 加载中: %s", name, path.c_str());
        std::ifstream f(path);
        if (!f.good()) { LOGE("[%s] 文件缺失!", name); return false; }

        net = std::shared_ptr<Interpreter>(Interpreter::createFromFile(path.c_str()));
        if (!net) return false;

        ScheduleConfig config;
        config.type = MNN_FORWARD_OPENCL; // 指定设备
        config.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
        config.numThread = 4;

        BackendConfig backendConfig;
        backendConfig.precision = BackendConfig::Precision_Low; // 指定精度
        config.backendConfig = &backendConfig;

        sess = net->createSession(config);
        return (sess != nullptr);
    }
};

static SAFlowPipeline* g_pipeline = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mnn_MainActivity_initEngine(JNIEnv* env, jobject thiz, jstring jCacheDir) {
    const char* path = env->GetStringUTFChars(jCacheDir, nullptr);
    std::string cacheDirStr = path;
    g_LogFilePath = cacheDirStr + "/native_debug.txt";

    if (g_pipeline) { delete g_pipeline; g_pipeline = nullptr; }

    try {
        g_pipeline = new SAFlowPipeline(cacheDirStr);
        if (g_pipeline->isValid()) {
            env->ReleaseStringUTFChars(jCacheDir, path);
            return JNI_TRUE;
        }
    } catch (...) {}

    env->ReleaseStringUTFChars(jCacheDir, path);
    return JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mnn_MainActivity_runStyleTransfer(
        JNIEnv* env, jobject thiz, jobject inputBitmap, jobject outputBitmap, jint styleIndex) {
    if (!g_pipeline) return false;
    return g_pipeline->run(env, inputBitmap, outputBitmap, styleIndex, 15);
}