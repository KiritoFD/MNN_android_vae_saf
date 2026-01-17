#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <android/bitmap.h>

// MNN 核心头文件
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/ImageProcess.hpp>

#define LOG_TAG "SD_NATIVE"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace MNN;
using namespace MNN::CV; // 必须引入 CV 命名空间，解决 ImageProcess 报错

// 引擎单例类
class SDEngine {
public:
    std::shared_ptr<Interpreter> net;
    Session* session = nullptr;
    Tensor* inputTensor = nullptr;
    std::shared_ptr<ImageProcess> imgProcessor;

    SDEngine(const char* modelPath) {
        // 1. 加载模型
        net = std::shared_ptr<Interpreter>(Interpreter::createFromFile(modelPath));
        if (!net) {
            LOGE("无法从路径加载模型: %s", modelPath);
            return;
        }

        // 2. 配置调度 (Vulkan 优先)
        ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        config.numThread = 4;

        BackendConfig backendConfig;
        backendConfig.precision = BackendConfig::Precision_Low; // 2026 核心建议：FP16 加速
        config.backendConfig = &backendConfig;

        session = net->createSession(config);
        inputTensor = net->getSessionInput(session, nullptr);

        // 3. 配置图像处理器 (将 Android RGBA 转换为模型 RGB)
        ImageProcess::Config imgConfig;
        imgConfig.sourceFormat = RGBA; // Android Bitmap 格式
        imgConfig.destFormat   = RGB;  // Transformer 模型输入格式

        // 归一化参数: (x - 127.5) / 127.5 => 映射到 [-1, 1]
        float mean[3]   = {127.5f, 127.5f, 127.5f};
        float normals[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
        ::memcpy(imgConfig.mean, mean, sizeof(mean));
        ::memcpy(imgConfig.normal, normals, sizeof(normals));

        imgProcessor = std::shared_ptr<ImageProcess>(ImageProcess::create(imgConfig));
        LOGI("MNN 引擎初始化完成，已开启 Vulkan 加速");
    }

    ~SDEngine() {
        if (net && session) {
            net->releaseSession(session);
        }
    }
};

// 全局静态指针
static SDEngine* g_engine = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_sdapp_MainActivity_initEngine(JNIEnv* env, jobject thiz, jstring jModelPath) {
    const char* path = env->GetStringUTFChars(jModelPath, nullptr);
    if (g_engine) delete g_engine;
    g_engine = new SDEngine(path);
    env->ReleaseStringUTFChars(jModelPath, path);
    return g_engine->net != nullptr;
}
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_sdapp_MainActivity_runStyleTransfer(
        JNIEnv* env, jobject thiz, jobject inputBitmap, jobject outputBitmap, jint styleIndex) {

    if (!g_engine || !g_engine->session) return false;

    AndroidBitmapInfo info;
    void* inputPixels = nullptr;
    void* outputPixels = nullptr;

    // 1. 锁定 Bitmap 内存
    if (AndroidBitmap_getInfo(env, inputBitmap, &info) < 0) return false;
    if (AndroidBitmap_lockPixels(env, inputBitmap, &inputPixels) < 0) return false;
    if (AndroidBitmap_lockPixels(env, outputBitmap, &outputPixels) < 0) {
        AndroidBitmap_unlockPixels(env, inputBitmap);
        return false;
    }

    int width = info.width;
    int height = info.height;

    // 2. 预处理 (Input Path)：这里 ImageProcess 依然可用，因为方向是 Raw -> Tensor
    g_engine->imgProcessor->convert((uint8_t*)inputPixels, width, height, 0, g_engine->inputTensor);

    // 3. 执行推理
    LOGI("正在执行 Transformer 风格迁移...");
    g_engine->net->runSession(g_engine->session);

    // 4. 后处理 (Output Path)：【核心修改】手动处理，避开编译错误
    auto outputTensorGPU = g_engine->net->getSessionOutput(g_engine->session, nullptr);

    // 4.1 创建一个临时的 Host Tensor，强制指定格式为 NHWC (便于像素遍历)
    // 假设模型输出是 RGB (3通道)
    std::shared_ptr<Tensor> hostTensor(Tensor::create<float>({1, height, width, 3}, nullptr, Tensor::TENSORFLOW));

    // 4.2 从 GPU 拷贝到 Host，MNN 会自动处理格式转换 (NC4HW4 -> NHWC)
    outputTensorGPU->copyToHostTensor(hostTensor.get());

    // 4.3 手动遍历像素进行 反归一化 和 格式转换 (RGB Float -> RGBA Uint8)
    float* srcData = hostTensor->host<float>();
    uint8_t* dstData = (uint8_t*)outputPixels;
    int pixelCount = width * height;

    for (int i = 0; i < pixelCount; i++) {
        // 读取 RGB (Float -1.0 ~ 1.0)
        float r = srcData[i * 3 + 0];
        float g = srcData[i * 3 + 1];
        float b = srcData[i * 3 + 2];

        // 反归一化: (val + 1) * 127.5
        int R = (int)((r * 127.5f) + 127.5f);
        int G = (int)((g * 127.5f) + 127.5f);
        int B = (int)((b * 127.5f) + 127.5f);

        // 防溢出 Clamp
        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);

        // 写入 RGBA
        dstData[i * 4 + 0] = (uint8_t)R;
        dstData[i * 4 + 1] = (uint8_t)G;
        dstData[i * 4 + 2] = (uint8_t)B;
        dstData[i * 4 + 3] = 255; // Alpha 通道设为不透明
    }

    // 5. 解锁内存
    AndroidBitmap_unlockPixels(env, inputBitmap);
    AndroidBitmap_unlockPixels(env, outputBitmap);

    LOGI("推理完成，Bitmap 已更新");
    return true;
}