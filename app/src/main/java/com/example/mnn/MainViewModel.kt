package com.example.mnn

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.InputStream

class MainViewModel(application: Application) : AndroidViewModel(application) {

    // UI 状态
    private val _uiState = MutableStateFlow(UiState())
    val uiState = _uiState.asStateFlow()

    // 引擎初始化状态
    private var isEngineReady = false

    init {
        // 1. 启动时异步初始化 MNN
        viewModelScope.launch(Dispatchers.Default) {
            try {
                // 这里的 init 是我们上一节写的 JNI 接口
                isEngineReady = MNNNative.init(getApplication<Application>().assets)
                _uiState.value = _uiState.value.copy(
                    statusMessage = if (isEngineReady) "模型加载完毕 (Snapdragon 8 Elite Ready)" else "模型加载失败"
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(statusMessage = "初始化错误: ${e.message}")
            }
        }
    }

    // 用户选择了图片
    fun onImageSelected(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val bitmap = loadAndScaleBitmap(uri)
            if (bitmap != null) {
                _uiState.value = _uiState.value.copy(
                    originalBitmap = bitmap,
                    resultBitmap = null, // 清空旧结果
                    statusMessage = "图片已就绪，请选择风格并生成"
                )
            }
        }
    }

    // 用户点击生成
    fun generate(styleId: Int) {
        if (!isEngineReady) return
        val input = _uiState.value.originalBitmap ?: return

        viewModelScope.launch(Dispatchers.Default) {
            _uiState.value = _uiState.value.copy(isProcessing = true, statusMessage = "正在推理 (Steps=20)...")

            // 准备一张空的 512x512 位图接收结果
            val output = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)

            val startTime = System.currentTimeMillis()

            // === 调用 JNI ===
            val success = MNNNative.generate(input, output, styleId, 20) // 20步

            val cost = System.currentTimeMillis() - startTime

            withContext(Dispatchers.Main) {
                if (success) {
                    _uiState.value = _uiState.value.copy(
                        isProcessing = false,
                        resultBitmap = output,
                        statusMessage = "完成! 耗时: ${cost}ms"
                    )
                } else {
                    _uiState.value = _uiState.value.copy(isProcessing = false, statusMessage = "推理失败")
                }
            }
        }
    }

    // 辅助：加载图片并缩放到 512x512
    private fun loadAndScaleBitmap(uri: Uri): Bitmap? {
        return try {
            val contentResolver = getApplication<Application>().contentResolver
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val original = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            // 强制缩放 (模型硬性要求)
            Bitmap.createScaledBitmap(original, 512, 512, true)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

// 简单的状态数据类
data class UiState(
    val statusMessage: String = "正在初始化模型...",
    val isProcessing: Boolean = false,
    val originalBitmap: Bitmap? = null,
    val resultBitmap: Bitmap? = null
)