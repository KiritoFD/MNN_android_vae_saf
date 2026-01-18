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
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(UiState())
    val uiState = _uiState.asStateFlow()

    var isEngineReady = false

    // 更新状态文本
    fun updateStatus(msg: String) {
        _uiState.value = _uiState.value.copy(statusMessage = msg)
    }

    // 设置处理中状态（锁定 UI）
    fun setProcessing(processing: Boolean) {
        _uiState.value = _uiState.value.copy(isProcessing = processing)
    }

    // 设置生成结果
    fun setResult(bitmap: Bitmap) {
        _uiState.value = _uiState.value.copy(resultBitmap = bitmap)
    }

    // 新增：设置 Flow 步数
    fun setSteps(steps: Int) {
        _uiState.value = _uiState.value.copy(steps = steps)
    }

    // 处理图片选择
    fun onImageSelected(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val bitmap = loadAndScaleBitmap(uri)
            if (bitmap != null) {
                _uiState.value = _uiState.value.copy(
                    originalBitmap = bitmap,
                    resultBitmap = null,
                    statusMessage = "图片已就绪"
                )
            } else {
                _uiState.value = _uiState.value.copy(statusMessage = "图片加载失败")
            }
        }
    }

    // 新增：替换模型文件功能
    fun replaceModelFile(uri: Uri, cacheDir: File, onComplete: (Boolean) -> Unit) {
        if (_uiState.value.isProcessing) {
            updateStatus("系统忙，请稍后...")
            return
        }

        updateStatus("正在上传新模型...")
        setProcessing(true)

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val contentResolver = getApplication<Application>().contentResolver
                val inputStream: InputStream? = contentResolver.openInputStream(uri)

                if (inputStream == null) {
                    withContext(Dispatchers.Main) {
                        updateStatus("无法读取文件")
                        setProcessing(false)
                        onComplete(false)
                    }
                    return@launch
                }

                // 强制覆盖名为 Flow.mnn 的文件
                val targetFile = File(cacheDir, "Flow.mnn")
                if (targetFile.exists()) {
                    targetFile.delete()
                }

                FileOutputStream(targetFile).use { output ->
                    inputStream.copyTo(output)
                }
                inputStream.close()

                withContext(Dispatchers.Main) {
                    updateStatus("模型写入完成，准备重载引擎...")
                    // 注意：这里不设为 false，等待 Engine 重启完成
                    onComplete(true)
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    updateStatus("上传失败: ${e.message}")
                    setProcessing(false)
                    onComplete(false)
                }
            }
        }
    }

    private fun loadAndScaleBitmap(uri: Uri): Bitmap? {
        return try {
            val contentResolver = getApplication<Application>().contentResolver
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val original = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            if (original != null) {
                // 1. 缩放
                val scaled = Bitmap.createScaledBitmap(original, 512, 512, true)
                // 2. 转换为 ARGB_8888 (软件位图) 以兼容 JNI
                if (scaled.config != Bitmap.Config.ARGB_8888) {
                    scaled.copy(Bitmap.Config.ARGB_8888, true)
                } else {
                    scaled
                }
            } else null
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

// UI 状态类，新增 steps 字段
data class UiState(
    val statusMessage: String = "正在初始化模型...",
    val isProcessing: Boolean = false,
    val originalBitmap: Bitmap? = null,
    val resultBitmap: Bitmap? = null,
    val steps: Int = 4 // 默认步数
)