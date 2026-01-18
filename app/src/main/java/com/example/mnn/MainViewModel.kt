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

    fun updateStatus(msg: String) {
        _uiState.value = _uiState.value.copy(statusMessage = msg)
    }

    fun setProcessing(processing: Boolean) {
        _uiState.value = _uiState.value.copy(isProcessing = processing)
    }

    fun setResult(bitmap: Bitmap) {
        _uiState.value = _uiState.value.copy(resultBitmap = bitmap)
    }

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

    /**
     * 新增功能：替换模型文件
     * 将用户选中的 Uri 内容复制到 cacheDir/Flow.mnn
     */
    fun replaceModelFile(uri: Uri, cacheDir: File, onComplete: (Boolean) -> Unit) {
        if (_uiState.value.isProcessing) {
            updateStatus("系统正忙，请稍后...")
            return
        }

        updateStatus("正在读取新模型文件...")
        setProcessing(true) // 锁定 UI

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val contentResolver = getApplication<Application>().contentResolver
                val inputStream: InputStream? = contentResolver.openInputStream(uri)

                if (inputStream == null) {
                    withContext(Dispatchers.Main) {
                        updateStatus("无法打开文件")
                        setProcessing(false)
                        onComplete(false)
                    }
                    return@launch
                }

                // 目标文件固定为 Flow.mnn，以便 C++ 引擎加载
                val targetFile = File(cacheDir, "Flow.mnn")
                if (targetFile.exists()) {
                    targetFile.delete()
                }

                FileOutputStream(targetFile).use { output ->
                    inputStream.copyTo(output)
                }
                inputStream.close()

                withContext(Dispatchers.Main) {
                    updateStatus("模型写入成功，准备重启引擎...")
                    // 注意：这里不立即 setProcessing(false)，等待引擎重启完成
                    onComplete(true)
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    updateStatus("模型更新失败: ${e.message}")
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
                // 2. 【关键】强制转换为 ARGB_8888 格式 (软件位图)，防止 HARDWARE Bitmap 导致 Native Crash
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

data class UiState(
    val statusMessage: String = "正在初始化模型...",
    val isProcessing: Boolean = false,
    val originalBitmap: Bitmap? = null,
    val resultBitmap: Bitmap? = null
)