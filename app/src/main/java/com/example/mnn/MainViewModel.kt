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
                // 哪怕 createScaledBitmap 返回了硬件位图，这一步也会把它转回来
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