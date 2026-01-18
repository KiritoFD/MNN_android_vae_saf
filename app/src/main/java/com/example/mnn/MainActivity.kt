package com.example.mnn

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {

    external fun initEngine(cacheDir: String): Boolean
    external fun runStyleTransfer(src: Bitmap, dst: Bitmap, styleId: Int): Boolean

    private val logFile by lazy { File(cacheDir, "java_debug.txt") }
    private val crashFile by lazy { File(cacheDir, "crash_log.txt") } // 专门记录闪退
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 【核武器】全局崩溃捕获器
        // 只要 App 闪退，就会把错误堆栈写进 crash_log.txt
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            handleCrash(throwable)
        }

        lifecycleScope.launch(Dispatchers.IO) {
            safeLoadLibrariesAndModels()
        }

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    StyleTransferScreen(
                        viewModel = viewModel,
                        onGenerate = { styleId -> runGeneration(styleId) }
                    )
                }
            }
        }
    }

    // 处理崩溃：写入文件
    private fun handleCrash(e: Throwable) {
        try {
            val sw = StringWriter()
            val pw = PrintWriter(sw)
            e.printStackTrace(pw)
            val stackTrace = sw.toString()

            val time = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
            val report = "\n[$time] CRASH REPORT:\n$stackTrace\n"

            // 写入 crash_log.txt
            FileOutputStream(crashFile, true).use {
                it.write(report.toByteArray())
            }

            // 同时写入 debug log
            writeLog("APP CRASHED! See crash_log.txt")
        } catch (ex: Exception) {
            // 既然都崩了，这里也没办法了
        } finally {
            // 必须杀掉进程，否则会黑屏卡死
            android.os.Process.killProcess(android.os.Process.myPid())
            System.exit(1)
        }
    }

    private fun runGeneration(styleId: Int) {
        val input = viewModel.uiState.value.originalBitmap ?: return
        if (!viewModel.isEngineReady) {
            viewModel.updateStatus("引擎未就绪")
            return
        }

        viewModel.setProcessing(true)
        viewModel.updateStatus("生成中 (OpenCL)...")

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                // 确保宽高一致，防止底层越界
                val w = 512
                val h = 512
                val output = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

                // 双重检查 Bitmap 状态
                if (input.isRecycled) throw RuntimeException("Input Bitmap is recycled!")
                if (input.width != w || input.height != h) throw RuntimeException("Input size mismatch: ${input.width}x${input.height}")

                val start = System.currentTimeMillis()
                val success = runStyleTransfer(input, output, styleId)
                val cost = System.currentTimeMillis() - start

                withContext(Dispatchers.Main) {
                    viewModel.setProcessing(false)
                    if (success) {
                        viewModel.setResult(output)
                        viewModel.updateStatus("完成! 耗时: ${cost}ms")
                    } else {
                        viewModel.updateStatus("生成失败 (看日志)")
                    }
                }
            } catch (e: Exception) {
                // 这里捕获的是逻辑错误，Thread.setDefaultUncaughtExceptionHandler 捕获的是闪退
                withContext(Dispatchers.Main) {
                    viewModel.setProcessing(false)
                    viewModel.updateStatus("Err: ${e.message}")
                    writeLog("Logic Error: ${e.message}")
                    // 如果是严重错误，手动触发记录
                    handleCrash(e)
                }
            }
        }
    }

    private suspend fun safeLoadLibrariesAndModels() {
        writeLog("App Start")
        try {
            System.loadLibrary("c++_shared")
            System.loadLibrary("MNN")
            System.loadLibrary("sd_engine")

            val modelFiles = listOf("Encoder.mnn", "Flow.mnn", "Decoder.mnn")
            for (fileName in modelFiles) {
                val outFile = File(cacheDir, fileName)
                // 每次都覆盖拷贝，防止文件损坏
                copyAssetResource(fileName, outFile)
            }

            val success = initEngine(cacheDir.absolutePath)
            withContext(Dispatchers.Main) {
                if (success) {
                    viewModel.isEngineReady = true
                    viewModel.updateStatus("引擎就绪 (Snapdragon 8 Elite)")
                } else {
                    viewModel.updateStatus("初始化失败")
                }
            }
        } catch (e: Throwable) {
            writeLog("Load Error: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun writeLog(msg: String) {
        try { FileOutputStream(logFile, true).use { it.write("[Java] $msg\n".toByteArray()) } } catch (e: Exception) {}
    }

    private fun copyAssetResource(assetName: String, outFile: File): Boolean {
        return try {
            assets.open(assetName).use { input -> FileOutputStream(outFile).use { output -> input.copyTo(output) } }
            true
        } catch (e: Exception) { false }
    }
}

// ==========================================
// UI 部分 (超级安全版)
// ==========================================

@Composable
fun StyleTransferScreen(
    viewModel: MainViewModel,
    onGenerate: (Int) -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()

    val photoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> if (uri != null) viewModel.onImageSelected(uri) }
    )

    var selectedStyleId by remember { mutableIntStateOf(0) }

    // 使用 LazyColumn 是解决滚动冲突的最佳方案
    // 如果这里还崩，说明是 Compose 版本或系统兼容性的大问题
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp), // 简单的 Padding，不依赖 systemBars
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // 顶部留白适配状态栏 (粗暴但有效)
        item { Spacer(modifier = Modifier.height(40.dp)) }

        item {
            Text("MNN Style Transfer", fontSize = 22.sp, fontWeight = FontWeight.Bold)
            Text("Adreno GPU (OpenCL)", fontSize = 12.sp, color = Color.Gray)
            Spacer(modifier = Modifier.height(12.dp))
        }

        item {
            Text(
                text = uiState.statusMessage,
                fontSize = 14.sp,
                color = if (uiState.isProcessing) Color.Red else Color.Black,
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFFF5F5F5), RoundedCornerShape(8.dp))
                    .padding(10.dp)
            )
            Spacer(modifier = Modifier.height(20.dp))
        }

        item {
            Text("Input (Original)", fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))
            BigImageCard(uiState.originalBitmap) {
                photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }
            Spacer(modifier = Modifier.height(20.dp))
        }

        item {
            Text("Output (Result)", fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))
            BigImageCard(uiState.resultBitmap) { }
            Spacer(modifier = Modifier.height(24.dp))
        }

        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Button(
                    onClick = { selectedStyleId = 0 },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (selectedStyleId == 0) MaterialTheme.colorScheme.primary else Color.LightGray
                    )
                ) { Text("Style A") }

                Button(
                    onClick = { selectedStyleId = 1 },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (selectedStyleId == 1) MaterialTheme.colorScheme.primary else Color.LightGray
                    )
                ) { Text("Style B") }
            }
            Spacer(modifier = Modifier.height(24.dp))
        }

        item {
            Button(
                onClick = {
                    if (uiState.originalBitmap == null) {
                        photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                    } else {
                        onGenerate(selectedStyleId)
                    }
                },
                enabled = !uiState.isProcessing,
                modifier = Modifier.fillMaxWidth().height(56.dp)
            ) {
                if (uiState.isProcessing) {
                    CircularProgressIndicator(color = Color.White, modifier = Modifier.size(24.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Generating...")
                } else {
                    Text(if (uiState.originalBitmap == null) "Select Image" else "Start Generate")
                }
            }
            Spacer(modifier = Modifier.height(100.dp)) // 底部超大留白，防止到底部闪退
        }
    }
}

@Composable
fun BigImageCard(bitmap: Bitmap?, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(RoundedCornerShape(12.dp))
            .background(Color.LightGray)
            .border(1.dp, Color.Gray, RoundedCornerShape(12.dp))
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        if (bitmap != null) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.FillBounds
            )
        } else {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("Tap to Select", color = Color.White, fontWeight = FontWeight.Bold)
                Text("512 x 512", color = Color.White.copy(alpha = 0.7f), fontSize = 12.sp)
            }
        }
    }
}