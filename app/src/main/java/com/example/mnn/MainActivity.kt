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

    // Native 方法声明
    external fun initEngine(cacheDir: String): Boolean
    external fun runStyleTransfer(src: Bitmap, dst: Bitmap, styleId: Int): Boolean

    companion object {
        init {
            try {
                // 按照官方库链接逻辑加载
                System.loadLibrary("MNN")
                System.loadLibrary("MNN_Express")
                System.loadLibrary("sd_engine")
                Log.i("SAFlow_JNI", "Native Libraries Loaded Successfully")
            } catch (e: Exception) {
                Log.e("SAFlow_JNI", "Native Lib Load Failed: ${e.message}")
            }
        }
    }

    private val logFile by lazy { File(cacheDir, "java_debug.txt") }
    private val crashFile by lazy { File(cacheDir, "crash_log.txt") }
    private val viewModel: MainViewModel by viewModels()

    // 新增：文件选择器，用于选择新的 .mnn 文件
    private val modelPickerLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            viewModel.replaceModelFile(uri, cacheDir) { success ->
                if (success) {
                    // 替换成功后，在 IO 线程重新加载引擎
                    lifecycleScope.launch(Dispatchers.IO) {
                        reloadEngine()
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 全局崩溃捕获
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            handleCrash(throwable)
        }

        // 异步初始化资源
        lifecycleScope.launch(Dispatchers.IO) {
            prepareModelsAndEngine()
        }

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    StyleTransferScreen(
                        viewModel = viewModel,
                        onGenerate = { styleId -> runGeneration(styleId) },
                        // 新增：点击上传按钮的回调
                        onUploadModel = {
                            // 启动文件选择器，过滤任意类型或指定 application/octet-stream
                            modelPickerLauncher.launch("*/*")
                        }
                    )
                }
            }
        }
    }

    // 新增：重新加载引擎的方法
    private suspend fun reloadEngine() {
        viewModel.isEngineReady = false
        // C++ 层的 initEngine 会 delete 旧指针并 new 新对象，读取最新的 Flow.mnn
        val success = initEngine(cacheDir.absolutePath)

        withContext(Dispatchers.Main) {
            viewModel.setProcessing(false) // 解锁 UI
            if (success) {
                viewModel.isEngineReady = true
                viewModel.updateStatus("新模型加载成功!")
            } else {
                viewModel.updateStatus("新模型加载失败 (JNI Error)")
            }
        }
    }

    private fun handleCrash(e: Throwable) {
        try {
            val sw = StringWriter()
            e.printStackTrace(PrintWriter(sw))
            val stackTrace = sw.toString()
            val time = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
            val report = "\n[$time] CRASH REPORT:\n$stackTrace\n"

            FileOutputStream(crashFile, true).use { it.write(report.toByteArray()) }
            writeLog("APP CRASHED! See crash_log.txt")
        } catch (ex: Exception) {
            Log.e("SAFlow_Crash", "Failed to write crash log")
        } finally {
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
        viewModel.updateStatus("生成中 (OpenGL)...")

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val w = 512
                val h = 512
                // 确保使用 ARGB_8888 适配 MNN ImageProcess
                val output = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

                if (input.isRecycled) throw RuntimeException("Input Bitmap is recycled!")

                val start = System.currentTimeMillis()
                val success = runStyleTransfer(input, output, styleId)
                val cost = System.currentTimeMillis() - start

                withContext(Dispatchers.Main) {
                    viewModel.setProcessing(false)
                    if (success) {
                        viewModel.setResult(output)
                        viewModel.updateStatus("完成! 耗时: ${cost}ms")
                    } else {
                        viewModel.updateStatus("生成失败 (后端回退)")
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    viewModel.setProcessing(false)
                    viewModel.updateStatus("Err: ${e.message}")
                    writeLog("Logic Error: ${e.message}")
                }
            }
        }
    }

    private suspend fun prepareModelsAndEngine() {
        writeLog("Preparing Engine...")
        try {
            // 拷贝 Asset 模型文件
            val modelFiles = listOf("Encoder.mnn", "Flow.mnn", "Decoder.mnn")
            for (fileName in modelFiles) {
                val outFile = File(cacheDir, fileName)
                // 只有文件不存在时才拷贝，避免覆盖用户上传的自定义 Flow.mnn
                // 如果你想每次启动都重置为默认，去掉 !outFile.exists() 判断即可
                if (!outFile.exists()) {
                    copyAssetResource(fileName, outFile)
                }
            }

            // 初始化 Native 引擎
            val success = initEngine(cacheDir.absolutePath)
            withContext(Dispatchers.Main) {
                if (success) {
                    viewModel.isEngineReady = true
                    viewModel.updateStatus("引擎就绪 (Snapdragon 8 Elite OpenGL)")
                } else {
                    viewModel.updateStatus("OpenGL 初始化失败")
                }
            }
        } catch (e: Throwable) {
            writeLog("Initialization Error: ${e.message}")
        }
    }

    private fun writeLog(msg: String) {
        try { FileOutputStream(logFile, true).use { it.write("[Java] $msg\n".toByteArray()) } } catch (e: Exception) {}
    }

    private fun copyAssetResource(assetName: String, outFile: File): Boolean {
        return try {
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output -> input.copyTo(output) }
            }
            true
        } catch (e: Exception) { false }
    }
}

// ==========================================
// UI 部分
// ==========================================

@Composable
fun StyleTransferScreen(
    viewModel: MainViewModel,
    onGenerate: (Int) -> Unit,
    onUploadModel: () -> Unit // 新增参数
) {
    val uiState by viewModel.uiState.collectAsState()
    val photoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> if (uri != null) viewModel.onImageSelected(uri) }
    )
    var selectedStyleId by remember { mutableIntStateOf(0) }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        item { Spacer(modifier = Modifier.height(40.dp)) }

        item {
            Text("MNN Style Transfer", fontSize = 22.sp, fontWeight = FontWeight.Bold)
            Text("Adreno GPU (OpenGL ES 3.0)", fontSize = 12.sp, color = Color.Gray)
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
            Text("Input Image", fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))
            BigImageCard(uiState.originalBitmap) {
                photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }
            Spacer(modifier = Modifier.height(20.dp))
        }

        item {
            Text("Result", fontWeight = FontWeight.Bold)
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
                ) { Text("Artistic") }

                Button(
                    onClick = { selectedStyleId = 1 },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (selectedStyleId == 1) MaterialTheme.colorScheme.primary else Color.LightGray
                    )
                ) { Text("Photo") }
            }
            Spacer(modifier = Modifier.height(16.dp))
        }

        // === 新增：上传自定义模型按钮 ===
        item {
            OutlinedButton(
                onClick = onUploadModel,
                enabled = !uiState.isProcessing,
                modifier = Modifier.fillMaxWidth().height(50.dp)
            ) {
                Text("Upload Custom Flow.mnn")
            }
            Spacer(modifier = Modifier.height(16.dp))
        }
        // ============================

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
                    Text("Processing...")
                } else {
                    Text(if (uiState.originalBitmap == null) "Select Image" else "Generate Style")
                }
            }
            Spacer(modifier = Modifier.height(100.dp))
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
                Text("Select Photo", color = Color.White, fontWeight = FontWeight.Bold)
                Text("Target: 512x512", color = Color.White.copy(alpha = 0.7f), fontSize = 12.sp)
            }
        }
    }
}