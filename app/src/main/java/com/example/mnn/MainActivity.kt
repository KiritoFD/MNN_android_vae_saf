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

    // Native 方法：注意增加了 steps 参数
    external fun initEngine(cacheDir: String): Boolean
    external fun runStyleTransfer(src: Bitmap, dst: Bitmap, styleId: Int, steps: Int): Boolean

    companion object {
        init {
            try {
                System.loadLibrary("MNN")
                System.loadLibrary("MNN_Express")
                System.loadLibrary("sd_engine") // 你的 C++ 库名
                Log.i("SAFlow", "Libraries Loaded")
            } catch (e: Exception) {
                Log.e("SAFlow", "Lib Load Failed: ${e.message}")
            }
        }
    }

    private val logFile by lazy { File(cacheDir, "java_debug.txt") }
    private val crashFile by lazy { File(cacheDir, "crash_log.txt") }
    private val viewModel: MainViewModel by viewModels()

    // 模型文件选择器
    private val modelPickerLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            viewModel.replaceModelFile(uri, cacheDir) { success ->
                if (success) {
                    // 文件替换成功，重新初始化 Native 引擎
                    lifecycleScope.launch(Dispatchers.IO) {
                        reloadEngine()
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 崩溃捕获
        Thread.setDefaultUncaughtExceptionHandler { _, throwable ->
            handleCrash(throwable)
        }

        // 初始化资源
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
                        onUploadModel = {
                            // 启动文件选择器，MNN文件通常识别为 octet-stream 或任意类型
                            modelPickerLauncher.launch("*/*")
                        }
                    )
                }
            }
        }
    }

    // 重载引擎 (用于模型上传后)
    private suspend fun reloadEngine() {
        viewModel.isEngineReady = false
        val success = initEngine(cacheDir.absolutePath)

        withContext(Dispatchers.Main) {
            viewModel.setProcessing(false) // 解锁 UI
            if (success) {
                viewModel.isEngineReady = true
                viewModel.updateStatus("新模型已加载 (CPU Mode)")
            } else {
                viewModel.updateStatus("模型加载失败 (JNI Error)")
            }
        }
    }

    private fun handleCrash(e: Throwable) {
        try {
            val sw = StringWriter()
            e.printStackTrace(PrintWriter(sw))
            val stackTrace = sw.toString()
            val time = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
            FileOutputStream(crashFile, true).use { it.write("\n[$time] CRASH:\n$stackTrace\n".toByteArray()) }
        } catch (_: Exception) {}
        finally {
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

        val steps = viewModel.uiState.value.steps
        viewModel.setProcessing(true)
        viewModel.updateStatus("生成中 (CPU: $steps 步)...")

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val output = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
                if (input.isRecycled) throw RuntimeException("Bitmap recycled")

                val start = System.currentTimeMillis()
                // 调用 Native，传入 steps
                val success = runStyleTransfer(input, output, styleId, steps)
                val cost = System.currentTimeMillis() - start

                withContext(Dispatchers.Main) {
                    viewModel.setProcessing(false)
                    if (success) {
                        viewModel.setResult(output)
                        viewModel.updateStatus("完成! 耗时: ${cost}ms")
                    } else {
                        viewModel.updateStatus("生成失败")
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    viewModel.setProcessing(false)
                    viewModel.updateStatus("错误: ${e.message}")
                }
            }
        }
    }

    private suspend fun prepareModelsAndEngine() {
        writeLog("Starting Init...")
        try {
            val modelFiles = listOf("Encoder.mnn", "Flow.mnn", "Decoder.mnn")
            for (fileName in modelFiles) {
                val outFile = File(cacheDir, fileName)
                // 如果文件不存在则从 assets 拷贝；若已存在(可能是用户上传的)则保留
                if (!outFile.exists()) {
                    assets.open(fileName).use { input ->
                        FileOutputStream(outFile).use { output -> input.copyTo(output) }
                    }
                }
            }

            val success = initEngine(cacheDir.absolutePath)
            withContext(Dispatchers.Main) {
                if (success) {
                    viewModel.isEngineReady = true
                    viewModel.updateStatus("引擎就绪 (CPU FP16)")
                } else {
                    viewModel.updateStatus("初始化失败")
                }
            }
        } catch (e: Throwable) {
            writeLog("Init Error: ${e.message}")
        }
    }

    private fun writeLog(msg: String) {
        try { FileOutputStream(logFile, true).use { it.write("$msg\n".toByteArray()) } } catch (_: Exception) {}
    }
}

// ================= UI 组件 =================

@Composable
fun StyleTransferScreen(
    viewModel: MainViewModel,
    onGenerate: (Int) -> Unit,
    onUploadModel: () -> Unit
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
            Text("CPU Optimized (FP16)", fontSize = 12.sp, color = Color.Gray)
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
            BigImageCard(uiState.originalBitmap) {
                photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }
            Spacer(modifier = Modifier.height(20.dp))
            BigImageCard(uiState.resultBitmap) { }
            Spacer(modifier = Modifier.height(24.dp))
        }

        // 风格选择按钮
        item {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
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

        // 步数滑块 (Steps Slider)
        item {
            Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp)) {
                Text("Inference Steps: ${uiState.steps}", fontSize = 14.sp, fontWeight = FontWeight.Bold)
                Slider(
                    value = uiState.steps.toFloat(),
                    onValueChange = { viewModel.setSteps(it.toInt()) },
                    valueRange = 1f..20f,
                    steps = 19,
                    enabled = !uiState.isProcessing
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
        }

        // 上传模型按钮
        item {
            OutlinedButton(
                onClick = onUploadModel,
                enabled = !uiState.isProcessing,
                modifier = Modifier.fillMaxWidth().height(48.dp)
            ) {
                Text("Upload Custom Flow.mnn")
            }
            Spacer(modifier = Modifier.height(16.dp))
        }

        // 生成按钮
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
                    Text(if (uiState.originalBitmap == null) "Select Image" else "Generate")
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
            .size(300.dp) // 固定大小稍微小一点适配屏幕
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
                Text("Select / Result", color = Color.White, fontWeight = FontWeight.Bold)
            }
        }
    }
}