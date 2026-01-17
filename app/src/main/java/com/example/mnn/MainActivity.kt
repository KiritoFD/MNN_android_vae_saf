package com.example.mnn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.foundation.clickable // 确保 clickable 被引用
import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
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

class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    StyleTransferScreen(viewModel)
                }
            }
        }
    }
}

@Composable
fun StyleTransferScreen(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsState()

    // 图片选择器
    val photoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> if (uri != null) viewModel.onImageSelected(uri) }
    )

    // 风格选择状态 (0 或 1)
    var selectedStyleId by remember { mutableIntStateOf(0) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()), // 允许滚动防止小屏遮挡
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // 1. 标题与状态
        Text("MNN Style Transfer", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Text("Snapdragon 8 Elite Edition", fontSize = 12.sp, color = Color.Gray)
        Spacer(modifier = Modifier.height(8.dp))

        // 状态条
        Card(
            colors = CardDefaults.cardColors(containerColor = if (uiState.isProcessing) Color(0xFFFFF3E0) else Color(0xFFE8F5E9)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = uiState.statusMessage,
                modifier = Modifier.padding(12.dp),
                fontSize = 14.sp
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 2. 图片展示区 (原图 vs 结果)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            // 原图卡片
            ImageCard("原图 (Input)", uiState.originalBitmap) {
                // 点击原图位置也可以触发选择
                photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }

            // 结果图卡片
            ImageCard("结果 (Output)", uiState.resultBitmap) {
                // 点击结果图可以是保存或者是大图预览（暂空）
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // 3. 风格选择器
        Text("选择目标风格 (Parameter S)", fontWeight = FontWeight.SemiBold)
        Spacer(modifier = Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            StyleOption(id = 0, name = "Style A (梵高风)", selectedId = selectedStyleId) { selectedStyleId = 0 }
            StyleOption(id = 1, name = "Style B (油画风)", selectedId = selectedStyleId) { selectedStyleId = 1 }
        }

        Spacer(modifier = Modifier.height(32.dp))

        // 4. 操作按钮
        if (uiState.originalBitmap == null) {
            Button(
                onClick = { photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                modifier = Modifier.fillMaxWidth().height(50.dp)
            ) {
                Text("上传图片")
            }
        } else {
            Button(
                onClick = { viewModel.generate(selectedStyleId) },
                enabled = !uiState.isProcessing, // 处理中禁用
                modifier = Modifier.fillMaxWidth().height(50.dp)
            ) {
                if (uiState.isProcessing) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), color = Color.White)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("生成中...")
                } else {
                    Text("开始转换 (Start)")
                }
            }
        }
    }
}

// 组件：单个图片展示卡片
@Composable
fun ImageCard(title: String, bitmap: Bitmap?, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            modifier = Modifier
                .size(160.dp) // 160dp 正方形
                .clip(RoundedCornerShape(12.dp))
                .background(Color.LightGray)
                .border(1.dp, Color.Gray, RoundedCornerShape(12.dp))
                .noRippleClickable(onClick), // 自定义点击事件
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
            } else {
                Text("waiting...", color = Color.DarkGray, fontSize = 12.sp)
            }
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(title, fontSize = 14.sp)
    }
}

// 组件：风格单选按钮
@Composable
fun StyleOption(id: Int, name: String, selectedId: Int, onClick: () -> Unit) {
    FilterChip(
        selected = (id == selectedId),
        onClick = onClick,
        label = { Text(name) },
        leadingIcon = {
            if (id == selectedId) {
                Icon(
                    androidx.compose.material.icons.Icons.Filled.Check,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
            }
        }
    )
}

// 辅助：去除点击波纹的修饰符（为了代码简洁，这里做个简版）
// 实际可以直接用 clickable
fun Modifier.noRippleClickable(onClick: () -> Unit): Modifier = this.then(
    Modifier.clickable { onClick() }
)