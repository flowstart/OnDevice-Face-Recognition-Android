package com.ml.shubham0204.facenet_android.presentation.screens.benchmark

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.ml.shubham0204.facenet_android.domain.offloading.OffloadingConfig
import com.ml.shubham0204.facenet_android.domain.offloading.OffloadingConfig.OffloadingMode
import kotlinx.coroutines.launch

/**
 * Benchmark screen for testing different computation partitioning modes.
 * 
 * This screen allows users to:
 * 1. Select different offloading modes
 * 2. Configure server settings
 * 3. Run performance tests
 * 4. View and export results
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BenchmarkScreen(
    viewModel: BenchmarkViewModel,
    onNavigateBack: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    val currentMode by viewModel.currentMode.collectAsState()
    val serverHost by viewModel.serverHost.collectAsState()
    val serverPort by viewModel.serverPort.collectAsState()
    val isServerReachable by viewModel.isServerReachable.collectAsState()
    val testResults by viewModel.testResults.collectAsState()
    val isRunning by viewModel.isRunning.collectAsState()
    val statusMessage by viewModel.statusMessage.collectAsState()
    val uploadedModes by viewModel.uploadedModes.collectAsState()
    val configSaved by viewModel.configSaved.collectAsState()
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("性能测试 Benchmark") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "返回")
                    }
                },
                actions = {
                    IconButton(
                        onClick = { viewModel.openReportList(context) }
                    ) {
                        Icon(Icons.Default.Share, "查看报告")
                    }
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Server Configuration
            item {
                ServerConfigCard(
                    serverHost = serverHost,
                    serverPort = serverPort,
                    isReachable = isServerReachable,
                    configSaved = configSaved,
                    onHostChange = { viewModel.updateServerHost(it) },
                    onPortChange = { viewModel.updateServerPort(it) },
                    onTestConnection = { viewModel.testServerConnection() },
                    onSaveConfig = { viewModel.saveConfig() }
                )
            }
            
            // Mode Selection
            item {
                ModeSelectionCard(
                    currentMode = currentMode,
                    uploadedModes = uploadedModes,
                    onModeSelected = { viewModel.setMode(it) }
                )
            }
            
            // Save and Report Buttons
            item {
                ReportCard(
                    statusMessage = statusMessage,
                    canGenerateReport = viewModel.canGenerateReport(),
                    uploadedCount = uploadedModes.size,
                    totalModes = OffloadingMode.values().size,
                    onGenerateReport = { viewModel.generateReport(context) },
                    onResetSession = { viewModel.resetSession() }
                )
            }
            
            // Results
            if (testResults.isNotEmpty()) {
                item {
                    Text(
                        "测试结果",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                }
                
                items(testResults) { result ->
                    ResultCard(result)
                }
            }
        }
    }
}

@Composable
fun ServerConfigCard(
    serverHost: String,
    serverPort: Int,
    isReachable: Boolean?,
    configSaved: Boolean,
    onHostChange: (String) -> Unit,
    onPortChange: (Int) -> Unit,
    onTestConnection: () -> Unit,
    onSaveConfig: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                "服务器配置",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            OutlinedTextField(
                value = serverHost,
                onValueChange = onHostChange,
                label = { Text("服务器 IP") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )
            
            OutlinedTextField(
                value = serverPort.toString(),
                onValueChange = { it.toIntOrNull()?.let(onPortChange) },
                label = { Text("端口") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Connection status
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(12.dp)
                            .background(
                                when (isReachable) {
                                    true -> Color.Green
                                    false -> Color.Red
                                    null -> Color.Gray
                                },
                                RoundedCornerShape(6.dp)
                            )
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        when (isReachable) {
                            true -> "已连接"
                            false -> "无法连接"
                            null -> "未测试"
                        },
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                
                Button(onClick = onTestConnection) {
                    Text("测试连接")
                }
            }
            
            // Save button
            Button(
                onClick = onSaveConfig,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (configSaved) 
                        MaterialTheme.colorScheme.secondary 
                    else 
                        MaterialTheme.colorScheme.primary
                )
            ) {
                Text(if (configSaved) "已保存" else "保存配置")
            }
        }
    }
}

@Composable
fun ModeSelectionCard(
    currentMode: OffloadingMode,
    uploadedModes: Set<OffloadingMode>,
    onModeSelected: (OffloadingMode) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "选择划分模式",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                // Upload status indicators
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    OffloadingMode.values().forEach { mode ->
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .background(
                                    if (uploadedModes.contains(mode)) Color(0xFF4CAF50)
                                    else Color.Gray.copy(alpha = 0.3f),
                                    RoundedCornerShape(5.dp)
                                )
                        )
                    }
                }
            }
            
            Text(
                "上传状态: ${uploadedModes.size}/5",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
            
            OffloadingMode.values().forEach { mode ->
                ModeOption(
                    mode = mode,
                    isSelected = mode == currentMode,
                    isUploaded = uploadedModes.contains(mode),
                    onClick = { onModeSelected(mode) }
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModeOption(
    mode: OffloadingMode,
    isSelected: Boolean,
    isUploaded: Boolean = false,
    onClick: () -> Unit
) {
    val (title, description, localParts, cloudParts) = getModeInfo(mode)
    
    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected) 
                MaterialTheme.colorScheme.primaryContainer 
            else 
                MaterialTheme.colorScheme.surface
        ),
        shape = RoundedCornerShape(8.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        title,
                        fontWeight = FontWeight.Medium,
                        color = if (isSelected) 
                            MaterialTheme.colorScheme.onPrimaryContainer 
                        else 
                            MaterialTheme.colorScheme.onSurface
                    )
                    if (isUploaded) {
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            "✓",
                            color = Color(0xFF4CAF50),
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
                RadioButton(
                    selected = isSelected,
                    onClick = onClick
                )
            }
            Text(
                description,
                style = MaterialTheme.typography.bodySmall,
                color = if (isSelected) 
                    MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f) 
                else 
                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (localParts.isNotEmpty()) {
                    Text(
                        "📱 $localParts",
                        style = MaterialTheme.typography.labelSmall
                    )
                }
                if (cloudParts.isNotEmpty()) {
                    Text(
                        "☁️ $cloudParts",
                        style = MaterialTheme.typography.labelSmall
                    )
                }
            }
        }
    }
}

fun getModeInfo(mode: OffloadingMode): List<String> {
    return when (mode) {
        OffloadingMode.LOCAL_ONLY -> listOf(
            "Mode 0: 全本地",
            "所有计算在手机上执行，无网络通信",
            "检测+嵌入+搜索",
            ""
        )
        OffloadingMode.EMBEDDING_OFFLOAD -> listOf(
            "Mode 1: 嵌入卸载",
            "FaceNet 嵌入生成在云端执行",
            "检测+搜索",
            "嵌入生成"
        )
        OffloadingMode.SEARCH_OFFLOAD -> listOf(
            "Mode 2: 搜索卸载",
            "向量搜索在云端执行",
            "检测+嵌入",
            "向量搜索"
        )
        OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD -> listOf(
            "Mode 3: 嵌入+搜索卸载",
            "嵌入生成和向量搜索都在云端",
            "检测",
            "嵌入+搜索"
        )
        OffloadingMode.FULL_OFFLOAD -> listOf(
            "Mode 4: 全卸载",
            "几乎所有计算都在云端执行",
            "仅活体检测",
            "检测+嵌入+搜索"
        )
    }
}

@Composable
fun ReportCard(
    statusMessage: String,
    canGenerateReport: Boolean,
    uploadedCount: Int,
    totalModes: Int,
    onGenerateReport: () -> Unit,
    onResetSession: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                "性能报告",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            if (statusMessage.isNotEmpty()) {
                Text(
                    statusMessage,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            
            Text(
                "请在识别页面测试每个模式，并点击上传按钮。\n已上传: $uploadedCount / $totalModes 个模式",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Button(
                    onClick = onGenerateReport,
                    enabled = canGenerateReport,
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        if (canGenerateReport) "生成报告" else "请先上传所有模式"
                    )
                }
                
                OutlinedButton(
                    onClick = onResetSession,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("重置")
                }
            }
        }
    }
}

@Composable
fun ResultCard(result: BenchmarkResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                result.modeName,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            
            Divider()
            
            ResultRow("总延迟", "${result.totalTimeMs} ms")
            ResultRow("人脸检测", "${result.faceDetectionMs} ms")
            ResultRow("嵌入生成", "${result.embeddingMs} ms")
            ResultRow("向量搜索", "${result.searchMs} ms")
            ResultRow("网络传输", "${result.networkMs} ms")
            ResultRow("服务器处理", "${result.serverMs} ms")
            ResultRow("数据传输", "${result.dataTransferredKB} KB")
        }
    }
}

@Composable
fun ResultRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium
        )
    }
}

/**
 * Data class for benchmark results.
 */
data class BenchmarkResult(
    val modeName: String,
    val totalTimeMs: Long,
    val faceDetectionMs: Long,
    val embeddingMs: Long,
    val searchMs: Long,
    val networkMs: Long,
    val serverMs: Long,
    val dataTransferredKB: Float
)

