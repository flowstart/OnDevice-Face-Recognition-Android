package com.ml.shubham0204.facenet_android.presentation.screens.benchmark

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.ml.shubham0204.facenet_android.domain.offloading.CloudService
import com.ml.shubham0204.facenet_android.domain.offloading.OffloadingConfig
import com.ml.shubham0204.facenet_android.domain.offloading.OffloadingConfig.OffloadingMode
import com.ml.shubham0204.facenet_android.domain.offloading.OffloadingImageVectorUseCase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.koin.android.annotation.KoinViewModel
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * ViewModel for the Benchmark screen.
 * Handles mode selection, server configuration, and test execution.
 */
@KoinViewModel
class BenchmarkViewModel(
    private val cloudService: CloudService,
    private val offloadingUseCase: OffloadingImageVectorUseCase
) : ViewModel() {
    
    private val _currentMode = MutableStateFlow(OffloadingConfig.currentMode)
    val currentMode: StateFlow<OffloadingMode> = _currentMode
    
    private val _serverHost = MutableStateFlow(OffloadingConfig.serverHost)
    val serverHost: StateFlow<String> = _serverHost
    
    private val _serverPort = MutableStateFlow(OffloadingConfig.serverPort)
    val serverPort: StateFlow<Int> = _serverPort
    
    private val _isServerReachable = MutableStateFlow<Boolean?>(null)
    val isServerReachable: StateFlow<Boolean?> = _isServerReachable
    
    private val _testResults = MutableStateFlow<List<BenchmarkResult>>(emptyList())
    val testResults: StateFlow<List<BenchmarkResult>> = _testResults
    
    private val _isRunning = MutableStateFlow(false)
    val isRunning: StateFlow<Boolean> = _isRunning
    
    private val _statusMessage = MutableStateFlow("")
    val statusMessage: StateFlow<String> = _statusMessage
    
    // Store all results for export
    private val allResults = mutableListOf<BenchmarkResult>()
    
    fun setMode(mode: OffloadingMode) {
        _currentMode.value = mode
        OffloadingConfig.currentMode = mode
    }
    
    fun updateServerHost(host: String) {
        _serverHost.value = host
        OffloadingConfig.serverHost = host
        _isServerReachable.value = null
    }
    
    fun updateServerPort(port: Int) {
        _serverPort.value = port
        OffloadingConfig.serverPort = port
        _isServerReachable.value = null
    }
    
    fun testServerConnection() {
        viewModelScope.launch {
            _statusMessage.value = "正在测试连接..."
            _isServerReachable.value = cloudService.isServerReachable()
            _statusMessage.value = if (_isServerReachable.value == true) {
                "服务器连接成功！"
            } else {
                "无法连接到服务器，请检查IP和端口"
            }
        }
    }
    
    /**
     * Run benchmark for the current mode.
     */
    fun runBenchmark() {
        viewModelScope.launch {
            _isRunning.value = true
            _statusMessage.value = "正在测试 ${getModeDisplayName(_currentMode.value)}..."
            
            try {
                val result = runSingleBenchmark(_currentMode.value)
                allResults.add(result)
                _testResults.value = listOf(result) + _testResults.value
                _statusMessage.value = "测试完成！"
            } catch (e: Exception) {
                _statusMessage.value = "测试失败: ${e.message}"
            }
            
            _isRunning.value = false
        }
    }
    
    /**
     * Run benchmarks for all modes.
     */
    fun runAllBenchmarks() {
        viewModelScope.launch {
            _isRunning.value = true
            val results = mutableListOf<BenchmarkResult>()
            
            for (mode in OffloadingMode.values()) {
                _statusMessage.value = "正在测试 ${getModeDisplayName(mode)}..."
                
                // Skip cloud modes if server not reachable
                if (mode != OffloadingMode.LOCAL_ONLY && _isServerReachable.value != true) {
                    // Test connection first
                    _isServerReachable.value = cloudService.isServerReachable()
                    if (_isServerReachable.value != true) {
                        _statusMessage.value = "服务器未连接，跳过云端模式"
                        continue
                    }
                }
                
                try {
                    OffloadingConfig.currentMode = mode
                    val result = runSingleBenchmark(mode)
                    results.add(result)
                    allResults.add(result)
                } catch (e: Exception) {
                    _statusMessage.value = "模式 ${mode.name} 测试失败: ${e.message}"
                }
            }
            
            // Restore original mode
            OffloadingConfig.currentMode = _currentMode.value
            
            _testResults.value = results
            _statusMessage.value = "所有测试完成！共 ${results.size} 个模式"
            _isRunning.value = false
        }
    }
    
    /**
     * Run a single benchmark test.
     */
    private suspend fun runSingleBenchmark(mode: OffloadingMode): BenchmarkResult {
        return withContext(Dispatchers.Default) {
            // Create a test image (160x160 black image as placeholder)
            // In real use, this would be a camera frame
            val testBitmap = createTestBitmap()
            
            // Run multiple iterations and average
            val iterations = 10
            var totalTime = 0L
            var faceDetection = 0L
            var embedding = 0L
            var search = 0L
            var network = 0L
            var server = 0L
            var dataTransferred = 0L
            
            repeat(iterations) {
                val (metrics, _) = offloadingUseCase.getNearestPersonNameWithOffloading(
                    testBitmap,
                    flatSearch = false
                )
                
                metrics?.let {
                    totalTime += it.totalTime
                    faceDetection += it.timeFaceDetection
                    embedding += it.timeFaceEmbedding
                    search += it.timeVectorSearch
                    network += it.timeNetworkTransfer
                    server += it.timeServerProcessing
                    dataTransferred += it.dataTransferredBytes
                }
            }
            
            BenchmarkResult(
                modeName = getModeDisplayName(mode),
                totalTimeMs = totalTime / iterations,
                faceDetectionMs = faceDetection / iterations,
                embeddingMs = embedding / iterations,
                searchMs = search / iterations,
                networkMs = network / iterations,
                serverMs = server / iterations,
                dataTransferredKB = (dataTransferred / iterations) / 1024f
            )
        }
    }
    
    /**
     * Create a test bitmap for benchmarking.
     */
    private fun createTestBitmap(): Bitmap {
        return Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
    }
    
    /**
     * Export results to CSV file and share.
     */
    fun exportResults(context: Context) {
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                try {
                    val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
                        .format(Date())
                    val file = File(context.getExternalFilesDir(null), "benchmark_$timestamp.csv")
                    
                    file.bufferedWriter().use { writer ->
                        // Header
                        writer.write("模式,总延迟(ms),人脸检测(ms),嵌入生成(ms),向量搜索(ms),")
                        writer.write("网络传输(ms),服务器处理(ms),数据传输(KB)\n")
                        
                        // Data
                        for (result in allResults) {
                            writer.write("${result.modeName},")
                            writer.write("${result.totalTimeMs},")
                            writer.write("${result.faceDetectionMs},")
                            writer.write("${result.embeddingMs},")
                            writer.write("${result.searchMs},")
                            writer.write("${result.networkMs},")
                            writer.write("${result.serverMs},")
                            writer.write("${result.dataTransferredKB}\n")
                        }
                    }
                    
                    // Share file
                    withContext(Dispatchers.Main) {
                        val uri = FileProvider.getUriForFile(
                            context,
                            "${context.packageName}.provider",
                            file
                        )
                        
                        val intent = Intent(Intent.ACTION_SEND).apply {
                            type = "text/csv"
                            putExtra(Intent.EXTRA_STREAM, uri)
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        
                        context.startActivity(Intent.createChooser(intent, "导出测试结果"))
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(context, "导出失败: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }
    
    private fun getModeDisplayName(mode: OffloadingMode): String {
        return when (mode) {
            OffloadingMode.LOCAL_ONLY -> "Mode 0: 全本地"
            OffloadingMode.EMBEDDING_OFFLOAD -> "Mode 1: 嵌入卸载"
            OffloadingMode.SEARCH_OFFLOAD -> "Mode 2: 搜索卸载"
            OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD -> "Mode 3: 嵌入+搜索"
            OffloadingMode.FULL_OFFLOAD -> "Mode 4: 全卸载"
        }
    }
}

