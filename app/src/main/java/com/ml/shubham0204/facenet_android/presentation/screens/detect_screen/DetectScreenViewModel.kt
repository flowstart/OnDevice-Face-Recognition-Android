package com.ml.shubham0204.facenet_android.presentation.screens.detect_screen

import android.os.Build
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.ml.shubham0204.facenet_android.domain.ImageVectorUseCase
import com.ml.shubham0204.facenet_android.domain.PersonUseCase
import com.ml.shubham0204.facenet_android.domain.offloading.CloudService
import com.ml.shubham0204.facenet_android.domain.offloading.OffloadingConfig
import com.ml.shubham0204.facenet_android.domain.offloading.OffloadingConfig.OffloadingMode
import com.ml.shubham0204.facenet_android.domain.offloading.OffloadingImageVectorUseCase
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.koin.android.annotation.KoinViewModel

/**
 * Extended metrics including network time for display.
 */
data class ExtendedMetrics(
    val timeFaceDetection: Long,
    val timeFaceEmbedding: Long,
    val timeVectorSearch: Long,
    val timeFaceSpoofDetection: Long,
    val timeNetworkTransfer: Long,
    val timeServerProcessing: Long,
    val totalTime: Long,
    val dataTransferredBytes: Long,
    val modeName: String,
    val isFallback: Boolean = false  // 是否回退到本地模式
)

@KoinViewModel
class DetectScreenViewModel(
    val personUseCase: PersonUseCase,
    val imageVectorUseCase: ImageVectorUseCase,
    val offloadingImageVectorUseCase: OffloadingImageVectorUseCase,
    private val cloudService: CloudService,
) : ViewModel() {
    
    // Original metrics state (for compatibility)
    val faceDetectionMetricsState = mutableStateOf<com.ml.shubham0204.facenet_android.data.RecognitionMetrics?>(null)
    
    // Extended metrics with network info
    private val _extendedMetrics = MutableStateFlow<ExtendedMetrics?>(null)
    val extendedMetrics: StateFlow<ExtendedMetrics?> = _extendedMetrics
    
    // Upload status
    private val _uploadStatus = MutableStateFlow<String>("")
    val uploadStatus: StateFlow<String> = _uploadStatus
    
    // Current mode display
    private val _currentModeName = MutableStateFlow(getModeDisplayName(OffloadingConfig.currentMode))
    val currentModeName: StateFlow<String> = _currentModeName
    
    // Network error state
    private val _networkError = MutableStateFlow<String?>(null)
    val networkError: StateFlow<String?> = _networkError
    
    // Fallback mode state
    private val _isFallbackMode = MutableStateFlow(false)
    val isFallbackMode: StateFlow<Boolean> = _isFallbackMode

    fun getNumPeople(): Long = personUseCase.getCount()
    
    /**
     * Get current offloading mode.
     */
    fun getCurrentMode(): OffloadingMode = OffloadingConfig.currentMode
    
    /**
     * Update extended metrics from offloading use case.
     */
    fun updateExtendedMetrics(metrics: OffloadingImageVectorUseCase.OffloadingMetrics) {
        _extendedMetrics.value = ExtendedMetrics(
            timeFaceDetection = metrics.timeFaceDetection,
            timeFaceEmbedding = metrics.timeFaceEmbedding,
            timeVectorSearch = metrics.timeVectorSearch,
            timeFaceSpoofDetection = metrics.timeFaceSpoofDetection,
            timeNetworkTransfer = metrics.timeNetworkTransfer,
            timeServerProcessing = metrics.timeServerProcessing,
            totalTime = metrics.totalTime,
            dataTransferredBytes = metrics.dataTransferredBytes,
            modeName = getModeDisplayName(metrics.offloadingMode),
            isFallback = metrics.isFallback
        )
        _currentModeName.value = getModeDisplayName(OffloadingConfig.currentMode)
        _isFallbackMode.value = metrics.isFallback
        
        // 更新网络错误提示
        if (metrics.isFallback && metrics.offloadingMode != OffloadingConfig.OffloadingMode.LOCAL_ONLY) {
            _networkError.value = "已回退到本地模式"
        } else {
            _networkError.value = null
        }
    }
    
    /**
     * Set network error message.
     */
    fun setNetworkError(error: String?) {
        _networkError.value = error
    }
    
    /**
     * Upload current performance data to server.
     */
    fun uploadPerformanceData() {
        val metrics = _extendedMetrics.value ?: return
        
        viewModelScope.launch {
            _uploadStatus.value = "上传中..."
            
            val data = CloudService.PerformanceData(
                sessionId = OffloadingConfig.sessionId,
                mode = OffloadingConfig.currentMode.name,
                modeName = getModeDisplayName(OffloadingConfig.currentMode),
                faceDetectionMs = metrics.timeFaceDetection,
                embeddingMs = metrics.timeFaceEmbedding,
                vectorSearchMs = metrics.timeVectorSearch,
                spoofDetectionMs = metrics.timeFaceSpoofDetection,
                networkMs = metrics.timeNetworkTransfer,
                serverMs = metrics.timeServerProcessing,
                totalMs = metrics.totalTime,
                dataTransferredBytes = metrics.dataTransferredBytes,
                deviceInfo = "${Build.MANUFACTURER} ${Build.MODEL}",
                networkType = "5G" // TODO: Detect actual network type
            )
            
            val success = cloudService.uploadPerformanceData(data)
            
            if (success) {
                OffloadingConfig.uploadedModes.add(OffloadingConfig.currentMode)
                _uploadStatus.value = "上传成功！(${OffloadingConfig.uploadedModes.size}/5)"
            } else {
                _uploadStatus.value = "上传失败，请重试"
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
