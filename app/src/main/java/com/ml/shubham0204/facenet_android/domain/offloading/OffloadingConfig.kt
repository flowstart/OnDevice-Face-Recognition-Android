package com.ml.shubham0204.facenet_android.domain.offloading

/**
 * Configuration for computation offloading.
 * 
 * This class defines the different offloading modes supported by the application.
 * Each mode represents a different partitioning strategy between local (mobile device)
 * and remote (cloud server) computation.
 */
object OffloadingConfig {
    
    /**
     * Offloading mode enumeration.
     * 
     * Defines how computation is partitioned between mobile device and cloud server:
     * 
     * LOCAL_ONLY (Mode 0):
     *   - All computation on mobile device
     *   - No network communication
     *   - Baseline for comparison
     * 
     * EMBEDDING_OFFLOAD (Mode 1):
     *   - Face detection: LOCAL
     *   - Face embedding (FaceNet): CLOUD
     *   - Vector search: LOCAL
     *   - Data transferred: Cropped face image (~50KB)
     * 
     * SEARCH_OFFLOAD (Mode 2):
     *   - Face detection: LOCAL
     *   - Face embedding: LOCAL
     *   - Vector search: CLOUD
     *   - Data transferred: 512-dim embedding vector (~2KB)
     * 
     * EMBEDDING_AND_SEARCH_OFFLOAD (Mode 3):
     *   - Face detection: LOCAL
     *   - Face embedding: CLOUD
     *   - Vector search: CLOUD
     *   - Data transferred: Cropped face image (~50KB)
     * 
     * FULL_OFFLOAD (Mode 4):
     *   - All computation on cloud server
     *   - Data transferred: Full camera frame (~200KB-500KB)
     */
    enum class OffloadingMode {
        LOCAL_ONLY,              // Mode 0: All local
        EMBEDDING_OFFLOAD,       // Mode 1: Offload FaceNet embedding
        SEARCH_OFFLOAD,          // Mode 2: Offload vector search
        EMBEDDING_AND_SEARCH_OFFLOAD, // Mode 3: Offload embedding + search
        FULL_OFFLOAD             // Mode 4: Offload everything
    }
    
    /**
     * Current offloading mode.
     * Can be changed at runtime for testing different partitioning strategies.
     */
    var currentMode: OffloadingMode = OffloadingMode.LOCAL_ONLY
    
    /**
     * Server configuration.
     */
    var serverHost: String = "124.220.234.68"  // Default server IP
    var serverPort: Int = 8000
    
    /**
     * Session ID for grouping performance data uploads.
     */
    var sessionId: String = java.util.UUID.randomUUID().toString()
    
    /**
     * Track which modes have uploaded data in current session.
     */
    val uploadedModes: MutableSet<OffloadingMode> = mutableSetOf()
    
    /**
     * Check if all modes have been uploaded.
     */
    fun allModesUploaded(): Boolean = uploadedModes.size == OffloadingMode.values().size
    
    /**
     * Reset session for new test.
     */
    fun resetSession() {
        sessionId = java.util.UUID.randomUUID().toString()
        uploadedModes.clear()
    }
    
    val serverBaseUrl: String
        get() = "http://$serverHost:$serverPort"
    
    /**
     * Network timeout configuration (milliseconds).
     */
    var connectTimeout: Long = 5000
    var readTimeout: Long = 30000
    var writeTimeout: Long = 30000
    
    /**
     * Image compression quality for network transfer (0-100).
     * Lower quality = smaller size = faster transfer = lower accuracy
     */
    var imageQuality: Int = 85
    
    /**
     * Get human-readable description of the current mode.
     */
    fun getModeDescription(mode: OffloadingMode): String {
        return when (mode) {
            OffloadingMode.LOCAL_ONLY -> "全本地执行 (无网络通信)"
            OffloadingMode.EMBEDDING_OFFLOAD -> "嵌入生成卸载 (传输裁剪的人脸图像)"
            OffloadingMode.SEARCH_OFFLOAD -> "向量搜索卸载 (传输嵌入向量)"
            OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD -> "嵌入+搜索卸载 (传输裁剪的人脸图像)"
            OffloadingMode.FULL_OFFLOAD -> "全流程卸载 (传输完整相机帧)"
        }
    }
    
    /**
     * Get the components executed locally for each mode.
     */
    fun getLocalComponents(mode: OffloadingMode): List<String> {
        return when (mode) {
            OffloadingMode.LOCAL_ONLY -> listOf("人脸检测", "嵌入生成", "向量搜索", "活体检测")
            OffloadingMode.EMBEDDING_OFFLOAD -> listOf("人脸检测", "向量搜索", "活体检测")
            OffloadingMode.SEARCH_OFFLOAD -> listOf("人脸检测", "嵌入生成", "活体检测")
            OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD -> listOf("人脸检测", "活体检测")
            OffloadingMode.FULL_OFFLOAD -> listOf("活体检测")
        }
    }
    
    /**
     * Get the components executed on cloud for each mode.
     */
    fun getCloudComponents(mode: OffloadingMode): List<String> {
        return when (mode) {
            OffloadingMode.LOCAL_ONLY -> emptyList()
            OffloadingMode.EMBEDDING_OFFLOAD -> listOf("嵌入生成")
            OffloadingMode.SEARCH_OFFLOAD -> listOf("向量搜索")
            OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD -> listOf("嵌入生成", "向量搜索")
            OffloadingMode.FULL_OFFLOAD -> listOf("人脸检测", "嵌入生成", "向量搜索")
        }
    }
}

