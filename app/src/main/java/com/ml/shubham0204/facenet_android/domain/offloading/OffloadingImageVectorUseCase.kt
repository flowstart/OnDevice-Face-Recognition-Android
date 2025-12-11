package com.ml.shubham0204.facenet_android.domain.offloading

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.ml.shubham0204.facenet_android.data.ImagesVectorDB
import com.ml.shubham0204.facenet_android.domain.embeddings.FaceNet
import com.ml.shubham0204.facenet_android.domain.face_detection.FaceSpoofDetector
import com.ml.shubham0204.facenet_android.domain.face_detection.MediapipeFaceDetector
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import org.koin.core.annotation.Single
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.time.DurationUnit
import kotlin.time.measureTimedValue

/**
 * Extended ImageVectorUseCase that supports computation offloading simulation.
 * 
 * 核心逻辑：
 * - 所有模式都使用本地能力进行识别（确保功能正常）
 * - Mode 1-4 发真实网络请求获取延迟数据
 * - 等网络请求返回后再返回结果（模拟云端等待效果）
 * - 断网时回退到本地，并标记 isFallback = true
 */
@Single
class OffloadingImageVectorUseCase(
    private val mediapipeFaceDetector: MediapipeFaceDetector,
    private val faceSpoofDetector: FaceSpoofDetector,
    private val imagesVectorDB: ImagesVectorDB,
    private val faceNet: FaceNet,
    private val cloudService: CloudService,
) {
    companion object {
        private const val TAG = "OffloadingImageVectorUseCase"
    }
    
    /**
     * Extended metrics including network time.
     */
    data class OffloadingMetrics(
        val timeFaceDetection: Long,
        val timeFaceEmbedding: Long,
        val timeVectorSearch: Long,
        val timeFaceSpoofDetection: Long,
        val timeNetworkTransfer: Long,
        val timeServerProcessing: Long,
        val totalTime: Long,
        val offloadingMode: OffloadingConfig.OffloadingMode,
        val dataTransferredBytes: Long,
        val isFallback: Boolean = false  // 是否回退到本地模式
    )
    
    /**
     * Face recognition result with additional offloading info.
     */
    data class OffloadingFaceRecognitionResult(
        val personName: String,
        val boundingBox: Rect,
        val spoofResult: FaceSpoofDetector.FaceSpoofResult? = null,
        val similarity: Float = 0f
    )
    
    /**
     * Perform face recognition with the configured offloading mode.
     */
    suspend fun getNearestPersonNameWithOffloading(
        frameBitmap: Bitmap,
        flatSearch: Boolean = false,
    ): Pair<OffloadingMetrics?, List<OffloadingFaceRecognitionResult>> {
        val mode = OffloadingConfig.currentMode
        Log.d(TAG, "Processing with mode: $mode")
        
        return when (mode) {
            OffloadingConfig.OffloadingMode.LOCAL_ONLY -> 
                processLocalOnly(frameBitmap, flatSearch)
            OffloadingConfig.OffloadingMode.EMBEDDING_OFFLOAD -> 
                processWithCloudDelay(frameBitmap, flatSearch, mode)
            OffloadingConfig.OffloadingMode.SEARCH_OFFLOAD -> 
                processWithCloudDelay(frameBitmap, flatSearch, mode)
            OffloadingConfig.OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD -> 
                processWithCloudDelay(frameBitmap, flatSearch, mode)
            OffloadingConfig.OffloadingMode.FULL_OFFLOAD -> 
                processWithCloudDelay(frameBitmap, flatSearch, mode)
        }
    }
    
    /**
     * Mode 0: All computation on local device.
     */
    private suspend fun processLocalOnly(
        frameBitmap: Bitmap,
        flatSearch: Boolean
    ): Pair<OffloadingMetrics?, List<OffloadingFaceRecognitionResult>> {
        val totalStart = System.currentTimeMillis()
        
        // Face detection
        val (faceDetectionResult, t1) = measureTimedValue { 
            mediapipeFaceDetector.getAllCroppedFaces(frameBitmap) 
        }
        
        val results = ArrayList<OffloadingFaceRecognitionResult>()
        var avgT2 = 0L
        var avgT3 = 0L
        var avgT4 = 0L
        
        for ((croppedBitmap, boundingBox) in faceDetectionResult) {
            // Embedding generation
            val (embedding, t2) = measureTimedValue { faceNet.getFaceEmbedding(croppedBitmap) }
            avgT2 += t2.toLong(DurationUnit.MILLISECONDS)
            
            // Vector search
            val (recognitionResult, t3) = measureTimedValue { 
                imagesVectorDB.getNearestEmbeddingPersonName(embedding, flatSearch) 
            }
            avgT3 += t3.toLong(DurationUnit.MILLISECONDS)
            
            if (recognitionResult == null) {
                results.add(OffloadingFaceRecognitionResult("Not recognized", boundingBox))
                continue
            }
            
            // Spoof detection
            val spoofResult = faceSpoofDetector.detectSpoof(frameBitmap, boundingBox)
            avgT4 += spoofResult.timeMillis
            
            // Calculate similarity
            val distance = cosineDistance(embedding, recognitionResult.faceEmbedding)
            
            if (distance > 0.4) {
                results.add(OffloadingFaceRecognitionResult(
                    recognitionResult.personName, 
                    boundingBox, 
                    spoofResult,
                    distance
                ))
            } else {
                results.add(OffloadingFaceRecognitionResult(
                    "Not recognized", 
                    boundingBox, 
                    spoofResult,
                    distance
                ))
            }
        }
        
        val totalTime = System.currentTimeMillis() - totalStart
        val numFaces = faceDetectionResult.size.coerceAtLeast(1)
        
        val metrics = OffloadingMetrics(
            timeFaceDetection = t1.toLong(DurationUnit.MILLISECONDS),
            timeFaceEmbedding = avgT2 / numFaces,
            timeVectorSearch = avgT3 / numFaces,
            timeFaceSpoofDetection = avgT4 / numFaces,
            timeNetworkTransfer = 0,
            timeServerProcessing = 0,
            totalTime = totalTime,
            offloadingMode = OffloadingConfig.OffloadingMode.LOCAL_ONLY,
            dataTransferredBytes = 0,
            isFallback = false
        )
        
        return Pair(metrics, results)
    }
    
    /**
     * Mode 1-4: 使用本地能力识别，同时发网络请求获取延迟数据
     * 
     * 逻辑：
     * 1. 并行执行：本地识别 + 网络请求
     * 2. 等待两者都完成
     * 3. 使用本地识别的结果，但记录网络延迟数据
     */
    private suspend fun processWithCloudDelay(
        frameBitmap: Bitmap,
        flatSearch: Boolean,
        mode: OffloadingConfig.OffloadingMode
    ): Pair<OffloadingMetrics?, List<OffloadingFaceRecognitionResult>> = coroutineScope {
        val totalStart = System.currentTimeMillis()
        var isFallback = false
        var networkTime = 0L
        var serverTime = 0L
        var dataTransferred = 0L
        
        // Face detection (always local)
        val (faceDetectionResult, t1) = measureTimedValue { 
            mediapipeFaceDetector.getAllCroppedFaces(frameBitmap) 
        }
        
        val results = ArrayList<OffloadingFaceRecognitionResult>()
        var avgT2 = 0L
        var avgT3 = 0L
        var avgT4 = 0L
        
        for ((croppedBitmap, boundingBox) in faceDetectionResult) {
            // 并行执行：本地识别 + 网络请求
            val localResultDeferred = async {
                // 本地执行嵌入生成
                val (embedding, embeddingTime) = measureTimedValue { 
                    faceNet.getFaceEmbedding(croppedBitmap) 
                }
                avgT2 += embeddingTime.toLong(DurationUnit.MILLISECONDS)
                
                // 本地执行向量搜索
                val (searchResult, searchTime) = measureTimedValue { 
                    imagesVectorDB.getNearestEmbeddingPersonName(embedding, flatSearch) 
                }
                avgT3 += searchTime.toLong(DurationUnit.MILLISECONDS)
                
                Pair(embedding, searchResult)
            }
            
            // 先等待本地识别完成
            val (embedding, searchResult) = localResultDeferred.await()
            
            // 然后发网络请求获取延迟数据（避免 TFLite 并发问题）
            val networkDeferred = async {
                try {
                    val networkStart = System.currentTimeMillis()
                    val response = when (mode) {
                        OffloadingConfig.OffloadingMode.EMBEDDING_OFFLOAD -> {
                            dataTransferred += estimateImageSize(croppedBitmap)
                            val resp = cloudService.generateEmbedding(croppedBitmap)
                            NetworkResult(resp.processingTimeMs.toLong(), false)
                        }
                        OffloadingConfig.OffloadingMode.SEARCH_OFFLOAD -> {
                            // 使用已生成的 embedding，避免重复调用 TFLite
                            dataTransferred += embedding.size * 4L
                            val resp = cloudService.searchVector(embedding)
                            NetworkResult(resp.processingTimeMs.toLong(), false)
                        }
                        OffloadingConfig.OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD -> {
                            dataTransferred += estimateImageSize(croppedBitmap)
                            val resp = cloudService.embeddingAndSearch(croppedBitmap)
                            NetworkResult(resp.totalTimeMs.toLong(), false)
                        }
                        OffloadingConfig.OffloadingMode.FULL_OFFLOAD -> {
                            dataTransferred += estimateImageSize(frameBitmap)
                            val resp = cloudService.fullPipeline(frameBitmap)
                            NetworkResult(resp.metrics.totalMs.toLong(), false)
                        }
                        else -> NetworkResult(0, false)
                    }
                    val totalNetworkTime = System.currentTimeMillis() - networkStart
                    networkTime = totalNetworkTime
                    serverTime = response.serverTime
                    response
                } catch (e: Exception) {
                    Log.w(TAG, "Network request failed, using fallback: ${e.message}")
                    isFallback = true
                    NetworkResult(0, true)
                }
            }
            val networkResult = networkDeferred.await()
            
            if (networkResult.failed) {
                isFallback = true
            }
            
            if (searchResult == null) {
                results.add(OffloadingFaceRecognitionResult("Not recognized", boundingBox))
                continue
            }
            
            // Spoof detection (always local)
            val spoofResult = faceSpoofDetector.detectSpoof(frameBitmap, boundingBox)
            avgT4 += spoofResult.timeMillis
            
            // 使用本地识别的结果
            val distance = cosineDistance(embedding, searchResult.faceEmbedding)
            
            if (distance > 0.4) {
                results.add(OffloadingFaceRecognitionResult(
                    searchResult.personName, boundingBox, spoofResult, distance
                ))
            } else {
                results.add(OffloadingFaceRecognitionResult(
                    "Not recognized", boundingBox, spoofResult, distance
                ))
            }
        }
        
        val totalTime = System.currentTimeMillis() - totalStart
        val numFaces = faceDetectionResult.size.coerceAtLeast(1)
        
        // 根据模式决定哪些时间显示为0（表示在云端执行）
        val (displayEmbeddingTime, displaySearchTime) = when (mode) {
            OffloadingConfig.OffloadingMode.EMBEDDING_OFFLOAD -> 
                Pair(0L, avgT3 / numFaces)  // 嵌入在云端，搜索在本地
            OffloadingConfig.OffloadingMode.SEARCH_OFFLOAD -> 
                Pair(avgT2 / numFaces, 0L)  // 嵌入在本地，搜索在云端
            OffloadingConfig.OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD,
            OffloadingConfig.OffloadingMode.FULL_OFFLOAD -> 
                Pair(0L, 0L)  // 都在云端
            else -> Pair(avgT2 / numFaces, avgT3 / numFaces)
        }
        
        val displayDetectionTime = if (mode == OffloadingConfig.OffloadingMode.FULL_OFFLOAD) {
            0L  // Mode 4: 人脸检测也在云端
        } else {
            t1.toLong(DurationUnit.MILLISECONDS)
        }
        
        val metrics = OffloadingMetrics(
            timeFaceDetection = displayDetectionTime,
            timeFaceEmbedding = displayEmbeddingTime,
            timeVectorSearch = displaySearchTime,
            timeFaceSpoofDetection = avgT4 / numFaces,
            timeNetworkTransfer = (networkTime - serverTime).coerceAtLeast(0),
            timeServerProcessing = serverTime,
            totalTime = totalTime,
            offloadingMode = mode,
            dataTransferredBytes = dataTransferred,
            isFallback = isFallback
        )
        
        Pair(metrics, results)
    }
    
    /**
     * Network request result holder.
     */
    private data class NetworkResult(
        val serverTime: Long,
        val failed: Boolean
    )
    
    /**
     * Estimate compressed image size in bytes.
     */
    private fun estimateImageSize(bitmap: Bitmap): Long {
        val quality = OffloadingConfig.imageQuality
        val compressionFactor = 0.1f
        return (bitmap.width * bitmap.height * 3 * (quality / 100f) * compressionFactor).toLong()
    }
    
    private fun cosineDistance(x1: FloatArray, x2: FloatArray): Float {
        var mag1 = 0.0f
        var mag2 = 0.0f
        var product = 0.0f
        for (i in x1.indices) {
            mag1 += x1[i].pow(2)
            mag2 += x2[i].pow(2)
            product += x1[i] * x2[i]
        }
        mag1 = sqrt(mag1)
        mag2 = sqrt(mag2)
        return product / (mag1 * mag2)
    }
}
