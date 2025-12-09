package com.ml.shubham0204.facenet_android.domain.offloading

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.ml.shubham0204.facenet_android.data.FaceImageRecord
import com.ml.shubham0204.facenet_android.data.ImagesVectorDB
import com.ml.shubham0204.facenet_android.data.RecognitionMetrics
import com.ml.shubham0204.facenet_android.domain.ImageVectorUseCase
import com.ml.shubham0204.facenet_android.domain.embeddings.FaceNet
import com.ml.shubham0204.facenet_android.domain.face_detection.FaceSpoofDetector
import com.ml.shubham0204.facenet_android.domain.face_detection.MediapipeFaceDetector
import org.koin.core.annotation.Single
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.time.DurationUnit
import kotlin.time.measureTimedValue

/**
 * Extended ImageVectorUseCase that supports computation offloading.
 * 
 * This class implements different partitioning strategies for face recognition:
 * - Mode 0: All local (baseline)
 * - Mode 1: Offload embedding generation
 * - Mode 2: Offload vector search
 * - Mode 3: Offload embedding + search
 * - Mode 4: Offload full pipeline
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
        val dataTransferredBytes: Long
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
                processEmbeddingOffload(frameBitmap, flatSearch)
            OffloadingConfig.OffloadingMode.SEARCH_OFFLOAD -> 
                processSearchOffload(frameBitmap)
            OffloadingConfig.OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD -> 
                processEmbeddingAndSearchOffload(frameBitmap)
            OffloadingConfig.OffloadingMode.FULL_OFFLOAD -> 
                processFullOffload(frameBitmap)
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
            dataTransferredBytes = 0
        )
        
        return Pair(metrics, results)
    }
    
    /**
     * Mode 1: Offload embedding generation to cloud.
     */
    private suspend fun processEmbeddingOffload(
        frameBitmap: Bitmap,
        flatSearch: Boolean
    ): Pair<OffloadingMetrics?, List<OffloadingFaceRecognitionResult>> {
        val totalStart = System.currentTimeMillis()
        var totalNetworkTime = 0L
        var totalServerTime = 0L
        var totalDataTransferred = 0L
        
        // Local: Face detection
        val (faceDetectionResult, t1) = measureTimedValue { 
            mediapipeFaceDetector.getAllCroppedFaces(frameBitmap) 
        }
        
        val results = ArrayList<OffloadingFaceRecognitionResult>()
        var avgT3 = 0L
        var avgT4 = 0L
        
        for ((croppedBitmap, boundingBox) in faceDetectionResult) {
            // Cloud: Embedding generation
            val networkStart = System.currentTimeMillis()
            val embeddingResponse = try {
                cloudService.generateEmbedding(croppedBitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Cloud embedding failed, falling back to local: ${e.message}")
                // Fallback to local
                val embedding = faceNet.getFaceEmbedding(croppedBitmap)
                CloudService.EmbeddingResponse(embedding, 0f)
            }
            val networkTime = System.currentTimeMillis() - networkStart
            totalNetworkTime += networkTime
            totalServerTime += embeddingResponse.processingTimeMs.toLong()
            totalDataTransferred += estimateImageSize(croppedBitmap)
            
            val embedding = embeddingResponse.embedding
            
            // Local: Vector search
            val (recognitionResult, t3) = measureTimedValue { 
                imagesVectorDB.getNearestEmbeddingPersonName(embedding, flatSearch) 
            }
            avgT3 += t3.toLong(DurationUnit.MILLISECONDS)
            
            if (recognitionResult == null) {
                results.add(OffloadingFaceRecognitionResult("Not recognized", boundingBox))
                continue
            }
            
            // Local: Spoof detection
            val spoofResult = faceSpoofDetector.detectSpoof(frameBitmap, boundingBox)
            avgT4 += spoofResult.timeMillis
            
            val distance = cosineDistance(embedding, recognitionResult.faceEmbedding)
            
            if (distance > 0.4) {
                results.add(OffloadingFaceRecognitionResult(
                    recognitionResult.personName, boundingBox, spoofResult, distance
                ))
            } else {
                results.add(OffloadingFaceRecognitionResult(
                    "Not recognized", boundingBox, spoofResult, distance
                ))
            }
        }
        
        val totalTime = System.currentTimeMillis() - totalStart
        val numFaces = faceDetectionResult.size.coerceAtLeast(1)
        
        val metrics = OffloadingMetrics(
            timeFaceDetection = t1.toLong(DurationUnit.MILLISECONDS),
            timeFaceEmbedding = 0, // Done on server
            timeVectorSearch = avgT3 / numFaces,
            timeFaceSpoofDetection = avgT4 / numFaces,
            timeNetworkTransfer = totalNetworkTime - totalServerTime,
            timeServerProcessing = totalServerTime,
            totalTime = totalTime,
            offloadingMode = OffloadingConfig.OffloadingMode.EMBEDDING_OFFLOAD,
            dataTransferredBytes = totalDataTransferred
        )
        
        return Pair(metrics, results)
    }
    
    /**
     * Mode 2: Offload vector search to cloud.
     */
    private suspend fun processSearchOffload(
        frameBitmap: Bitmap
    ): Pair<OffloadingMetrics?, List<OffloadingFaceRecognitionResult>> {
        val totalStart = System.currentTimeMillis()
        var totalNetworkTime = 0L
        var totalServerTime = 0L
        var totalDataTransferred = 0L
        
        // Local: Face detection
        val (faceDetectionResult, t1) = measureTimedValue { 
            mediapipeFaceDetector.getAllCroppedFaces(frameBitmap) 
        }
        
        val results = ArrayList<OffloadingFaceRecognitionResult>()
        var avgT2 = 0L
        var avgT4 = 0L
        
        for ((croppedBitmap, boundingBox) in faceDetectionResult) {
            // Local: Embedding generation
            val (embedding, t2) = measureTimedValue { faceNet.getFaceEmbedding(croppedBitmap) }
            avgT2 += t2.toLong(DurationUnit.MILLISECONDS)
            
            // Cloud: Vector search
            val networkStart = System.currentTimeMillis()
            val searchResponse = try {
                cloudService.searchVector(embedding)
            } catch (e: Exception) {
                Log.e(TAG, "Cloud search failed, falling back to local: ${e.message}")
                val localResult = imagesVectorDB.getNearestEmbeddingPersonName(embedding, false)
                CloudService.SearchResponse(
                    localResult?.personName ?: "Not recognized",
                    if (localResult != null) cosineDistance(embedding, localResult.faceEmbedding) else 0f,
                    0f
                )
            }
            val networkTime = System.currentTimeMillis() - networkStart
            totalNetworkTime += networkTime
            totalServerTime += searchResponse.processingTimeMs.toLong()
            totalDataTransferred += embedding.size * 4L // 4 bytes per float
            
            // Local: Spoof detection
            val spoofResult = faceSpoofDetector.detectSpoof(frameBitmap, boundingBox)
            avgT4 += spoofResult.timeMillis
            
            if (searchResponse.similarity > 0.4) {
                results.add(OffloadingFaceRecognitionResult(
                    searchResponse.personName, boundingBox, spoofResult, searchResponse.similarity
                ))
            } else {
                results.add(OffloadingFaceRecognitionResult(
                    "Not recognized", boundingBox, spoofResult, searchResponse.similarity
                ))
            }
        }
        
        val totalTime = System.currentTimeMillis() - totalStart
        val numFaces = faceDetectionResult.size.coerceAtLeast(1)
        
        val metrics = OffloadingMetrics(
            timeFaceDetection = t1.toLong(DurationUnit.MILLISECONDS),
            timeFaceEmbedding = avgT2 / numFaces,
            timeVectorSearch = 0, // Done on server
            timeFaceSpoofDetection = avgT4 / numFaces,
            timeNetworkTransfer = totalNetworkTime - totalServerTime,
            timeServerProcessing = totalServerTime,
            totalTime = totalTime,
            offloadingMode = OffloadingConfig.OffloadingMode.SEARCH_OFFLOAD,
            dataTransferredBytes = totalDataTransferred
        )
        
        return Pair(metrics, results)
    }
    
    /**
     * Mode 3: Offload embedding + search to cloud.
     */
    private suspend fun processEmbeddingAndSearchOffload(
        frameBitmap: Bitmap
    ): Pair<OffloadingMetrics?, List<OffloadingFaceRecognitionResult>> {
        val totalStart = System.currentTimeMillis()
        var totalNetworkTime = 0L
        var totalServerTime = 0L
        var totalDataTransferred = 0L
        
        // Local: Face detection
        val (faceDetectionResult, t1) = measureTimedValue { 
            mediapipeFaceDetector.getAllCroppedFaces(frameBitmap) 
        }
        
        val results = ArrayList<OffloadingFaceRecognitionResult>()
        var avgT4 = 0L
        
        for ((croppedBitmap, boundingBox) in faceDetectionResult) {
            // Cloud: Embedding + Search
            val networkStart = System.currentTimeMillis()
            val response = try {
                cloudService.embeddingAndSearch(croppedBitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Cloud embedding+search failed, falling back to local: ${e.message}")
                val embedding = faceNet.getFaceEmbedding(croppedBitmap)
                val localResult = imagesVectorDB.getNearestEmbeddingPersonName(embedding, false)
                CloudService.EmbeddingAndSearchResponse(
                    localResult?.personName ?: "Not recognized",
                    if (localResult != null) cosineDistance(embedding, localResult.faceEmbedding) else 0f,
                    embedding,
                    0f, 0f, 0f
                )
            }
            val networkTime = System.currentTimeMillis() - networkStart
            totalNetworkTime += networkTime
            totalServerTime += response.totalTimeMs.toLong()
            totalDataTransferred += estimateImageSize(croppedBitmap)
            
            // Local: Spoof detection
            val spoofResult = faceSpoofDetector.detectSpoof(frameBitmap, boundingBox)
            avgT4 += spoofResult.timeMillis
            
            if (response.similarity > 0.4) {
                results.add(OffloadingFaceRecognitionResult(
                    response.personName, boundingBox, spoofResult, response.similarity
                ))
            } else {
                results.add(OffloadingFaceRecognitionResult(
                    "Not recognized", boundingBox, spoofResult, response.similarity
                ))
            }
        }
        
        val totalTime = System.currentTimeMillis() - totalStart
        val numFaces = faceDetectionResult.size.coerceAtLeast(1)
        
        val metrics = OffloadingMetrics(
            timeFaceDetection = t1.toLong(DurationUnit.MILLISECONDS),
            timeFaceEmbedding = 0, // Done on server
            timeVectorSearch = 0, // Done on server
            timeFaceSpoofDetection = avgT4 / numFaces,
            timeNetworkTransfer = totalNetworkTime - totalServerTime,
            timeServerProcessing = totalServerTime,
            totalTime = totalTime,
            offloadingMode = OffloadingConfig.OffloadingMode.EMBEDDING_AND_SEARCH_OFFLOAD,
            dataTransferredBytes = totalDataTransferred
        )
        
        return Pair(metrics, results)
    }
    
    /**
     * Mode 4: Full pipeline offloading to cloud.
     */
    private suspend fun processFullOffload(
        frameBitmap: Bitmap
    ): Pair<OffloadingMetrics?, List<OffloadingFaceRecognitionResult>> {
        val totalStart = System.currentTimeMillis()
        
        // Estimate data size
        val dataTransferred = estimateImageSize(frameBitmap)
        
        // Cloud: Full pipeline
        val networkStart = System.currentTimeMillis()
        val response = try {
            cloudService.fullPipeline(frameBitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Cloud full pipeline failed: ${e.message}")
            // Return empty result on failure
            return Pair(null, emptyList())
        }
        val networkTime = System.currentTimeMillis() - networkStart
        
        val results = ArrayList<OffloadingFaceRecognitionResult>()
        var avgT4 = 0L
        
        // Local: Spoof detection (still needs to be local for now)
        for (face in response.faces) {
            val boundingBox = Rect(
                face.bbox.x,
                face.bbox.y,
                face.bbox.x + face.bbox.width,
                face.bbox.y + face.bbox.height
            )
            
            val spoofResult = faceSpoofDetector.detectSpoof(frameBitmap, boundingBox)
            avgT4 += spoofResult.timeMillis
            
            if (face.similarity > 0.4) {
                results.add(OffloadingFaceRecognitionResult(
                    face.personName, boundingBox, spoofResult, face.similarity
                ))
            } else {
                results.add(OffloadingFaceRecognitionResult(
                    "Not recognized", boundingBox, spoofResult, face.similarity
                ))
            }
        }
        
        val totalTime = System.currentTimeMillis() - totalStart
        val numFaces = response.faces.size.coerceAtLeast(1)
        
        val metrics = OffloadingMetrics(
            timeFaceDetection = 0, // Done on server
            timeFaceEmbedding = 0, // Done on server
            timeVectorSearch = 0, // Done on server
            timeFaceSpoofDetection = avgT4 / numFaces,
            timeNetworkTransfer = networkTime - response.metrics.totalMs.toLong(),
            timeServerProcessing = response.metrics.totalMs.toLong(),
            totalTime = totalTime,
            offloadingMode = OffloadingConfig.OffloadingMode.FULL_OFFLOAD,
            dataTransferredBytes = dataTransferred
        )
        
        return Pair(metrics, results)
    }
    
    /**
     * Estimate compressed image size in bytes.
     */
    private fun estimateImageSize(bitmap: Bitmap): Long {
        val quality = OffloadingConfig.imageQuality
        // Rough estimation: width * height * 3 (RGB) * quality/100 * compression_factor
        val compressionFactor = 0.1f // JPEG compression is very effective
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

