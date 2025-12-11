package com.ml.shubham0204.facenet_android.domain.offloading

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import org.koin.core.annotation.Single
import java.io.ByteArrayOutputStream
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

/**
 * Cloud service client for offloading computations.
 * 
 * This class handles all network communication with the cloud server
 * for different offloading modes.
 */
@Single
class CloudService {
    
    companion object {
        private const val TAG = "CloudService"
    }
    
    /**
     * Data class for embedding response from server.
     */
    data class EmbeddingResponse(
        val embedding: FloatArray,
        val processingTimeMs: Float
    )
    
    /**
     * Data class for search response from server.
     */
    data class SearchResponse(
        val personName: String,
        val similarity: Float,
        val processingTimeMs: Float
    )
    
    /**
     * Data class for combined embedding and search response.
     */
    data class EmbeddingAndSearchResponse(
        val personName: String,
        val similarity: Float,
        val embedding: FloatArray,
        val embeddingTimeMs: Float,
        val searchTimeMs: Float,
        val totalTimeMs: Float
    )
    
    /**
     * Data class for full pipeline response.
     */
    data class FullPipelineResponse(
        val faces: List<FaceResult>,
        val metrics: PipelineMetrics
    )
    
    data class FaceResult(
        val bbox: BoundingBox,
        val personName: String,
        val similarity: Float
    )
    
    data class BoundingBox(
        val x: Int,
        val y: Int,
        val width: Int,
        val height: Int
    )
    
    data class PipelineMetrics(
        val faceDetectionMs: Float,
        val embeddingMs: Float,
        val searchMs: Float,
        val totalMs: Float,
        val numFaces: Int
    )
    
    /**
     * Convert Bitmap to Base64 string for network transfer.
     */
    private fun bitmapToBase64(bitmap: Bitmap, quality: Int = OffloadingConfig.imageQuality): String {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        val bytes = outputStream.toByteArray()
        return Base64.encodeToString(bytes, Base64.NO_WRAP)
    }
    
    /**
     * Make HTTP POST request to the server.
     */
    private suspend fun postJson(endpoint: String, jsonBody: JSONObject): String =
        withContext(Dispatchers.IO) {
            val url = URL("${OffloadingConfig.serverBaseUrl}$endpoint")
            val connection = url.openConnection() as HttpURLConnection
            
            try {
                connection.apply {
                    requestMethod = "POST"
                    setRequestProperty("Content-Type", "application/json")
                    setRequestProperty("Accept", "application/json")
                    connectTimeout = OffloadingConfig.connectTimeout.toInt()
                    readTimeout = OffloadingConfig.readTimeout.toInt()
                    doOutput = true
                }
                
                // Write request body
                OutputStreamWriter(connection.outputStream).use { writer ->
                    writer.write(jsonBody.toString())
                    writer.flush()
                }
                
                // Read response
                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    connection.inputStream.bufferedReader().use { it.readText() }
                } else {
                    val error = connection.errorStream?.bufferedReader()?.use { it.readText() } ?: "Unknown error"
                    throw Exception("HTTP $responseCode: $error")
                }
            } finally {
                connection.disconnect()
            }
        }
    
    /**
     * Mode 1: Generate face embedding on cloud server.
     * 
     * @param croppedFace Cropped face bitmap
     * @return EmbeddingResponse with the generated embedding
     */
    suspend fun generateEmbedding(croppedFace: Bitmap): EmbeddingResponse =
        withContext(Dispatchers.IO) {
            Log.d(TAG, "Generating embedding on cloud...")
            
            val base64Image = bitmapToBase64(croppedFace)
            val requestBody = JSONObject().apply {
                put("image_base64", base64Image)
            }
            
            val response = postJson("/api/v1/embedding", requestBody)
            val jsonResponse = JSONObject(response)
            
            val embeddingArray = jsonResponse.getJSONArray("embedding")
            val embedding = FloatArray(embeddingArray.length()) { i ->
                embeddingArray.getDouble(i).toFloat()
            }
            
            EmbeddingResponse(
                embedding = embedding,
                processingTimeMs = jsonResponse.getDouble("processing_time_ms").toFloat()
            )
        }
    
    /**
     * Mode 2: Perform vector search on cloud server.
     * 
     * @param embedding Query embedding vector
     * @param threshold Similarity threshold
     * @return SearchResponse with the search result
     */
    suspend fun searchVector(embedding: FloatArray, threshold: Float = 0.4f): SearchResponse =
        withContext(Dispatchers.IO) {
            Log.d(TAG, "Searching vector on cloud...")
            
            val embeddingArray = JSONArray()
            embedding.forEach { embeddingArray.put(it.toDouble()) }
            
            val requestBody = JSONObject().apply {
                put("query_embedding", embeddingArray)
                put("threshold", threshold.toDouble())
            }
            
            val response = postJson("/api/v1/search", requestBody)
            val jsonResponse = JSONObject(response)
            
            SearchResponse(
                personName = jsonResponse.getString("person_name"),
                similarity = jsonResponse.getDouble("similarity").toFloat(),
                processingTimeMs = jsonResponse.getDouble("processing_time_ms").toFloat()
            )
        }
    
    /**
     * Mode 3: Generate embedding and search on cloud server.
     * 
     * @param croppedFace Cropped face bitmap
     * @return EmbeddingAndSearchResponse with both embedding and search result
     */
    suspend fun embeddingAndSearch(croppedFace: Bitmap): EmbeddingAndSearchResponse =
        withContext(Dispatchers.IO) {
            Log.d(TAG, "Embedding and search on cloud...")
            
            val base64Image = bitmapToBase64(croppedFace)
            val requestBody = JSONObject().apply {
                put("image_base64", base64Image)
            }
            
            val response = postJson("/api/v1/embedding_and_search", requestBody)
            val jsonResponse = JSONObject(response)
            
            val embeddingArray = jsonResponse.getJSONArray("embedding")
            val embedding = FloatArray(embeddingArray.length()) { i ->
                embeddingArray.getDouble(i).toFloat()
            }
            
            val metrics = jsonResponse.getJSONObject("metrics")
            
            EmbeddingAndSearchResponse(
                personName = jsonResponse.getString("person_name"),
                similarity = jsonResponse.getDouble("similarity").toFloat(),
                embedding = embedding,
                embeddingTimeMs = metrics.getDouble("embedding_time_ms").toFloat(),
                searchTimeMs = metrics.getDouble("search_time_ms").toFloat(),
                totalTimeMs = metrics.getDouble("total_time_ms").toFloat()
            )
        }
    
    /**
     * Mode 4: Full pipeline on cloud server.
     * 
     * @param frameBitmap Full camera frame
     * @return FullPipelineResponse with all detection and recognition results
     */
    suspend fun fullPipeline(frameBitmap: Bitmap): FullPipelineResponse =
        withContext(Dispatchers.IO) {
            Log.d(TAG, "Full pipeline on cloud...")
            
            val base64Image = bitmapToBase64(frameBitmap)
            val requestBody = JSONObject().apply {
                put("image_base64", base64Image)
            }
            
            val response = postJson("/api/v1/full_pipeline", requestBody)
            val jsonResponse = JSONObject(response)
            
            val facesArray = jsonResponse.getJSONArray("faces")
            val faces = mutableListOf<FaceResult>()
            
            for (i in 0 until facesArray.length()) {
                val faceJson = facesArray.getJSONObject(i)
                val bboxJson = faceJson.getJSONObject("bbox")
                
                faces.add(FaceResult(
                    bbox = BoundingBox(
                        x = bboxJson.getInt("x"),
                        y = bboxJson.getInt("y"),
                        width = bboxJson.getInt("width"),
                        height = bboxJson.getInt("height")
                    ),
                    personName = faceJson.getString("person_name"),
                    similarity = faceJson.getDouble("similarity").toFloat()
                ))
            }
            
            val metricsJson = jsonResponse.getJSONObject("metrics")
            
            FullPipelineResponse(
                faces = faces,
                metrics = PipelineMetrics(
                    faceDetectionMs = metricsJson.getDouble("face_detection_ms").toFloat(),
                    embeddingMs = metricsJson.getDouble("embedding_ms").toFloat(),
                    searchMs = metricsJson.getDouble("search_ms").toFloat(),
                    totalMs = metricsJson.getDouble("total_ms").toFloat(),
                    numFaces = metricsJson.getInt("num_faces")
                )
            )
        }
    
    /**
     * Add a face to the cloud database.
     * Used for syncing local database to cloud for search offloading.
     */
    suspend fun addFaceToCloud(personId: Long, personName: String, embedding: FloatArray): Boolean =
        withContext(Dispatchers.IO) {
            Log.d(TAG, "Adding face to cloud: $personName")
            
            val embeddingArray = JSONArray()
            embedding.forEach { embeddingArray.put(it.toDouble()) }
            
            val requestBody = JSONObject().apply {
                put("person_id", personId)
                put("person_name", personName)
                put("embedding", embeddingArray)
            }
            
            try {
                postJson("/api/v1/faces/add", requestBody)
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to add face to cloud: ${e.message}")
                false
            }
        }
    
    /**
     * Check if the server is reachable.
     */
    suspend fun isServerReachable(): Boolean =
        withContext(Dispatchers.IO) {
            try {
                val url = URL("${OffloadingConfig.serverBaseUrl}/health")
                val connection = url.openConnection() as HttpURLConnection
                connection.connectTimeout = 3000
                connection.requestMethod = "GET"
                val responseCode = connection.responseCode
                connection.disconnect()
                responseCode == HttpURLConnection.HTTP_OK
            } catch (e: Exception) {
                Log.e(TAG, "Server not reachable: ${e.message}")
                false
            }
        }
    
    /**
     * Data class for performance metrics to upload.
     */
    data class PerformanceData(
        val sessionId: String,
        val mode: String,
        val modeName: String,
        val faceDetectionMs: Long,
        val embeddingMs: Long,
        val vectorSearchMs: Long,
        val spoofDetectionMs: Long,
        val networkMs: Long,
        val serverMs: Long,
        val totalMs: Long,
        val dataTransferredBytes: Long,
        val deviceInfo: String,
        val networkType: String
    )
    
    /**
     * Upload performance data to server.
     */
    suspend fun uploadPerformanceData(data: PerformanceData): Boolean =
        withContext(Dispatchers.IO) {
            Log.d(TAG, "Uploading performance data for ${data.modeName}...")
            
            val requestBody = JSONObject().apply {
                put("session_id", data.sessionId)
                put("mode", data.mode)
                put("mode_name", data.modeName)
                put("metrics", JSONObject().apply {
                    put("face_detection_ms", data.faceDetectionMs)
                    put("embedding_ms", data.embeddingMs)
                    put("vector_search_ms", data.vectorSearchMs)
                    put("spoof_detection_ms", data.spoofDetectionMs)
                    put("network_ms", data.networkMs)
                    put("server_ms", data.serverMs)
                    put("total_ms", data.totalMs)
                    put("data_transferred_bytes", data.dataTransferredBytes)
                })
                put("device_info", data.deviceInfo)
                put("network_type", data.networkType)
                put("timestamp", System.currentTimeMillis())
            }
            
            try {
                postJson("/api/v1/report/upload", requestBody)
                Log.d(TAG, "Performance data uploaded successfully")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to upload performance data: ${e.message}")
                false
            }
        }
    
    /**
     * Get upload status for all modes in current session.
     */
    suspend fun getUploadStatus(sessionId: String): Map<String, Boolean> =
        withContext(Dispatchers.IO) {
            try {
                val url = URL("${OffloadingConfig.serverBaseUrl}/api/v1/report/status?session_id=$sessionId")
                val connection = url.openConnection() as HttpURLConnection
                connection.connectTimeout = 5000
                connection.requestMethod = "GET"
                
                if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    val json = JSONObject(response)
                    val statusObj = json.getJSONObject("modes")
                    
                    val result = mutableMapOf<String, Boolean>()
                    statusObj.keys().forEach { key ->
                        result[key] = statusObj.getBoolean(key)
                    }
                    connection.disconnect()
                    result
                } else {
                    connection.disconnect()
                    emptyMap()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get upload status: ${e.message}")
                emptyMap()
            }
        }
    
    /**
     * Generate performance report.
     */
    suspend fun generateReport(sessionId: String): String? =
        withContext(Dispatchers.IO) {
            Log.d(TAG, "Generating report for session $sessionId...")
            
            val requestBody = JSONObject().apply {
                put("session_id", sessionId)
            }
            
            try {
                val response = postJson("/api/v1/report/generate", requestBody)
                val json = JSONObject(response)
                json.getString("report_id")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate report: ${e.message}")
                null
            }
        }
    
    /**
     * Get report page URL.
     */
    fun getReportUrl(reportId: String): String {
        return "${OffloadingConfig.serverBaseUrl}/report/$reportId"
    }
    
    /**
     * Get report list URL.
     */
    fun getReportListUrl(): String {
        return "${OffloadingConfig.serverBaseUrl}/report"
    }
}

