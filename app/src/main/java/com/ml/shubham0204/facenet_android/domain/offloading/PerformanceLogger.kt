package com.ml.shubham0204.facenet_android.domain.offloading

import android.content.Context
import android.os.Build
import android.util.Log
import org.koin.core.annotation.Single
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * Performance logger for recording and exporting experiment results.
 * 
 * This class collects performance metrics for different offloading modes
 * and exports them in CSV format for analysis.
 */
@Single
class PerformanceLogger(private val context: Context) {
    
    companion object {
        private const val TAG = "PerformanceLogger"
        private const val LOG_FILE_NAME = "performance_log.csv"
    }
    
    /**
     * Single performance record.
     */
    data class PerformanceRecord(
        val timestamp: Long,
        val offloadingMode: String,
        val timeFaceDetection: Long,
        val timeFaceEmbedding: Long,
        val timeVectorSearch: Long,
        val timeFaceSpoofDetection: Long,
        val timeNetworkTransfer: Long,
        val timeServerProcessing: Long,
        val totalTime: Long,
        val dataTransferredBytes: Long,
        val numFaces: Int,
        val networkType: String,
        val deviceModel: String,
        val serverHost: String
    )
    
    private val records = mutableListOf<PerformanceRecord>()
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.getDefault())
    
    /**
     * Log a performance metric.
     */
    fun logMetrics(
        metrics: OffloadingImageVectorUseCase.OffloadingMetrics,
        numFaces: Int,
        networkType: String = "WiFi"
    ) {
        val record = PerformanceRecord(
            timestamp = System.currentTimeMillis(),
            offloadingMode = metrics.offloadingMode.name,
            timeFaceDetection = metrics.timeFaceDetection,
            timeFaceEmbedding = metrics.timeFaceEmbedding,
            timeVectorSearch = metrics.timeVectorSearch,
            timeFaceSpoofDetection = metrics.timeFaceSpoofDetection,
            timeNetworkTransfer = metrics.timeNetworkTransfer,
            timeServerProcessing = metrics.timeServerProcessing,
            totalTime = metrics.totalTime,
            dataTransferredBytes = metrics.dataTransferredBytes,
            numFaces = numFaces,
            networkType = networkType,
            deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}",
            serverHost = OffloadingConfig.serverHost
        )
        
        records.add(record)
        Log.d(TAG, "Logged performance record: mode=${record.offloadingMode}, total=${record.totalTime}ms")
    }
    
    /**
     * Get statistics for a specific mode.
     */
    fun getStatistics(mode: OffloadingConfig.OffloadingMode): Map<String, Any> {
        val modeRecords = records.filter { it.offloadingMode == mode.name }
        
        if (modeRecords.isEmpty()) {
            return mapOf("error" to "No records for mode $mode")
        }
        
        val totalTimes = modeRecords.map { it.totalTime }
        val networkTimes = modeRecords.map { it.timeNetworkTransfer }
        val serverTimes = modeRecords.map { it.timeServerProcessing }
        
        return mapOf(
            "mode" to mode.name,
            "count" to modeRecords.size,
            "totalTime" to mapOf(
                "avg" to totalTimes.average(),
                "min" to totalTimes.minOrNull(),
                "max" to totalTimes.maxOrNull(),
                "std" to calculateStd(totalTimes)
            ),
            "networkTime" to mapOf(
                "avg" to networkTimes.average(),
                "min" to networkTimes.minOrNull(),
                "max" to networkTimes.maxOrNull()
            ),
            "serverTime" to mapOf(
                "avg" to serverTimes.average(),
                "min" to serverTimes.minOrNull(),
                "max" to serverTimes.maxOrNull()
            ),
            "avgDataTransferred" to modeRecords.map { it.dataTransferredBytes }.average()
        )
    }
    
    /**
     * Export records to CSV file.
     */
    fun exportToCsv(): File {
        val file = File(context.getExternalFilesDir(null), LOG_FILE_NAME)
        
        file.bufferedWriter().use { writer ->
            // Header
            writer.write("timestamp,datetime,mode,face_detection_ms,embedding_ms,vector_search_ms,")
            writer.write("spoof_detection_ms,network_transfer_ms,server_processing_ms,")
            writer.write("total_ms,data_bytes,num_faces,network_type,device,server\n")
            
            // Data
            for (record in records) {
                writer.write("${record.timestamp},")
                writer.write("${dateFormat.format(Date(record.timestamp))},")
                writer.write("${record.offloadingMode},")
                writer.write("${record.timeFaceDetection},")
                writer.write("${record.timeFaceEmbedding},")
                writer.write("${record.timeVectorSearch},")
                writer.write("${record.timeFaceSpoofDetection},")
                writer.write("${record.timeNetworkTransfer},")
                writer.write("${record.timeServerProcessing},")
                writer.write("${record.totalTime},")
                writer.write("${record.dataTransferredBytes},")
                writer.write("${record.numFaces},")
                writer.write("${record.networkType},")
                writer.write("${record.deviceModel},")
                writer.write("${record.serverHost}\n")
            }
        }
        
        Log.d(TAG, "Exported ${records.size} records to ${file.absolutePath}")
        return file
    }
    
    /**
     * Get all records.
     */
    fun getAllRecords(): List<PerformanceRecord> = records.toList()
    
    /**
     * Clear all records.
     */
    fun clearRecords() {
        records.clear()
        Log.d(TAG, "Cleared all performance records")
    }
    
    /**
     * Get record count.
     */
    fun getRecordCount(): Int = records.size
    
    /**
     * Generate summary report.
     */
    fun generateSummaryReport(): String {
        val sb = StringBuilder()
        sb.appendLine("=" .repeat(60))
        sb.appendLine("Performance Test Summary Report")
        sb.appendLine("Generated: ${dateFormat.format(Date())}")
        sb.appendLine("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        sb.appendLine("=" .repeat(60))
        sb.appendLine()
        
        for (mode in OffloadingConfig.OffloadingMode.values()) {
            val stats = getStatistics(mode)
            if (stats.containsKey("error")) continue
            
            sb.appendLine("Mode: ${mode.name}")
            sb.appendLine("-".repeat(40))
            sb.appendLine("  Sample count: ${stats["count"]}")
            
            @Suppress("UNCHECKED_CAST")
            val totalStats = stats["totalTime"] as Map<String, Any?>
            sb.appendLine("  Total time (ms):")
            sb.appendLine("    Average: ${"%.2f".format(totalStats["avg"])}")
            sb.appendLine("    Min: ${totalStats["min"]}")
            sb.appendLine("    Max: ${totalStats["max"]}")
            sb.appendLine("    Std: ${"%.2f".format(totalStats["std"])}")
            
            @Suppress("UNCHECKED_CAST")
            val networkStats = stats["networkTime"] as Map<String, Any?>
            sb.appendLine("  Network time (ms):")
            sb.appendLine("    Average: ${"%.2f".format(networkStats["avg"])}")
            
            @Suppress("UNCHECKED_CAST")
            val serverStats = stats["serverTime"] as Map<String, Any?>
            sb.appendLine("  Server processing (ms):")
            sb.appendLine("    Average: ${"%.2f".format(serverStats["avg"])}")
            
            sb.appendLine("  Avg data transferred: ${"%.2f".format(stats["avgDataTransferred"])} bytes")
            sb.appendLine()
        }
        
        return sb.toString()
    }
    
    private fun calculateStd(values: List<Long>): Double {
        if (values.size < 2) return 0.0
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return kotlin.math.sqrt(variance)
    }
}

