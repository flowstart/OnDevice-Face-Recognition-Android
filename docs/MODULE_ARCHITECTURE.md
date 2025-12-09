# 应用模块架构分析

## 1. 系统概述

本应用是一个在 Android 设备上运行的人脸识别系统，采用 Clean Architecture 架构设计，使用 Kotlin 语言和 Jetpack Compose UI 框架开发。

## 2. 模块架构图

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Presentation Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  DetectScreen   │  │  AddFaceScreen  │  │  FaceListScreen │              │
│  │  (实时识别界面)  │  │  (添加人脸界面)  │  │  (人脸列表界面)  │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐              │
│  │DetectScreenVM   │  │AddFaceScreenVM  │  │FaceListScreenVM │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
└───────────┼────────────────────┼────────────────────┼────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               Domain Layer                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        ImageVectorUseCase                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │人脸检测      │→│嵌入生成      │→│向量搜索      │→│活体检测      │  │   │
│  │  │MediaPipe    │  │FaceNet      │  │ImagesVectorDB│  │FaceSpoofDet│  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    OffloadingImageVectorUseCase                       │   │
│  │  支持5种计算划分模式，可将部分计算卸载到云端                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ PersonUseCase   │  │ CloudService    │  │PerformanceLogger│              │
│  │ (人员管理)       │  │ (云端通信)       │  │(性能日志)        │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                Data Layer                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  PersonDB       │  │  ImagesVectorDB │  │  ObjectBoxStore │              │
│  │  (人员数据库)    │  │  (向量数据库)    │  │  (数据库初始化)  │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心处理流程

```
Camera Frame                                                    Recognition Result
     │                                                                 ▲
     ▼                                                                 │
┌─────────────────────────────────────────────────────────────────────┴──────┐
│                           Face Recognition Pipeline                         │
│                                                                             │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│  │   Stage 1  │    │   Stage 2  │    │   Stage 3  │    │   Stage 4  │     │
│  │            │    │            │    │            │    │            │     │
│  │ Face       │───▶│ Face       │───▶│ Vector     │───▶│ Spoof      │     │
│  │ Detection  │    │ Embedding  │    │ Search     │    │ Detection  │     │
│  │            │    │            │    │            │    │            │     │
│  │ MediaPipe  │    │ FaceNet    │    │ ObjectBox  │    │ FASNet     │     │
│  │ BlazeFace  │    │ 512-dim    │    │ HNSW/Flat  │    │ Scale 2.7  │     │
│  │            │    │            │    │            │    │ Scale 4.0  │     │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘     │
│       │                 │                 │                 │              │
│       │                 │                 │                 │              │
│  Input: Bitmap     Input: Cropped    Input: 512-dim    Input: Frame +     │
│  Output: BBox      Face (160x160)    Float Vector      BBox               │
│         + Cropped  Output: 512-dim   Output: Nearest   Output: Spoof      │
│         Face       Float Vector      Neighbor          Score              │
│                                                                             │
│  Time: ~30ms       Time: ~100ms      Time: ~5ms        Time: ~50ms        │
│  (on device)       (on device)       (on device)       (on device)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 各模块详细说明

### 3.1 人脸检测模块 (MediapipeFaceDetector)

**文件位置**: `domain/face_detection/MediapipeFaceDetector.kt`

**功能**:
- 使用 MediaPipe BlazeFace 模型检测图像中的人脸
- 返回人脸边界框和裁剪后的人脸图像
- 处理图像旋转（EXIF 方向）

**输入输出**:
```
输入: Bitmap (相机帧或图片)
输出: List<Pair<Bitmap, Rect>> (裁剪的人脸图像, 边界框)
```

**性能特点**:
- 模型文件: `blaze_face_short_range.tflite` (~200KB)
- 典型耗时: 20-40ms (设备相关)
- 计算量: 轻量级

### 3.2 人脸嵌入模块 (FaceNet)

**文件位置**: `domain/embeddings/FaceNet.kt`

**功能**:
- 使用 FaceNet 模型生成人脸嵌入向量
- 输入 160x160 的人脸图像
- 输出 512 维浮点向量

**输入输出**:
```
输入: Bitmap (160x160 裁剪的人脸)
输出: FloatArray (512 维嵌入向量)
```

**性能特点**:
- 模型文件: `facenet_512.tflite` (~92MB)
- 典型耗时: 80-150ms (设备相关)
- 计算量: **重量级** (最耗时的模块)
- 支持 GPU 加速

### 3.3 向量搜索模块 (ImagesVectorDB)

**文件位置**: `data/ImagesVectorDB.kt`

**功能**:
- 存储人脸嵌入向量
- 执行最近邻搜索（余弦相似度）
- 支持 HNSW 近似搜索和精确搜索

**输入输出**:
```
输入: FloatArray (512 维查询向量)
输出: FaceImageRecord? (最近邻记录，包含人名)
```

**性能特点**:
- 数据库: ObjectBox (支持向量索引)
- HNSW 搜索: ~1-5ms
- 精确搜索: ~10-50ms (取决于数据库大小)

### 3.4 活体检测模块 (FaceSpoofDetector)

**文件位置**: `domain/face_detection/FaceSpoofDetector.kt`

**功能**:
- 检测人脸是否为欺骗攻击（照片、视频等）
- 使用两个不同尺度的 MiniFASNet 模型
- 输出欺骗分数和判定结果

**输入输出**:
```
输入: Bitmap (原始帧), Rect (人脸边界框)
输出: FaceSpoofResult (isSpoof, score, timeMillis)
```

**性能特点**:
- 模型文件: `spoof_model_scale_2_7.tflite`, `spoof_model_scale_4_0.tflite`
- 典型耗时: 30-60ms
- 计算量: 中等

### 3.5 云端服务模块 (CloudService)

**文件位置**: `domain/offloading/CloudService.kt`

**功能**:
- 与云端服务器通信
- 支持四种卸载模式的 API 调用
- 处理图像序列化和网络传输

**API 端点**:
- `/api/v1/embedding` - 嵌入生成
- `/api/v1/search` - 向量搜索
- `/api/v1/embedding_and_search` - 嵌入+搜索
- `/api/v1/full_pipeline` - 完整流程

## 4. 数据模型

### 4.1 FaceImageRecord

```kotlin
@Entity
data class FaceImageRecord(
    @Id var recordID: Long = 0,
    @Index var personID: Long = 0,
    var personName: String = "",
    @HnswIndex(dimensions = 512, distanceType = VectorDistanceType.COSINE)
    var faceEmbedding: FloatArray = floatArrayOf()
)
```

### 4.2 PersonRecord

```kotlin
@Entity
data class PersonRecord(
    @Id var personID: Long = 0,
    var personName: String = "",
    var numImages: Long = 0,
    var addTime: Long = 0
)
```

### 4.3 RecognitionMetrics

```kotlin
data class RecognitionMetrics(
    val timeFaceDetection: Long,
    val timeVectorSearch: Long,
    val timeFaceEmbedding: Long,
    val timeFaceSpoofDetection: Long
)
```

## 5. 依赖关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            External Dependencies                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ TensorFlow Lite │  │    MediaPipe    │  │    ObjectBox    │              │
│  │ (ML Runtime)    │  │ (Face Detection)│  │ (Vector Store)  │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│           ▼                    ▼                    ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Domain Layer                                 │   │
│  │                                                                       │   │
│  │  FaceNet.kt ◀──── TFLite                                            │   │
│  │  FaceSpoofDetector.kt ◀──── TFLite                                  │   │
│  │  MediapipeFaceDetector.kt ◀──── MediaPipe                           │   │
│  │  ImagesVectorDB.kt ◀──── ObjectBox                                  │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Jetpack Compose │  │    CameraX      │  │      Koin       │              │
│  │ (UI Framework)  │  │ (Camera Access) │  │ (DI Framework)  │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 6. 计算复杂度分析

| 模块 | 计算类型 | 时间复杂度 | 内存占用 | 可卸载性 |
|------|---------|-----------|---------|---------|
| 人脸检测 | CNN 推理 | O(WH) | ~50MB | 高 |
| 嵌入生成 | CNN 推理 | O(1) | ~200MB | **很高** |
| 向量搜索 | 近邻搜索 | O(log N) / O(N) | ~N×2KB | 中 |
| 活体检测 | CNN 推理 | O(1) | ~30MB | 高 |

**说明**:
- W, H: 图像宽高
- N: 数据库中的记录数
- 嵌入生成是最适合卸载的模块（计算密集、输入输出数据量适中）

## 7. 模块划分适配性评估

| 划分方案 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| 全本地 | 无网络依赖、低延迟 | 耗电、受限于设备性能 | 离线、低延迟要求 |
| 嵌入卸载 | 减轻移动端负担 | 需传输图像 (~50KB) | WiFi 环境 |
| 搜索卸载 | 传输量最小 (~2KB) | 需同步数据库 | 大规模数据库 |
| 嵌入+搜索 | 移动端负担最小 | 图像传输开销 | 低端设备 |
| 全卸载 | 移动端几乎无计算 | 大量数据传输 (~200KB) | 高带宽、云资源丰富 |

