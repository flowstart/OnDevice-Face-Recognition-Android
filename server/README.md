# Face Recognition Cloud Server

云端人脸识别计算卸载服务器

## 功能概述

本服务器提供 REST API 用于将移动端的计算密集型任务卸载到云端执行。

### 支持的卸载模式

| 模式 | 描述 | API 端点 |
|------|------|---------|
| Mode 1: 嵌入生成卸载 | 仅卸载 FaceNet 嵌入生成 | `/api/v1/embedding` |
| Mode 2: 向量搜索卸载 | 仅卸载向量数据库搜索 | `/api/v1/search` |
| Mode 3: 嵌入+搜索卸载 | 卸载嵌入生成和向量搜索 | `/api/v1/embedding_and_search` |
| Mode 4: 全流程卸载 | 卸载整个识别流程 | `/api/v1/full_pipeline` |

## 快速开始

### 1. 安装依赖

```bash
cd server
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. 准备模型文件

将以下模型文件放入 `server/models/` 目录：
- `facenet_512.tflite` - FaceNet 人脸嵌入模型
- `blaze_face_short_range.tflite` - MediaPipe 人脸检测模型

可以从 Android 项目的 `app/src/main/assets/` 目录复制。

```bash
mkdir -p models
cp ../app/src/main/assets/facenet_512.tflite models/
cp ../app/src/main/assets/blaze_face_short_range.tflite models/
```

### 3. 启动服务器

```bash
# 开发模式
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. 访问 API 文档

打开浏览器访问: http://localhost:8000/docs

## API 使用示例

### 模式1: 嵌入生成

```python
import requests
import base64

# 读取图片并转为 base64
with open("face.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/api/v1/embedding",
    json={"image_base64": image_base64}
)

result = response.json()
print(f"Embedding dim: {len(result['embedding'])}")
print(f"Processing time: {result['processing_time_ms']:.2f}ms")
```

### 模式2: 向量搜索

```python
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query_embedding": embedding,  # 512维向量
        "threshold": 0.4
    }
)
```

### 添加人脸到数据库

```python
response = requests.post(
    "http://localhost:8000/api/v1/faces/add",
    json={
        "person_id": 1,
        "person_name": "张三",
        "embedding": embedding  # 512维向量
    }
)
```

## 性能测试

服务器提供基准测试端点：

```bash
curl -X POST "http://localhost:8000/api/v1/benchmark/embedding?iterations=10" \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "..."}'
```

## 部署建议

### 本地局域网测试

1. 确保手机和电脑在同一局域网
2. 获取电脑的局域网 IP：`ifconfig` 或 `ipconfig`
3. 启动服务器：`uvicorn main:app --host 0.0.0.0 --port 8000`
4. 在 Android 客户端配置服务器地址：`http://电脑IP:8000`

### 云服务器部署

1. 准备一台云服务器（推荐 2核4G 以上）
2. 安装 Python 3.9+ 和依赖
3. 配置防火墙开放 8000 端口
4. 使用 supervisor 或 systemd 管理进程
5. 可选：配置 Nginx 反向代理和 HTTPS

## 性能指标

在典型服务器配置（4核CPU）上的参考性能：

| 操作 | 耗时 |
|------|------|
| FaceNet 嵌入生成 | ~50-100ms |
| 向量搜索 (1000条记录) | ~1-5ms |
| 人脸检测 | ~30-50ms |
| 完整流程 | ~100-200ms |

注意：网络传输延迟不包含在上述时间内。

