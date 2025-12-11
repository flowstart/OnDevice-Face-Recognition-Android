"""
Cloud Server for Face Recognition Computation Offloading
=========================================================

简化版服务器 - 用于测试计算划分性能
不需要 TensorFlow/MediaPipe，使用模拟计算来测量网络延迟

实际部署时可以添加真正的模型推理

Author: Software Architecture Course Project
"""

import io
import time
import base64
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    """Request model for face embedding generation"""
    image_base64: str


class EmbeddingResponse(BaseModel):
    """Response model for face embedding"""
    embedding: List[float]
    processing_time_ms: float


class VectorSearchRequest(BaseModel):
    """Request model for vector search"""
    query_embedding: List[float]
    threshold: float = 0.4


class VectorSearchResponse(BaseModel):
    """Response model for vector search"""
    person_name: str
    similarity: float
    processing_time_ms: float


class AddFaceRequest(BaseModel):
    """Request model for adding a face to the database"""
    person_id: int
    person_name: str
    embedding: List[float]


class FullPipelineRequest(BaseModel):
    """Request model for full pipeline processing"""
    image_base64: str


class FullPipelineResponse(BaseModel):
    """Response model for full pipeline"""
    faces: List[dict]
    metrics: dict


class VectorStore:
    """
    Simple in-memory vector store using cosine similarity.
    """
    
    def __init__(self):
        self.embeddings = []  # List of (person_id, person_name, embedding)
        self.embedding_dim = 512
    
    def add_embedding(self, person_id: int, person_name: str, embedding: np.ndarray):
        """Add a face embedding to the store"""
        self.embeddings.append((person_id, person_name, np.array(embedding)))
        logger.info(f"Added embedding for {person_name}, total: {len(self.embeddings)}")
    
    def search(self, query_embedding: np.ndarray, threshold: float = 0.4) -> Optional[dict]:
        """Search for the nearest embedding."""
        if not self.embeddings:
            return None
        
        query = np.array(query_embedding)
        best_match = None
        best_similarity = -1
        
        for person_id, person_name, embedding in self.embeddings:
            similarity = self._cosine_similarity(query, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'person_id': person_id,
                    'person_name': person_name,
                    'similarity': float(similarity)
                }
        
        if best_match and best_match['similarity'] > threshold:
            return best_match
        
        return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def clear(self):
        """Clear all embeddings"""
        self.embeddings = []
    
    def get_count(self) -> int:
        """Get number of stored embeddings"""
        return len(self.embeddings)


# Global vector store
vector_store = VectorStore()


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


def simulate_facenet_embedding(image: Image.Image) -> np.ndarray:
    """
    Simulate FaceNet embedding generation.
    
    In production, this would run actual TFLite inference.
    For benchmarking, we simulate the computation time.
    """
    # Simulate processing time (80-120ms like real FaceNet)
    time.sleep(0.08 + np.random.random() * 0.04)
    
    # Generate a deterministic embedding based on image content
    # This ensures same image gives same embedding
    img_array = np.array(image.resize((160, 160)))
    np.random.seed(int(img_array.sum()) % 2**31)
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding


def simulate_face_detection(image: Image.Image) -> List[dict]:
    """
    Simulate face detection.
    
    Returns a single face in the center of the image.
    """
    # Simulate processing time (20-40ms)
    time.sleep(0.02 + np.random.random() * 0.02)
    
    w, h = image.size
    # Return a face in the center
    face_w = min(w, h) // 2
    face_h = face_w
    
    return [{
        'x': (w - face_w) // 2,
        'y': (h - face_h) // 2,
        'width': face_w,
        'height': face_h,
        'confidence': 0.95
    }]


def simulate_vector_search_delay():
    """
    Simulate vector search processing time.
    Real vector search with FAISS or similar would take 5-15ms.
    """
    time.sleep(0.005 + np.random.random() * 0.01)  # 5-15ms


# Create FastAPI app
app = FastAPI(
    title="Face Recognition Offloading Server",
    description="云端人脸识别计算卸载服务器 (简化版)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Face Recognition Offloading Server",
        "mode": "simulation",
        "vector_store_count": vector_store.get_count()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "mode": "simulation"
    }


# ============================================================
# Offloading Mode 1: Face Embedding Generation Only
# ============================================================

@app.post("/api/v1/embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate face embedding from a cropped face image.
    
    Offloading Mode: EMBEDDING_ONLY
    """
    start_time = time.time()
    
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Generate embedding (simulated)
        embedding = simulate_facenet_embedding(image)
        
        processing_time = (time.time() - start_time) * 1000
        
        return EmbeddingResponse(
            embedding=embedding.tolist(),
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Offloading Mode 2: Vector Search Only
# ============================================================

@app.post("/api/v1/search", response_model=VectorSearchResponse)
async def search_vector(request: VectorSearchRequest):
    """
    Search for the nearest face in the vector store.
    
    Offloading Mode: SEARCH_ONLY
    """
    start_time = time.time()
    
    try:
        query_embedding = np.array(request.query_embedding)
        
        # Simulate vector search processing time (5-15ms)
        simulate_vector_search_delay()
        
        result = vector_store.search(query_embedding, request.threshold)
        
        processing_time = (time.time() - start_time) * 1000
        
        if result:
            return VectorSearchResponse(
                person_name=result['person_name'],
                similarity=result['similarity'],
                processing_time_ms=processing_time
            )
        else:
            return VectorSearchResponse(
                person_name="Not recognized",
                similarity=0.0,
                processing_time_ms=processing_time
            )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Offloading Mode 3: Embedding + Search Combined
# ============================================================

@app.post("/api/v1/embedding_and_search")
async def embedding_and_search(request: EmbeddingRequest):
    """
    Generate embedding and perform vector search.
    
    Offloading Mode: EMBEDDING_AND_SEARCH
    """
    start_time = time.time()
    embedding_start = time.time()
    
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Generate embedding (80-120ms simulated)
        embedding = simulate_facenet_embedding(image)
        embedding_time = (time.time() - embedding_start) * 1000
        
        # Search (5-15ms simulated)
        search_start = time.time()
        simulate_vector_search_delay()
        result = vector_store.search(embedding, threshold=0.4)
        search_time = (time.time() - search_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "person_name": result['person_name'] if result else "Not recognized",
            "similarity": result['similarity'] if result else 0.0,
            "embedding": embedding.tolist(),
            "metrics": {
                "embedding_time_ms": embedding_time,
                "search_time_ms": search_time,
                "total_time_ms": total_time
            }
        }
    except Exception as e:
        logger.error(f"Embedding and search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Offloading Mode 4: Full Pipeline
# ============================================================

@app.post("/api/v1/full_pipeline", response_model=FullPipelineResponse)
async def full_pipeline(request: FullPipelineRequest):
    """
    Process full pipeline: detection + embedding + search.
    
    Offloading Mode: FULL_OFFLOAD
    """
    total_start = time.time()
    
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Face detection (simulated)
        detection_start = time.time()
        detected_faces = simulate_face_detection(image)
        detection_time = (time.time() - detection_start) * 1000
        
        faces = []
        total_embedding_time = 0
        total_search_time = 0
        
        for face_bbox in detected_faces:
            # Crop face
            x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']
            cropped = image.crop((x, y, x + w, y + h))
            
            # Generate embedding (80-120ms simulated)
            embedding_start = time.time()
            embedding = simulate_facenet_embedding(cropped)
            total_embedding_time += (time.time() - embedding_start) * 1000
            
            # Search (5-15ms simulated)
            search_start = time.time()
            simulate_vector_search_delay()
            result = vector_store.search(embedding, threshold=0.4)
            total_search_time += (time.time() - search_start) * 1000
            
            faces.append({
                "bbox": face_bbox,
                "person_name": result['person_name'] if result else "Not recognized",
                "similarity": result['similarity'] if result else 0.0
            })
        
        total_time = (time.time() - total_start) * 1000
        num_faces = max(len(detected_faces), 1)
        
        return FullPipelineResponse(
            faces=faces,
            metrics={
                "face_detection_ms": detection_time,
                "embedding_ms": total_embedding_time / num_faces,
                "search_ms": total_search_time / num_faces,
                "total_ms": total_time,
                "num_faces": len(detected_faces)
            }
        )
    except Exception as e:
        logger.error(f"Full pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Database Management APIs
# ============================================================

@app.post("/api/v1/faces/add")
async def add_face(request: AddFaceRequest):
    """Add a face embedding to the database"""
    vector_store.add_embedding(
        request.person_id,
        request.person_name,
        np.array(request.embedding)
    )
    
    return {"status": "success", "total_faces": vector_store.get_count()}


@app.delete("/api/v1/faces/clear")
async def clear_faces():
    """Clear all faces from the database"""
    vector_store.clear()
    return {"status": "success", "total_faces": 0}


@app.get("/api/v1/faces/count")
async def get_face_count():
    """Get the number of stored faces"""
    return {"count": vector_store.get_count()}


# ============================================================
# Benchmark Endpoint
# ============================================================

@app.post("/api/v1/benchmark/embedding")
async def benchmark_embedding(request: EmbeddingRequest, iterations: int = 10):
    """Benchmark embedding generation"""
    image = decode_base64_image(request.image_base64)
    
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = simulate_facenet_embedding(image)
        times.append((time.time() - start) * 1000)
    
    return {
        "iterations": iterations,
        "times_ms": times,
        "avg_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times)
    }


# ============================================================
# Performance Report APIs
# ============================================================

import uuid
from datetime import datetime
from collections import defaultdict

# Storage
performance_data_store = defaultdict(dict)
reports_store = {}

class PerformanceMetrics(BaseModel):
    face_detection_ms: int
    embedding_ms: int
    vector_search_ms: int
    spoof_detection_ms: int
    network_ms: int
    server_ms: int
    total_ms: int
    data_transferred_bytes: int = 0

class PerformanceDataUpload(BaseModel):
    session_id: str
    mode: str
    mode_name: str
    metrics: PerformanceMetrics
    device_info: str
    network_type: str
    timestamp: int

class GenerateReportRequest(BaseModel):
    session_id: str

@app.post("/api/v1/report/upload")
async def upload_performance_data(data: PerformanceDataUpload):
    session_id = data.session_id
    mode = data.mode
    
    # 如果该模式已有数据，累加计算平均值
    if mode in performance_data_store[session_id]:
        existing = performance_data_store[session_id][mode]
        count = existing.get("upload_count", 1) + 1
        old_metrics = existing["metrics"]
        new_metrics = data.metrics.dict()
        
        # 计算累加平均值
        avg_metrics = {}
        for key in new_metrics:
            old_val = old_metrics.get(key, 0)
            new_val = new_metrics[key]
            # 累加平均: new_avg = old_avg + (new_val - old_avg) / count
            avg_metrics[key] = old_val + (new_val - old_val) / count
        
        performance_data_store[session_id][mode] = {
            "mode": mode, "mode_name": data.mode_name,
            "metrics": avg_metrics, "device_info": data.device_info,
            "network_type": data.network_type, "timestamp": data.timestamp,
            "uploaded_at": datetime.now().isoformat(),
            "upload_count": count
        }
    else:
        # 第一次上传该模式
        performance_data_store[session_id][mode] = {
            "mode": mode, "mode_name": data.mode_name,
            "metrics": data.metrics.dict(), "device_info": data.device_info,
            "network_type": data.network_type, "timestamp": data.timestamp,
            "uploaded_at": datetime.now().isoformat(),
            "upload_count": 1
        }
    
    count = performance_data_store[session_id][mode]["upload_count"]
    return {
        "status": "success", 
        "uploaded_modes": list(performance_data_store[session_id].keys()),
        "mode": mode,
        "upload_count": count
    }

@app.get("/api/v1/report/status")
async def get_upload_status(session_id: str):
    all_modes = ["LOCAL_ONLY", "EMBEDDING_OFFLOAD", "SEARCH_OFFLOAD", 
                 "EMBEDDING_AND_SEARCH_OFFLOAD", "FULL_OFFLOAD"]
    uploaded = performance_data_store.get(session_id, {})
    return {"session_id": session_id, "modes": {m: m in uploaded for m in all_modes}}

@app.post("/api/v1/report/generate")
async def generate_report(request: GenerateReportRequest):
    session_data = performance_data_store.get(request.session_id, {})
    if not session_data:
        raise HTTPException(status_code=404, detail="No data")
    report_id = str(uuid.uuid4())[:8]
    first = list(session_data.values())[0]
    modes_data = [{"mode": m, **d["metrics"], "mode_name": d["mode_name"], 
                   "upload_count": d.get("upload_count", 1)} 
                  for m, d in session_data.items()]
    # 按 Mode 0-4 顺序排序
    mode_order = ["LOCAL_ONLY", "EMBEDDING_OFFLOAD", "SEARCH_OFFLOAD", 
                  "EMBEDDING_AND_SEARCH_OFFLOAD", "FULL_OFFLOAD"]
    modes_data.sort(key=lambda x: mode_order.index(x["mode"]) if x["mode"] in mode_order else 99)
    reports_store[report_id] = {
        "report_id": report_id, "session_id": request.session_id,
        "created_at": datetime.now().isoformat(), "device_info": first["device_info"],
        "modes": modes_data, "total_modes_tested": len(modes_data)
    }
    return {"status": "success", "report_id": report_id}

@app.get("/api/v1/report/list")
async def list_reports():
    reports = [{"report_id": r["report_id"], "created_at": r["created_at"], 
                "device_info": r["device_info"]} for r in reports_store.values()]
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    return {"reports": reports, "total": len(reports)}

@app.get("/api/v1/report/{report_id}")
async def get_report(report_id: str):
    if report_id not in reports_store:
        raise HTTPException(status_code=404, detail="Report not found")
    return reports_store[report_id]

# ============================================================
# HTML Report Pages
# ============================================================

@app.get("/report", response_class=HTMLResponse)
async def report_list_page():
    reports = list(reports_store.values())
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    rows = "".join([f'<tr><td>{r["report_id"]}</td><td>{r["device_info"]}</td>'
                    f'<td>{r["created_at"][:19]}</td><td>{r["total_modes_tested"]}</td>'
                    f'<td><a href="/report/{r["report_id"]}">查看</a></td></tr>' for r in reports])
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>性能报告列表</title>
<style>body{{font-family:Arial;padding:20px;background:#f5f5f5}}
table{{width:100%;border-collapse:collapse;background:white}}
th,td{{border:1px solid #ddd;padding:12px;text-align:left}}
th{{background:#4CAF50;color:white}}a{{color:#4CAF50}}</style></head>
<body><h1>性能测试报告列表</h1>
<table><tr><th>报告ID</th><th>设备</th><th>创建时间</th><th>模式数</th><th>操作</th></tr>
{rows if rows else "<tr><td colspan='5'>暂无报告</td></tr>"}</table></body></html>'''

@app.get("/report/{report_id}", response_class=HTMLResponse)
async def report_detail_page(report_id: str):
    if report_id not in reports_store:
        return HTMLResponse("<h1>报告不存在</h1>", status_code=404)
    r = reports_store[report_id]
    rows = "".join([f'<tr><td>{m["mode_name"]}</td><td>{int(m["face_detection_ms"])}</td>'
                    f'<td>{int(m["embedding_ms"])}</td><td>{int(m["vector_search_ms"])}</td>'
                    f'<td>{int(m["network_ms"])}</td><td>{int(m["server_ms"])}</td>'
                    f'<td><b>{int(m["total_ms"])}</b></td>'
                    f'<td>{m.get("upload_count", 1)}次</td></tr>' for m in r["modes"]])
    chart_labels = str([m["mode_name"] for m in r["modes"]])
    chart_data = str([m["total_ms"] for m in r["modes"]])
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>性能报告 {report_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>body{{font-family:Arial;padding:20px;background:#f5f5f5;max-width:1000px;margin:auto}}
.card{{background:white;padding:20px;border-radius:8px;margin:15px 0;box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:10px}}
th{{background:#4CAF50;color:white}}canvas{{max-height:300px}}</style></head>
<body><a href="/report">← 返回列表</a>
<div class="card"><h1>性能测试报告</h1>
<p><b>报告ID:</b> {report_id}</p><p><b>设备:</b> {r["device_info"]}</p>
<p><b>测试时间:</b> {r["created_at"][:19]}</p></div>
<div class="card"><h2>性能对比</h2>
<table><tr><th>模式</th><th>检测(ms)</th><th>嵌入(ms)</th><th>搜索(ms)</th>
<th>网络(ms)</th><th>服务器(ms)</th><th>总延迟(ms)</th><th>采样次数</th></tr>{rows}</table></div>
<div class="card"><h2>延迟对比图</h2><canvas id="chart"></canvas></div>
<script>new Chart(document.getElementById("chart"),{{type:"bar",
data:{{labels:{chart_labels},datasets:[{{label:"总延迟(ms)",data:{chart_data},
backgroundColor:"#4CAF50"}}]}},options:{{responsive:true}}}})</script></body></html>'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
