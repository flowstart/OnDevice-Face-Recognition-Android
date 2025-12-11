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
        
        # Generate embedding
        embedding = simulate_facenet_embedding(image)
        embedding_time = (time.time() - embedding_start) * 1000
        
        # Search
        search_start = time.time()
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
            
            # Generate embedding
            embedding_start = time.time()
            embedding = simulate_facenet_embedding(cropped)
            total_embedding_time += (time.time() - embedding_start) * 1000
            
            # Search
            search_start = time.time()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
