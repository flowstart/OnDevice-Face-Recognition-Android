"""
Cloud Server for Face Recognition Computation Offloading
=========================================================

This server provides REST APIs for offloading compute-intensive tasks
from the Android client to the cloud.

Supported Offloading Modes:
1. Face Embedding Generation (FaceNet)
2. Vector Search (Nearest Neighbor Search)
3. Face Embedding + Vector Search Combined
4. Full Pipeline (Face Detection + Embedding + Search)

Author: Software Architecture Course Project
"""

import io
import time
import base64
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
face_detector = None
facenet_model = None
vector_store = None


class EmbeddingRequest(BaseModel):
    """Request model for face embedding generation"""
    image_base64: str  # Base64 encoded cropped face image


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
    image_base64: str  # Base64 encoded full frame image


class FullPipelineResponse(BaseModel):
    """Response model for full pipeline"""
    faces: List[dict]
    metrics: dict


class FaceNetModel:
    """
    FaceNet model wrapper for generating face embeddings.
    Uses TensorFlow Lite for inference.
    """
    
    def __init__(self, model_path: str = "models/facenet_512.tflite"):
        import tensorflow as tf
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_size = 160
        self.embedding_dim = 512
        
        logger.info(f"FaceNet model loaded: input_shape={self.input_details[0]['shape']}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for FaceNet model"""
        # Resize to 160x160
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert to float32
        image = image.astype(np.float32)
        
        # Standardize: (x - mean) / std
        mean = np.mean(image)
        std = np.std(image)
        std = max(std, 1.0 / np.sqrt(image.size))
        image = (image - mean) / std
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Generate face embedding from cropped face image"""
        # Preprocess
        input_data = self.preprocess(face_image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return embedding[0]


class FaceDetector:
    """
    Face detector using MediaPipe BlazeFace.
    """
    
    def __init__(self, model_path: str = "models/blaze_face_short_range.tflite"):
        import mediapipe as mp
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        logger.info("MediaPipe Face Detector loaded")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in the image.
        Returns list of bounding boxes.
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure bounds are valid
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 0 and height > 0:
                    faces.append({
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'confidence': detection.score[0]
                    })
        
        return faces


class VectorStore:
    """
    Simple in-memory vector store using cosine similarity.
    For production, use FAISS or similar.
    """
    
    def __init__(self):
        self.embeddings = []  # List of (person_id, person_name, embedding)
        self.embedding_dim = 512
    
    def add_embedding(self, person_id: int, person_name: str, embedding: np.ndarray):
        """Add a face embedding to the store"""
        self.embeddings.append((person_id, person_name, np.array(embedding)))
        logger.info(f"Added embedding for {person_name}, total: {len(self.embeddings)}")
    
    def search(self, query_embedding: np.ndarray, threshold: float = 0.4) -> Optional[dict]:
        """
        Search for the nearest embedding.
        Returns None if no match above threshold.
        """
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
        return dot_product / (norm_a * norm_b)
    
    def clear(self):
        """Clear all embeddings"""
        self.embeddings = []
    
    def get_count(self) -> int:
        """Get number of stored embeddings"""
        return len(self.embeddings)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array image"""
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global face_detector, facenet_model, vector_store
    
    logger.info("Initializing models...")
    
    try:
        # Initialize models (lazy loading - will be initialized on first use if files exist)
        vector_store = VectorStore()
        
        # Try to load FaceNet model
        try:
            facenet_model = FaceNetModel()
        except Exception as e:
            logger.warning(f"FaceNet model not loaded: {e}")
            facenet_model = None
        
        # Try to load Face Detector
        try:
            face_detector = FaceDetector()
        except Exception as e:
            logger.warning(f"Face Detector not loaded: {e}")
            face_detector = None
        
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
    
    yield
    
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Face Recognition Offloading Server",
    description="Cloud server for offloading face recognition computations",
    version="1.0.0",
    lifespan=lifespan
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
        "models": {
            "face_detector": face_detector is not None,
            "facenet": facenet_model is not None,
            "vector_store_count": vector_store.get_count() if vector_store else 0
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


# ============================================================
# Offloading Mode 1: Face Embedding Generation Only
# ============================================================

@app.post("/api/v1/embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate face embedding from a cropped face image.
    
    Offloading Mode: EMBEDDING_ONLY
    - Client performs face detection locally
    - Client sends cropped face image to server
    - Server generates embedding and returns it
    - Client performs vector search locally
    """
    if facenet_model is None:
        raise HTTPException(status_code=503, detail="FaceNet model not available")
    
    start_time = time.time()
    
    try:
        # Decode image
        face_image = decode_base64_image(request.image_base64)
        
        # Generate embedding
        embedding = facenet_model.get_embedding(face_image)
        
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
    - Client performs face detection locally
    - Client generates embedding locally
    - Client sends embedding to server
    - Server performs vector search and returns result
    """
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available")
    
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
    - Client performs face detection locally
    - Client sends cropped face image to server
    - Server generates embedding AND performs vector search
    - Server returns recognition result
    """
    if facenet_model is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Models not available")
    
    start_time = time.time()
    embedding_start = time.time()
    
    try:
        # Decode image
        face_image = decode_base64_image(request.image_base64)
        
        # Generate embedding
        embedding = facenet_model.get_embedding(face_image)
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
    - Client sends full camera frame to server
    - Server performs face detection, embedding, and search
    - Server returns all results
    """
    if face_detector is None or facenet_model is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Models not available")
    
    total_start = time.time()
    
    try:
        # Decode image
        frame = decode_base64_image(request.image_base64)
        
        # Face detection
        detection_start = time.time()
        detected_faces = face_detector.detect_faces(frame)
        detection_time = (time.time() - detection_start) * 1000
        
        faces = []
        total_embedding_time = 0
        total_search_time = 0
        
        for face_bbox in detected_faces:
            # Crop face
            x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']
            cropped_face = frame[y:y+h, x:x+w]
            
            # Generate embedding
            embedding_start = time.time()
            embedding = facenet_model.get_embedding(cropped_face)
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
        
        return FullPipelineResponse(
            faces=faces,
            metrics={
                "face_detection_ms": detection_time,
                "embedding_ms": total_embedding_time / max(len(detected_faces), 1),
                "search_ms": total_search_time / max(len(detected_faces), 1),
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
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available")
    
    vector_store.add_embedding(
        request.person_id,
        request.person_name,
        np.array(request.embedding)
    )
    
    return {"status": "success", "total_faces": vector_store.get_count()}


@app.post("/api/v1/faces/add_with_image")
async def add_face_with_image(
    person_id: int,
    person_name: str,
    image: UploadFile = File(...)
):
    """Add a face by uploading an image"""
    if facenet_model is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Models not available")
    
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Generate embedding
        embedding = facenet_model.get_embedding(img)
        
        # Store
        vector_store.add_embedding(person_id, person_name, embedding)
        
        return {"status": "success", "total_faces": vector_store.get_count()}
    except Exception as e:
        logger.error(f"Add face failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/faces/clear")
async def clear_faces():
    """Clear all faces from the database"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available")
    
    vector_store.clear()
    return {"status": "success", "total_faces": 0}


@app.get("/api/v1/faces/count")
async def get_face_count():
    """Get the number of stored faces"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available")
    
    return {"count": vector_store.get_count()}


# ============================================================
# Benchmark/Testing Endpoints
# ============================================================

@app.post("/api/v1/benchmark/embedding")
async def benchmark_embedding(request: EmbeddingRequest, iterations: int = 10):
    """Benchmark embedding generation"""
    if facenet_model is None:
        raise HTTPException(status_code=503, detail="FaceNet model not available")
    
    face_image = decode_base64_image(request.image_base64)
    
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = facenet_model.get_embedding(face_image)
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

