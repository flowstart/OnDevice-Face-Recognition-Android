#!/usr/bin/env python3
"""
Performance Benchmark Script for Face Recognition Offloading

This script runs comprehensive benchmarks to measure the performance
of different computation partitioning strategies.

Usage:
    python benchmark.py --server http://localhost:8000 --iterations 100

Author: Software Architecture Course Project
"""

import argparse
import time
import json
import base64
import statistics
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import requests
import numpy as np

# Try to import image processing libraries
try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not installed, some features may not work")


class BenchmarkRunner:
    """
    Benchmark runner for testing different offloading modes.
    """
    
    def __init__(self, server_url: str, output_dir: str = "benchmark_results"):
        self.server_url = server_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        
    def check_server(self) -> bool:
        """Check if server is running and accessible."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Server check failed: {e}")
            return False
    
    def generate_test_image(self, width: int = 640, height: int = 480) -> str:
        """Generate a test image and return as base64 string."""
        if not HAS_PIL:
            # Generate random bytes as fallback
            random_bytes = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8).tobytes()
            return base64.b64encode(random_bytes).decode()
        
        # Generate a random image
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def generate_test_embedding(self, dim: int = 512) -> List[float]:
        """Generate a random test embedding vector."""
        embedding = np.random.randn(dim).astype(np.float32)
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def benchmark_endpoint(
        self,
        endpoint: str,
        payload: Dict,
        iterations: int = 100,
        warmup: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark a single API endpoint.
        
        Returns statistics including:
        - Average latency
        - Min/Max latency
        - Standard deviation
        - Percentiles (50th, 95th, 99th)
        """
        url = f"{self.server_url}{endpoint}"
        
        # Warmup iterations
        print(f"  Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            try:
                requests.post(url, json=payload, timeout=30)
            except Exception:
                pass
        
        # Benchmark iterations
        latencies = []
        errors = 0
        server_times = []
        
        print(f"  Running benchmark ({iterations} iterations)...")
        for i in range(iterations):
            try:
                start = time.time()
                response = requests.post(url, json=payload, timeout=30)
                end = time.time()
                
                if response.status_code == 200:
                    latency_ms = (end - start) * 1000
                    latencies.append(latency_ms)
                    
                    # Extract server processing time if available
                    result = response.json()
                    if 'processing_time_ms' in result:
                        server_times.append(result['processing_time_ms'])
                    elif 'metrics' in result and 'total_time_ms' in result['metrics']:
                        server_times.append(result['metrics']['total_time_ms'])
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                if i < 3:  # Print first few errors
                    print(f"    Error: {e}")
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{iterations}")
        
        # Calculate statistics
        if not latencies:
            return {
                "error": "All requests failed",
                "errors": errors
            }
        
        latencies_sorted = sorted(latencies)
        
        stats = {
            "endpoint": endpoint,
            "iterations": iterations,
            "successful": len(latencies),
            "errors": errors,
            "latency_ms": {
                "avg": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "p50": latencies_sorted[len(latencies) // 2],
                "p95": latencies_sorted[int(len(latencies) * 0.95)],
                "p99": latencies_sorted[int(len(latencies) * 0.99)]
            }
        }
        
        if server_times:
            stats["server_time_ms"] = {
                "avg": statistics.mean(server_times),
                "min": min(server_times),
                "max": max(server_times)
            }
            stats["network_time_ms"] = {
                "avg": stats["latency_ms"]["avg"] - statistics.mean(server_times)
            }
        
        return stats
    
    def run_embedding_benchmark(self, iterations: int = 100) -> Dict:
        """Benchmark Mode 1: Embedding generation only."""
        print("\n[Mode 1] Benchmarking embedding generation...")
        
        test_image = self.generate_test_image(160, 160)  # FaceNet input size
        payload = {"image_base64": test_image}
        
        return self.benchmark_endpoint(
            "/api/v1/embedding",
            payload,
            iterations
        )
    
    def run_search_benchmark(self, iterations: int = 100) -> Dict:
        """Benchmark Mode 2: Vector search only."""
        print("\n[Mode 2] Benchmarking vector search...")
        
        test_embedding = self.generate_test_embedding(512)
        payload = {
            "query_embedding": test_embedding,
            "threshold": 0.4
        }
        
        return self.benchmark_endpoint(
            "/api/v1/search",
            payload,
            iterations
        )
    
    def run_embedding_and_search_benchmark(self, iterations: int = 100) -> Dict:
        """Benchmark Mode 3: Embedding + search combined."""
        print("\n[Mode 3] Benchmarking embedding + search...")
        
        test_image = self.generate_test_image(160, 160)
        payload = {"image_base64": test_image}
        
        return self.benchmark_endpoint(
            "/api/v1/embedding_and_search",
            payload,
            iterations
        )
    
    def run_full_pipeline_benchmark(self, iterations: int = 100) -> Dict:
        """Benchmark Mode 4: Full pipeline."""
        print("\n[Mode 4] Benchmarking full pipeline...")
        
        test_image = self.generate_test_image(640, 480)  # Full frame size
        payload = {"image_base64": test_image}
        
        return self.benchmark_endpoint(
            "/api/v1/full_pipeline",
            payload,
            iterations
        )
    
    def run_all_benchmarks(self, iterations: int = 100) -> Dict[str, Any]:
        """Run all benchmarks and compile results."""
        print("=" * 60)
        print("Face Recognition Offloading Benchmark")
        print(f"Server: {self.server_url}")
        print(f"Iterations per test: {iterations}")
        print("=" * 60)
        
        results = {
            "metadata": {
                "server_url": self.server_url,
                "iterations": iterations,
                "timestamp": datetime.now().isoformat(),
            },
            "benchmarks": {}
        }
        
        # Check server
        if not self.check_server():
            print("ERROR: Server is not accessible!")
            return results
        
        # Run benchmarks
        results["benchmarks"]["embedding"] = self.run_embedding_benchmark(iterations)
        results["benchmarks"]["search"] = self.run_search_benchmark(iterations)
        results["benchmarks"]["embedding_and_search"] = self.run_embedding_and_search_benchmark(iterations)
        results["benchmarks"]["full_pipeline"] = self.run_full_pipeline_benchmark(iterations)
        
        self.results.append(results)
        return results
    
    def export_results(self, results: Dict[str, Any]) -> str:
        """Export results to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export JSON
        json_path = self.output_dir / f"benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Export CSV summary
        csv_path = self.output_dir / f"benchmark_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "mode", "iterations", "successful", "errors",
                "avg_latency_ms", "min_latency_ms", "max_latency_ms",
                "p50_ms", "p95_ms", "p99_ms", "std_ms",
                "avg_server_ms", "avg_network_ms"
            ])
            
            # Data rows
            for mode, data in results.get("benchmarks", {}).items():
                if "error" in data:
                    continue
                
                latency = data.get("latency_ms", {})
                server = data.get("server_time_ms", {})
                network = data.get("network_time_ms", {})
                
                writer.writerow([
                    mode,
                    data.get("iterations", 0),
                    data.get("successful", 0),
                    data.get("errors", 0),
                    f"{latency.get('avg', 0):.2f}",
                    f"{latency.get('min', 0):.2f}",
                    f"{latency.get('max', 0):.2f}",
                    f"{latency.get('p50', 0):.2f}",
                    f"{latency.get('p95', 0):.2f}",
                    f"{latency.get('p99', 0):.2f}",
                    f"{latency.get('std', 0):.2f}",
                    f"{server.get('avg', 0):.2f}",
                    f"{network.get('avg', 0):.2f}"
                ])
        
        print(f"\nResults exported to:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")
        
        return str(json_path)
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        benchmarks = results.get("benchmarks", {})
        
        # Table header
        print(f"\n{'Mode':<25} {'Avg (ms)':<12} {'P95 (ms)':<12} {'Server (ms)':<12} {'Network (ms)':<12}")
        print("-" * 75)
        
        for mode, data in benchmarks.items():
            if "error" in data:
                print(f"{mode:<25} ERROR: {data['error']}")
                continue
            
            latency = data.get("latency_ms", {})
            server = data.get("server_time_ms", {})
            network = data.get("network_time_ms", {})
            
            print(f"{mode:<25} "
                  f"{latency.get('avg', 0):>10.2f}  "
                  f"{latency.get('p95', 0):>10.2f}  "
                  f"{server.get('avg', 0):>10.2f}  "
                  f"{network.get('avg', 0):>10.2f}")
        
        print("-" * 75)
        print("\nNotes:")
        print("- 'embedding': Mode 1 - Offload FaceNet embedding generation")
        print("- 'search': Mode 2 - Offload vector search")
        print("- 'embedding_and_search': Mode 3 - Offload both embedding and search")
        print("- 'full_pipeline': Mode 4 - Offload entire recognition pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark face recognition offloading performance"
    )
    parser.add_argument(
        "--server", "-s",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of iterations per test (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.server, args.output)
    results = runner.run_all_benchmarks(args.iterations)
    runner.print_summary(results)
    runner.export_results(results)


if __name__ == "__main__":
    main()

