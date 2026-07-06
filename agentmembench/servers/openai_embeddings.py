"""Serve a SentenceTransformers model through the OpenAI embeddings schema.

This small server lets every benchmark adapter use exactly the same local
embedding model without importing a framework-specific embedding wrapper.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    dimensions: int | None = None
    encoding_format: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--served-model-name", default="bge-m3")
    parser.add_argument("--device", default=os.getenv("EMBEDDING_DEVICE", "cuda"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18002)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="AgentMemBench local embedding service")
    model = SentenceTransformer(args.model, device=args.device)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": args.served_model_name,
            "dimension": model.get_sentence_embedding_dimension(),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": args.served_model_name, "object": "model"}],
        }

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        started = time.perf_counter()
        texts = [request.input] if isinstance(request.input, str) else request.input
        vectors = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        data = [
            {"object": "embedding", "index": index, "embedding": vector.tolist()}
            for index, vector in enumerate(vectors)
        ]
        return {
            "object": "list",
            "model": args.served_model_name,
            "data": data,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts),
            },
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    return app


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
