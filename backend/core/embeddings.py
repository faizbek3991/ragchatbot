from typing import List
from fastembed import TextEmbedding

# Local ONNX model, no API key needed. 384-dimensional vectors.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    # Loaded lazily: the first call downloads the model (~130 MB) and caches it.
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed document chunks. Blocking, so call via run_in_threadpool."""
    return [vec.tolist() for vec in _get_model().embed(texts)]


def embed_query(query: str) -> List[float]:
    """Embed a search query (bge models use a query-specific prefix)."""
    return next(iter(_get_model().query_embed(query))).tolist()
