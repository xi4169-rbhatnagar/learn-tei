from typing import List

from numpy import ndarray
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts: List[str]) -> ndarray:
    return model.encode(texts)
