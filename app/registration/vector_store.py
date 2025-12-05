from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np


class VectorStore:
    def __init__(self, collection="objects", dim=1024):
        self.client = QdrantClient(path="data/qdrant")

        # ensure collection exists
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

        self.collection = collection

    def add(self, obj_id, vector, metadata=None):
        """Insert vector into Qdrant"""
        v = vector.tolist()
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=obj_id,
                    vector=v,
                    payload=metadata or {}
                )
            ]
        )

    def search(self, vector, top_k=3):
        """Return nearest embeddings"""
        result = self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=top_k
        )
        return result
