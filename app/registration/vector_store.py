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
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=obj_id,
                    vector=vector.tolist(),
                    payload=metadata or {}
                )
            ]
        )

    def search(self, vector, top_k=3):
        """Return nearest embeddings — using query_points for local Qdrant"""
        res = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=top_k
        )
        return res.points

    def next_id(self):
        """Generate the next unique ID based on the current collection."""
        try:
            # Retrieve all existing IDs in the collection
            points = self.client.scroll(
                collection_name=self.collection,
                limit=1,
                with_payload=False,
                with_vectors=False
            )

            if points and points.points:
                # Extract the maximum ID and increment
                max_id = max(point.id for point in points.points)
                return max_id + 1
            else:
                # Start from ID 1 if the collection is empty
                return 1
        except Exception as e:
            # Handle cases where the collection might be empty or inaccessible
            print(f"Error generating next ID: {e}")
            return 1

    def get_next_id(self):
        """Generate the next unique ID based on the current collection."""
        return self.next_id()
