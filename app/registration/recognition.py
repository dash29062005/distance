import numpy as np

class Recognizer:
    """
    Identity resolver using Qdrant only.
    """

    def __init__(self, vector_store, embedder,
                 similarity_threshold=0.82):

        self.store = vector_store       # Qdrant wrapper
        self.embedder = embedder
        self.sim_threshold = similarity_threshold

    # -----------------------------------------------------------
    def _best_match(self, embedding):
        """Query Qdrant → return (id, score)."""

        results = self.store.search(embedding, top_k=3)
        if not results:
            return None, 0.0

        # Qdrant returns: result.id, result.score, result.payload
        best = results[0]
        return best.id, float(best.score)

    # -----------------------------------------------------------
    def _new_id(self):
        """Get next available ID from Qdrant."""
        return self.store.get_next_id()

    # -----------------------------------------------------------
    def register_new(self, emb, bbox):
        """Create new embedding entry in Qdrant."""

        obj_id = self._new_id()

        metadata = {
            "bbox": [float(v) for v in bbox]
        }

        self.store.add(obj_id, emb, metadata)
        return obj_id

    # -----------------------------------------------------------
    def identify(self, crop_rgb, bbox):
        """
        Returns:
          {
            "id": ...,
            "is_new": True/False,
            "embedding": emb,
            "score": ...
          }
        """

        emb = self.embedder.get_embedding(crop_rgb)
        if emb is None:
            return None

        # search existing
        best_id, score = self._best_match(emb)

        # strong match → use it
        if best_id is not None and score >= self.sim_threshold:
            return {
                "id": best_id,
                "is_new": False,
                "embedding": emb,
                "score": score
            }

        # else → new identity
        new_id = self.register_new(emb, bbox)

        return {
            "id": new_id,
            "is_new": True,
            "embedding": emb,
            "score": score
        }
