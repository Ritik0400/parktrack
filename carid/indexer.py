import os, json
from typing import List, Dict, Any
import numpy as np
import faiss

class CarIndex:
    """
    Cosine-similarity index using FAISS (IndexIDMap over IndexFlatIP).
    Persists to disk (index + meta mapping).
    """
    def __init__(self, root_dir: str, dim: int):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self.index_path = os.path.join(self.root_dir, "index.faiss")
        self.meta_path  = os.path.join(self.root_dir, "meta.json")
        self.dim = dim
        self.index = None
        self.meta: Dict[str, Any] = {"next_id": 1, "items": {}}  # id -> meta dict
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            base = faiss.IndexFlatIP(self.dim)  # inner product over L2-normalized vectors = cosine
            self.index = faiss.IndexIDMap2(base)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]) -> List[int]:
        assert vecs.dtype == np.float32 and vecs.ndim == 2 and vecs.shape[1] == self.dim
        ids: List[int] = []
        for m in metas:
            id_ = int(self.meta.get("next_id", 1))
            self.meta["items"][str(id_)] = m
            ids.append(id_)
            self.meta["next_id"] = id_ + 1
        self.index.add_with_ids(vecs, np.asarray(ids, dtype=np.int64))
        self._save()
        return ids

    def search(self, qvecs: np.ndarray, k: int = 3) -> List[List[Dict[str, Any]]]:
        assert qvecs.dtype == np.float32 and qvecs.ndim == 2 and qvecs.shape[1] == self.dim
        D, I = self.index.search(qvecs, k)
        out: List[List[Dict[str, Any]]] = []
        for i in range(I.shape[0]):
            row: List[Dict[str, Any]] = []
            for j in range(I.shape[1]):
                idx = int(I[i, j])
                score = float(D[i, j])
                if idx == -1:
                    continue
                meta = self.meta["items"].get(str(idx), {})
                row.append({"id": idx, "score": round(score, 4), "meta": meta})
            out.append(row)
        return out
