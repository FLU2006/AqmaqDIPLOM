# import json
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np


# @dataclass
# class PersonProfile:
#     person_id: str          # например "Known_Nazar" или "Unknown_0007"
#     name: str               # человекочитаемое имя (можно = person_id)
#     embeddings: List[List[float]]  # список embeddings
#     mean_embedding: List[float]    # средний embedding
#     images: List[str]             # пути к сохранённым фото


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     a = a / (np.linalg.norm(a) + 1e-9)
#     b = b / (np.linalg.norm(b) + 1e-9)
#     return float(np.dot(a, b))


# class FaceDB:
#     def __init__(self, root_dir: Path):
#         self.root_dir = Path(root_dir)
#         self.db_path = self.root_dir / "faces_db.json"
#         self.root_dir.mkdir(parents=True, exist_ok=True)
#         self.profiles: Dict[str, PersonProfile] = {}
#         self._load()

#     def _load(self):
#         if self.db_path.exists():
#             data = json.loads(self.db_path.read_text(encoding="utf-8"))
#             for p in data.get("profiles", []):
#                 prof = PersonProfile(**p)
#                 self.profiles[prof.person_id] = prof

#     def _save(self):
#         data = {"profiles": [asdict(p) for p in self.profiles.values()]}
#         self.db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

#     def _next_unknown_id(self) -> str:
#         n = 0
#         while True:
#             pid = f"Unknown_{n:04d}"
#             if pid not in self.profiles:
#                 return pid
#             n += 1

#     def add_new_person(self, embedding: np.ndarray, image_path: str, name: Optional[str] = None) -> str:
#         person_id = self._next_unknown_id()
#         name = name or person_id

#         emb_list = [embedding.tolist()]
#         mean = embedding.tolist()

#         prof = PersonProfile(
#             person_id=person_id,
#             name=name,
#             embeddings=emb_list,
#             mean_embedding=mean,
#             images=[image_path],
#         )
#         self.profiles[person_id] = prof
#         self._save()
#         return person_id

#     def add_to_person(self, person_id: str, embedding: np.ndarray, image_path: str):
#         prof = self.profiles[person_id]
#         prof.embeddings.append(embedding.tolist())
#         prof.images.append(image_path)

#         embs = np.array(prof.embeddings, dtype=np.float32)
#         mean = embs.mean(axis=0)
#         prof.mean_embedding = mean.tolist()

#         self._save()

#     def match(self, embedding: np.ndarray, threshold: float) -> Tuple[Optional[str], float]:
#         best_id = None
#         best_score = -1.0

#         for pid, prof in self.profiles.items():
#             mean = np.array(prof.mean_embedding, dtype=np.float32)
#             score = cosine_similarity(embedding, mean)
#             if score > best_score:
#                 best_score = score
#                 best_id = pid

#         if best_id is None or best_score < threshold:
#             return None, best_score
#         return best_id, best_score