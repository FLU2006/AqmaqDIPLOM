import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class PersonProfile:
    person_id: str
    display_name: str
    mean_embedding: List[float]
    samples: int
    images: List[str]

class CollectionsDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, PersonProfile] = {}
        self._load()

    def _load(self):
        """Загрузка базы данных в память при старте."""
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text(encoding="utf-8"))
                for p in data.get("profiles", []):
                    prof = PersonProfile(**p)
                    self.profiles[prof.person_id] = prof
            except Exception as e:
                print(f"[ERROR] Failed to load DB: {e}")

    def _save(self):
        """Атомарное сохранение базы (защита от повреждения файла при сбоях)."""
        data = {"profiles": [asdict(p) for p in self.profiles.values()]}
        temp_path = self.db_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.db_path)

    def _next_unknown_id(self) -> str:
        i = 0
        while True:
            pid = f"Unknown_{i:04d}"
            if pid not in self.profiles:
                return pid
            i += 1

    def create_person(self, person_id: str, emb: np.ndarray, image_path: str, display_name: Optional[str] = None):
        """Создание нового профиля (Известного или Неизвестного)."""
        if person_id in self.profiles:
            self.add_sample(person_id, emb, image_path)
            return
            
        prof = PersonProfile(
            person_id=person_id,
            display_name=display_name or person_id,
            mean_embedding=emb.tolist(),
            samples=1,
            images=[image_path],
        )
        self.profiles[person_id] = prof
        self._save()

    def create_unknown(self, emb: np.ndarray, image_path: str) -> str:
        """Регистрация нового неопознанного лица."""
        pid = self._next_unknown_id()
        self.create_person(pid, emb, image_path)
        return pid

    def delete_person(self, person_id: str) -> bool:
        """Удаление профиля из базы."""
        if person_id in self.profiles:
            del self.profiles[person_id]
            self._save()
            return True
        return False

    def add_sample(self, person_id: str, emb: np.ndarray, image_path: str):
        """Обновление усредненного вектора лица (обучение на лету)."""
        if person_id not in self.profiles:
            return
        
        prof = self.profiles[person_id]
        old_mean = np.array(prof.mean_embedding, dtype=np.float32)
        n = prof.samples

        # Формула скользящего среднего для вектора лица
        new_mean = (old_mean * n + emb) / (n + 1)
        new_mean = new_mean / (np.linalg.norm(new_mean) + 1e-9)

        prof.mean_embedding = new_mean.tolist()
        prof.samples += 1
        if image_path not in prof.images:
            prof.images.append(image_path)
        self._save()

    def match(self, emb: np.ndarray, threshold: float) -> Tuple[Optional[str], float]:
        """Поиск совпадений с учетом порога уверенности."""
        best_id, best_score = self.best_match_no_threshold(emb)
        if best_id and best_score >= threshold:
            return best_id, best_score
        return None, best_score

    def best_match_no_threshold(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        """Поиск самого похожего лица без учета порога."""
        if not self.profiles:
            return None, -1.0
        
        best_id, best_score = None, -1.0
        for pid, prof in self.profiles.items():
            mean = np.array(prof.mean_embedding, dtype=np.float32)
            # Так как векторы уже нормализованы, используем быстрое скалярное произведение
            score = float(np.dot(emb, mean))
            if score > best_score:
                best_score = score
                best_id = pid
        return best_id, best_score

    def get_all_profiles(self) -> List[dict]:
        """Получить все профили (будет использоваться для дашборда/API)."""
        return [asdict(p) for p in self.profiles.values()]