import os
import sys
import warnings

# Подавляем назойливые FutureWarning от библиотеки insightface
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
from datetime import datetime
import json
import time
import threading
import concurrent.futures

import cv2
import numpy as np

try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(torch_lib_path)
except Exception:
    pass
# ==============================================================================

import onnxruntime as ort
ort.set_default_logger_severity(4)

from config import (
    DEBUG_LOGS, DETECTOR_BACKEND, DETECTOR_SIZE, DEVICE,
    MATCH_THRESHOLD, MERGE_UNKNOWN_THRESHOLD, DUPLICATE_SIM_THRESHOLD,
    MIN_FACE_SIZE, BLUR_THRESHOLD, TRACK_TIMEOUT_SEC,
    TRACK_MAX_CENTER_DISTANCE_RATIO, TRACK_MIN_IOU_FOR_MATCH,
    TRACK_CONFIRMATION_SAMPLES, TRACK_MAX_PENDING_SAMPLES,
    TRACK_CAPTURE_INTERVAL_SEC, TRACK_EVENT_COOLDOWN_SEC,
    PERSON_SAVE_COOLDOWN_SEC, OUT_ROOT, DB_PATH, INCIDENTS_PATH, SOURCE_NAME
)
from collections_db import CollectionsDB
from face_detector import InsightFaceDetector
from track_manager import TrackManager, cosine_similarity

# Пул потоков для фонового сохранения картинок
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ==========================================
# 🚀 КЛАСС ДЛЯ КАМЕРЫ БЕЗ ЛАГОВ
# ==========================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame

    def read(self):
        if self.frame is not None:
            return self.ret, self.frame.copy()
        return self.ret, None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def dprint(*args, **kwargs):
    if DEBUG_LOGS:
        print(*args, **kwargs)

def variance_of_laplacian(gray_img: np.ndarray) -> float:
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

def log_event(person_id: str, score: float, source: str, crop_path: str, track_id: int):
    INCIDENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "track_id": track_id,
        "person_id": person_id,
        "score": float(score),
        "crop_path": crop_path,
    }
    with INCIDENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    dprint(f"[LOG] incident written: {person_id} ({score:.4f})")

def save_task(img_bgr: np.ndarray, person_id: str, track_id: int, score: float):
    try:
        person_dir = OUT_ROOT / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = person_dir / f"{ts_str}.jpg"
        
        ok, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            out_path.write_bytes(buffer.tobytes())
            log_event(person_id, score, SOURCE_NAME, str(out_path), track_id)
            dprint(f"[ASYNC] Saved {person_id} (T{track_id})")
    except Exception as e:
        print(f"[ERROR] Save task failed: {e}")

def finalize_track_decision(track, db, crop_bgr, last_saved_ts, last_saved_emb, known_cache, now_ts):
    mean_emb = track.mean_embedding()
    if mean_emb is None:
        return None, 0.0, None, "pending"

    # Обновляем БД перед проверкой, вдруг кого-то добавили или переименовали через веб-интерфейс
    db._load()

    matched_id, score = db.match(mean_emb, threshold=MATCH_THRESHOLD)
    
    if matched_id is None and known_cache:
        best_cached_id, best_cached_score = None, 0.0
        for pid, emb in known_cache.items():
            sim = cosine_similarity(mean_emb, emb)
            if sim > best_cached_score:
                best_cached_score, best_cached_id = sim, pid
        if best_cached_score >= (MATCH_THRESHOLD - 0.1):
            matched_id, score = best_cached_id, best_cached_score

    if matched_id is not None:
        track.status = "known"
        track.person_id = matched_id
        track.best_score = score
        track.representative_emb = mean_emb

        prev_ts = last_saved_ts.get(matched_id, 0.0)
        prev_emb = last_saved_emb.get(matched_id)
        
        should_save = True
        if now_ts - prev_ts < PERSON_SAVE_COOLDOWN_SEC:
            should_save = False
        elif prev_emb is not None and cosine_similarity(mean_emb, prev_emb) >= DUPLICATE_SIM_THRESHOLD:
            should_save = False

        if should_save:
            executor.submit(save_task, crop_bgr.copy(), matched_id, track.track_id, score)
            last_saved_ts[matched_id] = now_ts
            last_saved_emb[matched_id] = mean_emb
            track.last_event_ts = now_ts
        else:
            known_cache[matched_id] = mean_emb
        return matched_id, score, None, "known"

    best_id, best_score = db.best_match_no_threshold(mean_emb)

    if best_id and not best_id.startswith("Unknown_"):
        if best_score >= (MATCH_THRESHOLD - 0.04):
            track.status = "pending"
            return None, best_score, None, "pending"

    if best_id and best_id.startswith("Unknown_") and best_score >= MERGE_UNKNOWN_THRESHOLD:
        track.status = "unknown"
        track.person_id = best_id
        track.best_score = best_score
        track.representative_emb = mean_emb
        
        executor.submit(save_task, crop_bgr.copy(), best_id, track.track_id, best_score)
        db.add_sample(best_id, mean_emb, "async_save_placeholder")
        track.last_event_ts = now_ts
        return best_id, best_score, None, "unknown"

    new_unknown_id = db.create_unknown(mean_emb, "async_save_placeholder")
    executor.submit(save_task, crop_bgr.copy(), new_unknown_id, track.track_id, 0.0)
    
    track.status = "unknown"
    track.person_id = new_unknown_id
    track.best_score = 0.0
    track.representative_emb = mean_emb
    track.last_event_ts = now_ts
    return new_unknown_id, 0.0, None, "unknown"

def check_gpu_status():
    print("\n" + "="*40)
    print("🚀 ПРОВЕРКА ДОСТУПНОСТИ ВИДЕОКАРТЫ (GPU)")
    print("="*40)
    ort_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in ort_providers:
        print("✅ InsightFace (ONNX): Модуль CUDA подключен")
    else:
        print("❌ InsightFace (ONNX): Работает на ПРОЦЕССОРЕ (CPU)")
    print("="*40 + "\n")

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    check_gpu_status()

    db = CollectionsDB(DB_PATH)
    tracker = TrackManager(
        track_timeout_sec=TRACK_TIMEOUT_SEC,
        max_center_distance_ratio=TRACK_MAX_CENTER_DISTANCE_RATIO,
        min_iou_for_match=TRACK_MIN_IOU_FOR_MATCH,
    )

    if DETECTOR_BACKEND != "insightface":
        raise RuntimeError(f"Unsupported detector backend: {DETECTOR_BACKEND}")

    # Отключаем вывод C++ ошибок
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
    except Exception:
        old_stderr = None

    try:
        detector = InsightFaceDetector(det_size=(DETECTOR_SIZE, DETECTOR_SIZE))
    finally:
        if old_stderr is not None:
            try:
                os.dup2(old_stderr, 2)
                os.close(devnull)
                os.close(old_stderr)
            except Exception:
                pass

    cap = ThreadedCamera(0)
    time.sleep(1)
    
    if not cap.ret:
        raise RuntimeError("Cannot open webcam. Close other apps.")

    last_saved_ts = {}
    last_saved_emb = {}
    known_cache = {}

    enroll_mode = False
    enroll_name = None

    is_processing = False
    latest_draw_data = []

    def process_frame(frame_copy, now_ts, current_enroll_mode, current_enroll_name):
        nonlocal is_processing, latest_draw_data
        try:
            tracker.begin_frame()
            faces = detector.app.get(frame_copy)
            new_draw_data = []

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int).tolist()
                x, y = max(0, x1), max(0, y1)
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                bbox = (x, y, w, h)

                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE: 
                    continue
                
                crop = frame_copy[y:y+h, x:x+w].copy()
                if crop.size == 0:
                    continue

                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur_score = variance_of_laplacian(crop_gray)
                if blur_score < BLUR_THRESHOLD: 
                    continue

                # Идеальные эмбеддинги от ArcFace
                emb = face.normed_embedding
                if emb is None:
                    emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)

                track = tracker.assign(bbox, now_ts)
                
                if hasattr(track, 'bbox') and track.bbox:
                    tx, ty, tw, th = track.bbox
                else:
                    tx, ty, tw, th = x, y, w, h

                if current_enroll_mode and current_enroll_name:
                    prev_ts = last_saved_ts.get(current_enroll_name, 0.0)
                    enough_time = (now_ts - prev_ts) >= PERSON_SAVE_COOLDOWN_SEC
                    
                    prev_emb = last_saved_emb.get(current_enroll_name)
                    not_duplicate = (prev_emb is None) or (cosine_similarity(emb, prev_emb) < DUPLICATE_SIM_THRESHOLD)

                    if enough_time and not_duplicate:
                        executor.submit(save_task, crop.copy(), current_enroll_name, track.track_id, 1.0)
                        db.create_person(current_enroll_name, emb, "enroll_async")
                        last_saved_ts[current_enroll_name] = now_ts
                        last_saved_emb[current_enroll_name] = emb

                    label = f"ENROLL {current_enroll_name} [T{track.track_id}]"
                    new_draw_data.append((tx, ty, tw, th, (0, 255, 255), label))
                    continue

                if track.status == "pending":
                    can_capture = (now_ts - track.last_capture_ts) >= TRACK_CAPTURE_INTERVAL_SEC
                    if can_capture and len(track.samples) < TRACK_CONFIRMATION_SAMPLES:
                        track.add_embedding(emb)
                        track.last_capture_ts = now_ts

                    if len(track.samples) >= TRACK_CONFIRMATION_SAMPLES or len(track.samples) >= TRACK_MAX_PENDING_SAMPLES:
                        person_id, score, _, status = finalize_track_decision(
                            track, db, crop, last_saved_ts, last_saved_emb, known_cache, now_ts
                        )
                        color = (0, 255, 0) if status == "known" else (0, 165, 255)
                        if person_id is None: color = (0, 200, 255)
                    else:
                        color = (0, 200, 255)

                    # Берем display_name из базы (на случай, если переименовали на сайте)
                    display_name = track.person_id
                    if track.person_id and track.person_id in db.profiles:
                         display_name = db.profiles[track.person_id].display_name

                    label = f"T{track.track_id} pending {len(track.samples)}/{TRACK_CONFIRMATION_SAMPLES}" if track.status == "pending" else f"{display_name} {track.best_score:.2f}"
                    new_draw_data.append((tx, ty, tw, th, color, label))
                    continue

                person_id = track.person_id or f"T{track.track_id}"
                
                # Проверяем display_name из базы
                display_name = person_id
                if person_id in db.profiles:
                    display_name = db.profiles[person_id].display_name

                score = track.best_score if track.best_score >= 0 else 0.0

                if track.representative_emb is not None and (now_ts - track.last_event_ts) >= TRACK_EVENT_COOLDOWN_SEC:
                    sim = cosine_similarity(emb, track.representative_emb)
                    if sim < DUPLICATE_SIM_THRESHOLD:
                        executor.submit(save_task, crop.copy(), person_id, track.track_id, score)
                        track.last_event_ts = now_ts
                        track.representative_emb = emb

                color = (60, 255, 60) if track.status == "known" else (0, 165, 255)
                
                # Делаем рамку зеленой, если мы вручную переименовали Unknown
                if display_name and not display_name.startswith("Unknown"):
                     color = (60, 255, 60)
                
                label = f"{display_name} {score:.2f} [T{track.track_id}]"
                new_draw_data.append((tx, ty, tw, th, color, label))

            tracker.cleanup(now_ts)
            latest_draw_data = new_draw_data
        except Exception as e:
            print(f"[ERROR] Detection thread error: {e}")
        finally:
            is_processing = False

    print("Controls:\n  Q = quit\n  E = toggle enroll mode\n  1 = Known_Nazar\n  2 = Known_Alzhan\n  3 = Known_Kanysh")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        now_ts = time.time()
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord("q"), ord("Q")): break
        if key in (ord("e"), ord("E")): enroll_mode = not enroll_mode; print("Enroll mode:", enroll_mode)
        if key == ord("1"): enroll_name = "Known_Nazar"; print("Enroll name:", enroll_name)
        elif key == ord("2"): enroll_name = "Known_Alzhan"; print("Enroll name:", enroll_name)
        elif key == ord("3"): enroll_name = "Known_Kanysh"; print("Enroll name:", enroll_name)

        if not is_processing:
            is_processing = True
            threading.Thread(target=process_frame, args=(frame.copy(), now_ts, enroll_mode, enroll_name), daemon=True).start()

        for (tx, ty, tw, th, color, label) in latest_draw_data:
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), color, 2)
            cv2.putText(frame, label, (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        
        frame_path = OUT_ROOT / "latest_frame.jpg"
        cv2.imwrite(str(frame_path), frame)

        cv2.imshow("Aqmyaq Recognition (Smooth Real-Time)", frame)

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)

if __name__ == "__main__":
    main()