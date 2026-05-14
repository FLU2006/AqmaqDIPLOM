import os
import json
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Импортируем из нашего ядра
from src.config import INCIDENTS_PATH, DB_PATH, OUT_ROOT
from src.collections_db import CollectionsDB

# ==============================================================================
# Хак для Windows: загрузка библиотек CUDA для ONNX Runtime (если есть)
# ==============================================================================
try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(torch_lib_path)
except Exception:
    pass

import onnxruntime as ort
ort.set_default_logger_severity(4)
from src.face_detector import InsightFaceDetector
# ==============================================================================

app = FastAPI(title="Aqmaq Recognition API", version="2.0")

# CORS для связи с React/Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if OUT_ROOT.exists():
    app.mount("/images", StaticFiles(directory=OUT_ROOT), name="images")

# Инициализация базы данных
db = CollectionsDB(DB_PATH)

# Инициализация нейросети (InsightFace) внутри сервера для обработки загруженных фото
try:
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
except Exception:
    old_stderr = None

try:
    detector = InsightFaceDetector(det_size=(320, 320))
finally:
    if old_stderr is not None:
        try:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
        except Exception:
            pass


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Aqmyaq Server is running"}

@app.get("/api/incidents")
def get_incidents(limit: int = 100):
    incidents = []
    if INCIDENTS_PATH.exists():
        with open(INCIDENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        crop_path = Path(event.get("crop_path", ""))
                        if crop_path.is_absolute():
                            try:
                                rel_path = crop_path.relative_to(OUT_ROOT)
                                event["image_url"] = f"/images/{rel_path.as_posix()}"
                            except ValueError:
                                event["image_url"] = f"/images/{crop_path.name}"
                        else:
                            event["image_url"] = f"/images/{crop_path.as_posix()}"
                        
                        incidents.append(event)
                    except Exception:
                        pass
    return list(reversed(incidents))[:limit]

@app.get("/api/memory")
def get_memory():
    db._load() # Обновляем данные с диска
    profiles = db.get_all_profiles()
    
    for profile in profiles:
        img_urls = []
        for img in profile.get("images", []):
            img_path = Path(img)
            try:
                if img_path.is_absolute():
                    try:
                        rel = img_path.relative_to(OUT_ROOT)
                        img_urls.append(f"/images/{rel.as_posix()}")
                    except ValueError:
                        img_urls.append(f"/images/{img_path.name}")
                else:
                    img_urls.append(f"/images/{img_path.as_posix()}")
            except Exception:
                pass
        profile["image_urls"] = img_urls
        
    return {"profiles": profiles}

# Модели данных
class ProfileUpdate(BaseModel):
    person_id: str
    display_name: str

class ProfileDelete(BaseModel):
    person_id: str

@app.post("/api/profiles/update")
def update_profile(data: ProfileUpdate):
    """Эндпоинт для переименования профилей (включая Unknown)"""
    db._load()
    if data.person_id in db.profiles:
        db.profiles[data.person_id].display_name = data.display_name
        db._save()
        return {"status": "success", "message": "Profile updated"}
    raise HTTPException(status_code=404, detail="Profile not found")

@app.post("/api/profiles/delete")
def delete_profile(data: ProfileDelete):
    """Эндпоинт для удаления профиля"""
    db._load()
    if db.delete_person(data.person_id):
        return {"status": "success", "message": "Profile deleted"}
    raise HTTPException(status_code=404, detail="Profile not found")

@app.post("/api/profiles/add")
async def add_profile(file: UploadFile = File(...), display_name: str = Form(...)):
    """Эндпоинт для загрузки новой фотографии, поиска лица на ней и добавления в базу"""
    # 1. Читаем картинку из запроса
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # 2. Ищем лицо нейросетью
    faces = detector.app.get(img)
    if not faces:
        raise HTTPException(status_code=400, detail="На фото не найдено ни одного лица!")
    
    # Берем самое крупное лицо на фото
    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    face = faces[0]
    
    # Достаем идеальный вектор лица (ArcFace Embedding)
    emb = face.normed_embedding
    if emb is None:
        emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)

    # 3. Сохраняем фотографию в папку
    safe_name = display_name.replace(" ", "_")
    person_dir = OUT_ROOT / f"Known_{safe_name}"
    person_dir.mkdir(parents=True, exist_ok=True)
    
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = person_dir / f"manual_{ts_str}.jpg"
    
    # Обрезаем лицо для красивой картинки профиля
    x1, y1, x2, y2 = face.bbox.astype(int).tolist()
    x, y = max(0, x1), max(0, y1)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    crop = img[y:y+h, x:x+w]
    
    if crop.size > 0:
        cv2.imwrite(str(out_path), crop)
    else:
        cv2.imwrite(str(out_path), img) # Fallback

    # 4. Добавляем в базу данных
    db._load()
    # Генерируем уникальный ID, чтобы избежать коллизий
    person_id = f"Known_{safe_name}_{ts_str[-4:]}"
    db.create_person(person_id=person_id, emb=emb, image_path=str(out_path), display_name=display_name)
    
    return {"status": "success", "person_id": person_id, "message": "Сотрудник успешно добавлен"}

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Анализ загруженного фото: поиск всех лиц и сравнение с базой (Forensic Analysis)"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    faces = detector.app.get(img)
    results = []
    db._load()

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        emb = face.normed_embedding
        if emb is None:
            emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)

        # Сравниваем лицо со всеми в базе
        matched_id, score = db.match(emb, threshold=0.40) # 0.40 - мягкий порог для поиска
        
        display_name = "Unknown"
        status = "unknown"
        
        if matched_id:
            status = "known"
            display_name = db.profiles[matched_id].display_name
        
        # Рисуем рамку и подпись прямо на картинке
        color = (0, 255, 0) if status == "known" else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        label = f"{display_name} {score*100:.0f}%" if matched_id else "Unknown"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        results.append({
            "bbox": [x1, y1, x2, y2],
            "status": status,
            "person_id": matched_id,
            "display_name": display_name,
            "score": float(score) if matched_id else 0.0
        })

    # Сохраняем результат в папку OUT_ROOT/analysis
    analyze_dir = OUT_ROOT / "analysis"
    analyze_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = analyze_dir / f"result_{ts_str}.jpg"
    cv2.imwrite(str(out_path), img)

    return {
        "status": "success",
        "faces_found": len(faces),
        "image_url": f"/images/analysis/result_{ts_str}.jpg",
        "results": results
    }

# ==============================================================================
# 🔴 МАГИЯ ВИДЕОПОТОКА (MJPEG STREAMING) ОПТИМИЗИРОВАННАЯ
# ==============================================================================

def get_fallback_frame():
    """Создаем заглушку 'NO SIGNAL', если скрипт камеры выключен или завис"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "WAITING FOR CAMERA SCRIPT...", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def generate_frames():
    """Чтение кадров с диска и отправка в формате видеопотока (ОПТИМИЗИРОВАНО БЕЗ МЕРЦАНИЙ)"""
    frame_path = OUT_ROOT / "latest_frame.jpg"
    fallback_data = get_fallback_frame()
    
    last_mtime = 0.0
    
    # Отправляем заглушку при старте потока
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + fallback_data + b'\r\n')
    
    while True:
        frame_to_send = None
        try:
            if frame_path.exists():
                current_mtime = frame_path.stat().st_mtime
                
                # 1. Если скрипт камеры завис (файл не обновлялся 3 секунды)
                if time.time() - current_mtime > 3.0:
                    if last_mtime != -1.0: 
                        frame_to_send = fallback_data
                        last_mtime = -1.0
                        
                # 2. Если появился НОВЫЙ кадр
                elif current_mtime != last_mtime:
                    with open(frame_path, "rb") as f:
                        data = f.read()
                    
                    # ======================================================
                    # 🪄 МАГИЯ АНТИ-МЕРЦАНИЯ:
                    # Защита от "половинчатых" кадров во время записи Windows.
                    # Любой 100% готовый JPEG всегда начинается на FF D8 и 
                    # заканчивается на FF D9. Если конца нет - кадр ещё пишется.
                    # ======================================================
                    if len(data) > 100 and data.startswith(b'\xff\xd8') and data.endswith(b'\xff\xd9'): 
                        frame_to_send = data
                        last_mtime = current_mtime
            else:
                if last_mtime != -1.0:
                    frame_to_send = fallback_data
                    last_mtime = -1.0
                    
        except Exception:
            # Файл заблокирован виндой в процессе записи, просто ждем
            pass

        # Отправляем кадр ТОЛЬКО если он полностью готов и обновился
        if frame_to_send is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
        
        # Минимальная пауза, чтобы не вешать процессор. Ожидание нового кадра
        time.sleep(0.01) 

@app.get("/api/video_feed")
def video_feed():
    """Эндпоинт, к которому подключается React-дашборд для получения видео"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")