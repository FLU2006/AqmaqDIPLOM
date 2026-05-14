# import sys
# from pathlib import Path
# from datetime import datetime

# import numpy as np
# import torch
# from PIL import Image
# from facenet_pytorch import InceptionResnetV1, MTCNN

# from face_db import FaceDB


# # порог подбирается экспериментально; стартуем с 0.65–0.75
# THRESHOLD = 0.70

# DATA_OUT = Path("outputs") / "collections"
# DATA_OUT.mkdir(parents=True, exist_ok=True)


# def save_crop(crop: Image.Image, person_id: str) -> str:
#     person_dir = DATA_OUT / person_id
#     person_dir.mkdir(parents=True, exist_ok=True)

#     ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     out_path = person_dir / f"{ts}.jpg"
#     crop.save(out_path, quality=95)
#     return str(out_path)


# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python src/embed_infer.py <path_to_image>")
#         sys.exit(1)

#     img_path = Path(sys.argv[1])
#     if not img_path.exists():
#         print("File not found:", img_path)
#         sys.exit(1)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Device:", device)

#     # детекция и кроп лица
#     mtcnn = MTCNN(image_size=160, margin=20, device=device)
#     resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

#     img = Image.open(img_path).convert("RGB")
#     face_tensor = mtcnn(img)  # (3,160,160) или None

#     if face_tensor is None:
#         print("No face detected")
#         sys.exit(2)

#     # сохраним кроп для коллекции
#     crop_img = mtcnn.extract(img, mtcnn.detect(img)[0], save_path=None)[0]
#     # extract вернёт PIL кропы; берём первый
#     if isinstance(crop_img, list):
#         crop_img = crop_img[0]

#     # embedding
#     x = face_tensor.unsqueeze(0).to(device)
#     with torch.no_grad():
#         emb = resnet(x).cpu().numpy().astype(np.float32)[0]

#     db = FaceDB(Path("outputs") / "face_db")

#     matched_id, score = db.match(emb, threshold=THRESHOLD)

#     if matched_id is None:
#         # новый unknown
#         temp_path = save_crop(crop_img, "UNKNOWN_TMP")
#         new_id = db.add_new_person(emb, temp_path)
#         # перенесём фото в правильную папку
#         # (проще — пересохранить)
#         final_path = save_crop(crop_img, new_id)
#         # удалить временный файл
#         try:
#             Path(temp_path).unlink(missing_ok=True)
#         except Exception:
#             pass
#         # обновим ссылку на изображение
#         db.profiles[new_id].images[-1] = final_path
#         print(f"NEW person: {new_id} | score={score:.4f}")
#     else:
#         out_path = save_crop(crop_img, matched_id)
#         db.add_to_person(matched_id, emb, out_path)
#         print(f"MATCH: {matched_id} | score={score:.4f}")


# if __name__ == "__main__":
#     main()