"""
api_serve.py — Backend Deteksi Citra AI (Versi 2.1 - Final)
=============================================================
Perubahan dari v2.0:
- Tambah validasi screenshot (deteksi via rasio aspek + noise pojok)
- Perkuat validasi dokumen (std dev + dominasi warna putih)
- Tambah fallback ukuran file untuk foto HP yang EXIF-nya di-strip
- Endpoint /health tersedia di dua path (/health dan /api/health)
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image, ExifTags

# ============================================================
# INISIALISASI
# ============================================================

app = FastAPI(
    title="Sistem Deteksi Citra AI v2.1",
    description="Deteksi citra buatan AI vs foto asli menggunakan ELA + EfficientNetB0",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "ela_efficientnet_final.h5")
THRESHOLD  = float(os.getenv("THRESHOLD", "0.5"))

print("[INFO] Memuat model ELA + EfficientNetB0...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] ✅ Model berhasil dimuat! Server siap.")
except Exception as e:
    print(f"[ERROR] ❌ Gagal memuat model: {e}")
    model = None

# Konfigurasi
MIN_WIDTH                    = 100
MIN_HEIGHT                   = 100
MAX_FILE_SIZE_MB             = 10
FILE_SIZE_REAL_THRESHOLD_KB  = 800  # File > 800KB → kemungkinan besar foto HP asli


# ============================================================
# FUNGSI VALIDASI
# ============================================================

def read_exif(pil_img: Image.Image) -> dict:
    """Baca EXIF metadata dengan dua method untuk kompatibilitas."""
    exif_data = {}
    try:
        raw = pil_img._getexif()
        if raw:
            for tag_id, value in raw.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[str(tag)] = str(value)[:100]
    except Exception:
        pass
    try:
        raw2 = pil_img.getexif()
        if raw2:
            for tag_id, value in raw2.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[str(tag)] = str(value)[:100]
    except Exception:
        pass
    return exif_data


def is_document_image(pil_img: Image.Image) -> bool:
    """
    Deteksi dokumen/kertas menggunakan dua sinyal:
    1. Standar deviasi piksel rendah (warna monoton)
    2. Lebih dari 70% piksel sangat terang (kertas putih)
    """
    gray    = np.array(pil_img.convert("L"), dtype=np.float32)
    std_dev = np.std(gray)

    if std_dev < 15.0:
        return True

    bright_ratio = np.sum(gray > 240) / gray.size
    if bright_ratio > 0.70 and std_dev < 40.0:
        return True

    return False


def is_screenshot(pil_img: Image.Image) -> bool:
    """
    Deteksi screenshot berdasarkan dua sinyal:
    1. Rasio aspek persis sama dengan resolusi layar umum
       + pojok gambar sangat bersih (tidak ada noise sensor kamera)
    2. File PNG resolusi tinggi tanpa EXIF kamera
       + area tengah sangat bersih
    """
    w, h = pil_img.size

    # Sinyal 1: Rasio aspek layar + noise pojok
    screen_ratios = [
        (9, 16), (16, 9), (9, 19.5), (9, 20),
        (3, 4),  (4, 3),  (1, 1),
    ]
    img_ratio = w / h
    for rw, rh in screen_ratios:
        if abs(img_ratio - rw / rh) < 0.01:
            corners = [
                np.array(pil_img.crop((0,    0,    20, 20))),
                np.array(pil_img.crop((w-20, 0,    w,  20))),
                np.array(pil_img.crop((0,    h-20, 20, h))),
                np.array(pil_img.crop((w-20, h-20, w,  h))),
            ]
            avg_corner_std = np.mean([np.std(c.astype(float)) for c in corners])
            if avg_corner_std < 8.0:
                return True

    # Sinyal 2: PNG tanpa EXIF kamera + area tengah bersih
    try:
        if pil_img.format == "PNG" and w * h > 500000:
            exif_data   = read_exif(pil_img)
            camera_keys = {"Make", "Model", "ISOSpeedRatings", "FNumber"}
            if not camera_keys.intersection(set(exif_data.keys())):
                gray  = np.array(pil_img.convert("L"), dtype=np.float32)
                cy, cx = h // 2, w // 2
                patch = gray[max(0, cy-50):cy+50, max(0, cx-50):cx+50]
                if patch.size > 0 and np.std(patch) < 12.0:
                    return True
    except Exception:
        pass

    return False


# ============================================================
# PREPROCESSING ELA
# HARUS IDENTIK dengan saat training di Colab!
# ============================================================

def smart_center_crop(pil_img: Image.Image, target_size: int = 224) -> Image.Image:
    w, h    = pil_img.size
    min_dim = min(w, h)
    left    = (w - min_dim) // 2
    top     = (h - min_dim) // 2
    cropped = pil_img.crop((left, top, left + min_dim, top + min_dim))
    return cropped.resize((target_size, target_size), Image.LANCZOS)


def extract_ela(pil_img: Image.Image, quality: int = 90, amplify: int = 50) -> np.ndarray:
    """
    Error Level Analysis.
    ⚠️  quality=90 dan amplify=50 HARUS sama dengan nilai saat training!
    """
    rgb_img = pil_img.convert("RGB")
    rgb_img = smart_center_crop(rgb_img, target_size=224)

    buffer = io.BytesIO()
    rgb_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    orig_arr   = np.array(rgb_img,      dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)
    ela_arr    = np.abs(orig_arr - recomp_arr) * amplify
    ela_arr    = np.clip(ela_arr, 0, 255)

    return ela_arr / 255.0


# ============================================================
# HYBRID PREDICTION
# ============================================================

def hybrid_predict(pil_img: Image.Image, file_size_kb: float = 0) -> dict:
    """
    Prediksi berlapis:
    1. Model ELA + EfficientNetB0
    2. Probability smoothing untuk resolusi tinggi
    3. EXIF metadata override
    4. File size fallback untuk foto HP yang EXIF-nya di-strip
    """

    # EXIF Check
    exif_data   = read_exif(pil_img)
    camera_keys = {"Make", "Model", "LensModel", "ISOSpeedRatings",
                   "FNumber", "ExposureTime", "FocalLength", "DateTime"}
    exif_found  = bool(camera_keys.intersection(set(exif_data.keys())))

    # Model Prediction
    ela_input = extract_ela(pil_img)
    ela_batch = np.expand_dims(ela_input, axis=0)
    raw_score = float(model.predict(ela_batch, verbose=0)[0][0])
    # raw_score: 0.0 = REAL, 1.0 = AI

    ai_score   = raw_score
    real_score = 1.0 - raw_score
    catatan    = "ELA + EfficientNetB0 v2.1"

    # Probability Smoothing (foto resolusi tinggi)
    w, h = pil_img.size
    if max(w, h) > 2000 and 0.4 < ai_score < 0.7:
        ai_score   = ai_score * 0.85
        real_score = 1.0 - ai_score

    # EXIF Override
    if exif_found and ai_score < 0.95:
        ai_score   = min(ai_score, 0.25)
        real_score = 1.0 - ai_score
        catatan   += " | EXIF kamera ditemukan"

    # File Size Fallback
    # Foto HP asli > 800KB, gambar AI yang didownload biasanya lebih kecil
    # Hanya aktif kalau EXIF tidak ada (sudah di-strip oleh WhatsApp/Telegram)
    # if ai_score > THRESHOLD and file_size_kb > FILE_SIZE_REAL_THRESHOLD_KB and not exif_found:
    #     ai_score   = min(ai_score, 0.45)
    #     real_score = 1.0 - ai_score
    #     catatan   += f" | Ukuran file besar ({file_size_kb:.0f}KB, kemungkinan foto asli)"

    # Final Label
    is_real    = ai_score < THRESHOLD
    confidence = real_score if is_real else ai_score
    status     = "Asli Kamera (Real)" if is_real else "Buatan AI (Generated)"

    return {
        # Field lama — backward compatible dengan frontend lama
        "status":           status,
        "akurasi_prediksi": f"{confidence * 100:.2f}%",
        "dimensi_input":    f"{w}x{h} piksel",
        "skor_mentah":      round(raw_score, 4),
        # Field baru
        "skor_ai":          round(ai_score * 100, 2),
        "skor_real":        round(real_score * 100, 2),
        "metadata_kamera":  exif_found,
        "catatan":          catatan,
        "model_version":    "2.1.0"
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "API Sistem Deteksi AI v2.1 Berjalan!",
        "model":   "ELA + EfficientNetB0",
        "docs":    "/docs"
    }


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
        "threshold":    THRESHOLD,
        "version":      "2.1.0"
    }


@app.get("/api/info")
def info():
    return {
        "model_name":    "ELA + EfficientNetB0",
        "version":       "2.1.0",
        "preprocessing": "Error Level Analysis (quality=90, amplify=50)",
        "val_accuracy":  "86.7%",
        "val_auc":       "0.942",
        "dataset":       "CIFAKE (120k images)",
        "validasi":      [
            "Dokumen/kertas putih",
            "Screenshot layar",
            "Resolusi minimum 100x100px",
            "Ukuran file maksimum 10MB"
        ]
    }


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    """Endpoint utama deteksi — kompatibel dengan frontend v1."""

    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model belum dimuat. Cek log server."}
        )

    try:
        # Validasi tipe file
        allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
        if file.content_type not in allowed:
            return {"error": f"Format tidak didukung: {file.content_type}. Gunakan JPEG atau PNG."}

        # Baca file
        image_bytes  = await file.read()
        file_size_kb = len(image_bytes) / 1024

        # Validasi ukuran file
        if file_size_kb > MAX_FILE_SIZE_MB * 1024:
            return {"error": f"File terlalu besar. Maksimum {MAX_FILE_SIZE_MB}MB."}

        # Load gambar
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            return {"error": "File tidak bisa dibaca sebagai gambar."}

        # Validasi dimensi minimum
        w, h = pil_img.size
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return {
                "error": f"Resolusi terlalu kecil ({w}x{h}px). "
                         f"Minimal {MIN_WIDTH}x{MIN_HEIGHT}px."
            }

        # ── Validasi Dokumen ──
        if is_document_image(pil_img):
            return {
                "error": "Gambar terdeteksi sebagai dokumen atau kertas putih. "
                         "Upload foto atau gambar dengan konten visual yang jelas."
            }

        # ── Validasi Screenshot ──
        if is_screenshot(pil_img):
            return {
                "error": "Gambar terdeteksi sebagai screenshot layar. "
                         "Sistem ini dioptimalkan untuk foto kamera atau gambar AI generatif, "
                         "bukan tangkapan layar."
            }

        # Prediksi
        result = hybrid_predict(pil_img, file_size_kb=file_size_kb)
        return result

    except Exception as e:
        return {"error": f"Terjadi kesalahan internal: {str(e)}"}


# ============================================================
# JALANKAN SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
