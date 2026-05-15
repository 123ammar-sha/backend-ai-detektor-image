from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io
from PIL import Image
from noise_extractor import extract_noise
import cv2

# 1. Inisialisasi Aplikasi API
app = FastAPI(title="Sistem Deteksi Citra AI")

# --- 2. Tambahkan blok kode CORS ini ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua domain (untuk development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Muat Otak AI (hanya dilakukan sekali saat server menyala)
print("[INFO] Sedang memuat model AI (Mohon tunggu)...")
model = tf.keras.models.load_model('ai_detector_model.h5')
print("[INFO] Model berhasil dimuat! Server siap menerima request.")

# Konfigurasi Batas Resolusi Minimum
MIN_WIDTH = 256
MIN_HEIGHT = 256


def is_document(image_path):
    # Baca gambar dalam mode grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    
    # Hitung standar deviasi (variasi warna)
    # Foto dokumen biasanya punya variasi warna yang rendah dibanding foto natural
    _, std_dev = cv2.meanStdDev(img)
    
    # Jika std_dev di bawah ambang batas tertentu (misal 45), 
    # kemungkinan besar itu dokumen atau gambar dengan informasi visual minim
    return std_dev[0][0] < 45

@app.get("/")
def home():
    return {"message": "API Sistem Deteksi AI Berjalan Lancar!"}

@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        # 3. Baca gambar ke memori untuk validasi dimensi (tanpa menyimpannya dulu)
        image_bytes = await file.read()
        img_check = Image.open(io.BytesIO(image_bytes))
        width, height = img_check.size
        
        # 4. Filter Gate: Tolak jika terlalu kecil
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return {
                "error": f"Resolusi gambar terlalu rendah ({width}x{height}). Silakan unggah gambar minimal {MIN_WIDTH}x{MIN_HEIGHT} piksel agar deteksi akurat."
            }
            
        # 5. Persiapkan path untuk file sementara
        temp_input = f"temp_{file.filename}"
        temp_noise = f"noise_{file.filename}"
        
        # 6. Simpan gambar mentah ke penyimpanan sementara
        with open(temp_input, "wb") as buffer:
            buffer.write(image_bytes)


        # --- VALIDASI DOKUMEN ---
        if is_document(temp_input):
            if os.path.exists(temp_input): os.remove(temp_input)
            return {
                "error": "Gambar terdeteksi sebagai dokumen atau teks. Sistem ini dioptimalkan untuk deteksi foto wajah atau pemandangan asli vs AI."
            }
        # ------------------------
            
        # 7. Terapkan High-Pass Filter (Ekstrak Artefak)
        extract_noise(temp_input, temp_noise)
        
        # 8. Persiapkan gambar untuk masuk ke model AI (wajib 224x224)
        img = image.load_img(temp_noise, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0 # Normalisasi piksel
        
        # 9. Minta AI menebak
        prediction = model.predict(img_array)[0][0]
        
        # Interpretasi hasil (0 = AI, 1 = Real)
        is_real = prediction > 0.5
        confidence = prediction if is_real else (1 - prediction)
        
        # 10. Bersihkan file sampah agar hardisk server tidak penuh
        if os.path.exists(temp_input): os.remove(temp_input)
        if os.path.exists(temp_noise): os.remove(temp_noise)
        
        # 11. Kembalikan hasil yang rapi
        return {
            "status": "Asli Kamera (Real)" if is_real else "Buatan AI (Generated)",
            "akurasi_prediksi": f"{confidence * 100:.2f}%",
            "dimensi_input": f"{width}x{height} piksel",
            "skor_mentah": float(prediction)
        }
        
    except Exception as e:
        # Pengecekan pembersihan file jika terjadi error di tengah proses
        if 'temp_input' in locals() and os.path.exists(temp_input): os.remove(temp_input)
        if 'temp_noise' in locals() and os.path.exists(temp_noise): os.remove(temp_noise)
        
        return {"error": f"Terjadi kesalahan internal: {str(e)}"}

if __name__ == "__main__":
    # Menjalankan server di localhost port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)