import os
import random
from noise_extractor import extract_noise

# Konfigurasi Direktori
RAW_DIR = 'dataset_raw'
OUT_DIR = 'dataset_noise'
CLASSES = ['real', 'ai']
SPLIT_RATIO = 0.8 # 80% data untuk Training, 20% untuk Validation

def setup_directories():
    """Mempersiapkan struktur folder output"""
    for split in ['train', 'val']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

def process_dataset():
    """Memproses semua gambar dan mendistribusikannya"""
    setup_directories()
    
    for cls in CLASSES:
        raw_class_dir = os.path.join(RAW_DIR, cls)
        
        if not os.path.exists(raw_class_dir):
            print(f"[WARNING] Folder {raw_class_dir} tidak ditemukan. Lewati...")
            continue
            
        # Filter file hanya untuk ekstensi gambar
        images = [f for f in os.listdir(raw_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"[WARNING] Tidak ada gambar di {raw_class_dir}.")
            continue
            
        # Acak urutan array untuk mencegah bias urutan data
        random.shuffle(images)
        
        # Tentukan titik potong array
        split_idx = int(len(images) * SPLIT_RATIO)
        train_data = images[:split_idx]
        val_data = images[split_idx:]
        
        # Fungsi pembantu untuk memproses dan menyimpan list gambar
        def process_and_save(image_list, split_type):
            for img_name in image_list:
                input_path = os.path.join(raw_class_dir, img_name)
                
                # Standarisasi output ke format .jpg
                output_name = os.path.splitext(img_name)[0] + '_noise.jpg'
                output_path = os.path.join(OUT_DIR, split_type, cls, output_name)
                
                # Memanggil fungsi High-Pass Filter dari Hari 1
                # Output print dari extract_noise diabaikan di sini agar terminal tetap rapi
                try:
                    extract_noise(input_path, output_path)
                except Exception as e:
                    print(f"Gagal memproses {img_name}: {e}")
                    
        print(f"Menyiapkan {len(train_data)} fitur Training untuk kelas '{cls}'...")
        process_and_save(train_data, 'train')
        
        print(f"Menyiapkan {len(val_data)} fitur Validation untuk kelas '{cls}'...")
        process_and_save(val_data, 'val')

if __name__ == '__main__':
    print("Mulai Batch Processing Dataset Artefak AI...")
    process_dataset()
    print("\n[SELESAI] Pipeline data berhasil dijalankan!")
    print(f"Silakan periksa folder '{OUT_DIR}' untuk melihat hasilnya.")