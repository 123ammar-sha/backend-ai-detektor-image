import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Konfigurasi Parameter
DATA_DIR = 'dataset_noise'
IMG_SIZE = (224, 224) # Ukuran standar MobileNetV2
BATCH_SIZE = 16 # Sesuaikan dengan RAM Anda (bisa diturunkan ke 8 jika berat)
EPOCHS = 10 # Untuk awal, 10 putaran sudah cukup

def build_model():
    """Membangun arsitektur Transfer Learning menggunakan MobileNetV2"""
    # 1. Muat base model (MobileNetV2 tanpa lapisan klasifikasi terakhir)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # 2. Bekukan (freeze) lapisan dasar agar bobot pra-latih tidak rusak
    for layer in base_model.layers:
        layer.trainable = False

    # 3. Tambahkan "Kepala" klasifikasi khusus untuk tugas kita (Asli vs AI)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Mencegah overfitting
    
    # Output layer: 1 node dengan sigmoid (0 = AI, 1 = Real - tergantung urutan abjad folder)
    predictions = Dense(1, activation='sigmoid')(x)

    # 4. Gabungkan menjadi satu model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 5. Kompilasi model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_system():
    # 1. Persiapkan Data Generator (Hanya normalisasi piksel, tanpa augmentasi berat agar noise tidak rusak)
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Pastikan label kelasnya (siapa yang jadi 0 dan siapa yang jadi 1)
    print(f"[INFO] Pemetaan Kelas: {train_generator.class_indices}")

    # 2. Bangun Model
    print("[INFO] Membangun Arsitektur Model...")
    model = build_model()

    # 3. Mulai Pelatihan
    print("[INFO] Memulai Proses Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # 4. Simpan Model Hasil Latihan
    model_path = 'ai_detector_model.h5'
    model.save(model_path)
    print(f"\n[SELESAI] Model berhasil dilatih dan disimpan sebagai '{model_path}'")

if __name__ == '__main__':
    train_system()