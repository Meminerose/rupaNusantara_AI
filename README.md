# Prototype Deteksi Etnis & Emosi

## 1. Install dependencies

```bash
python -m venv venv           # Atau bisa langsung: python -3.11 -m venv venv (biar mediapipe tidak bermasalah)
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Kalau ada GPU CUDA dan mau dipakai, install `torch`/`torchvision`sesuai
> versi CUDA di https://pytorch.org/get-started/locally/ sebelum
> `pip install -r requirements.txt` (baris torch di requirements ini generic
> CPU/GPU-agnostic).

## 2. Jalankan Program

```bash
streamlit run app.py
```

Ada 2 mode di sidebar:
- **Live Webcam** — real-time via `streamlit-webrtc`, klik START untuk mulai.
- **Upload Gambar** — untuk foto statis, bisa berisi beberapa wajah sekaligus.

## Struktur project

```
rupaNusantaraAI/
├── app.py           # UI Streamlit + orkestrasi pipeline
├── models.py        # arsitektur, loading checkpoint, fungsi predict()
├── face_utils.py     # deteksi multi-wajah (MediaPipe) + gambar bounding box
├── config.py         # SEMUA hal yang perlu disesuaikan ada di sini
├── requirements.txt
└── checkpoints/       # taruh file .pth model di sini
```

## Tips untuk demo panggung GIHN

- Live webcam menjalankan 2 model deep learning tiap frame untuk semua wajah
  yang terdeteksi — di laptop tanpa GPU bisa terasa agak lag kalau banyak
  wajah sekaligus. **Test dulu di device yang akan dipakai saat demo**, dan
  siapkan mode "Upload Gambar" sebagai cadangan kalau live terlalu berat.
- `FACE_DETECTION_MODEL_SELECTION = 1` di `config.py` dipakai supaya deteksi
  tetap jalan untuk wajah yang agak jauh dari kamera.
  Kalau audiens dekat kamera, bisa dicoba ganti ke `0`.
- Kalau model belum siap / checkpoint belum ada, app akan menampilkan pesan
  error yang jelas di halaman utama (bukan crash), jadi aman dipakai untuk
  development bertahap.


## Mengenai permasalahan dengan mediapipe mp.solutions yang tidak ter deteksi berikut langkah-langkah yang bisa dilakukan
- Penyebab ini terjadi biasanya adalah: versi python dan mediapipe tidak cocok maka dari itu
- Rekomendasi versi: Python 3.11 dan MediaPipe 0.10.21
- Step step:
  - Kalau sudah activate env, deactivate dan cek versi python (py -3.11 --version) jika muncul versi maka aman
  - Hapus env lama (Remove-Item -Recurse -Force .\venv)
  - Buat env baru dengan python 3.11 (py -3.11 -m venv venv)
  - Aktifkan env (.\venv\Scripts\activate)
  - Pastikan versi python lagi dan install mediapipe (python -m pip install --upgrade pip, pip install mediapipe==0.10.21) lalu cek juga dengan (python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))") jika muncul versi mediapipe dan TRUE maka bisa berjalan dengan normal
  - Lanjut ke step (pip install -r requirements.txt) dengan penyesuaian versi mediapipe di bagian requirements.txt sesuai step sebelum.

## Mengenai jenis model (Khusus untuk Etnis)
Disini ada 4 jenis model yang bisa dipakai dan dipilih. Cara penggantian model adalah:
- Buka **config.py** dan cari **ETHNICITY_MODEL_PATH** ketikan jenis model yang dimau sesuai dengan dict *ETHNICITY_CHECKPOINT*
- Masih di **config.py** cari **ETHNICITY_CHECKPOINT_PATH** jangan lupa sesuaikan nama model di *ETHNICITY_CHECKPOINT*
- beralih ke **models.py** cari fungsi bernama **load_ethnicity_model** lalu ke variabel **model_type** dan sesuiakan juga nama model nya
