"""
app.py
======
Prototype Streamlit untuk proyek Roro Jonggrang (GIHN).
Pipeline: MediaPipe (deteksi multi-wajah) -> ResNet50 (etnis) + EfficientNet (emosi)
dijalankan paralel untuk tiap wajah yang terdeteksi.

Cara jalan:
    streamlit run app.py

Sebelum dipakai, cek & lengkapi config.py (checkpoint path, urutan class label,
ukuran input, dsb).
"""

import time
import threading

from anyio import current_time
import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image

import config
import face_utils
import models


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🎭",
    layout="wide",
)


@st.cache_resource
def get_ice_servers():
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning(f"Twilio credentials tidak valid/ditemukan. Menggunakan STUN Google bawaan.")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": get_ice_servers()}
)


# ---------------------------------------------------------------------------
# Cached resource loaders
# Model & detector hanya dimuat sekali walau interaksi user berulang.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model etnis (ResNet50)...")
def get_ethnicity_model():
    return models.load_ethnicity_model()


@st.cache_resource(show_spinner="Memuat model emosi (EfficientNet)...")
def get_emotion_model():
    return models.load_emotion_model()


#@st.cache_resource(show_spinner=False)
def get_face_detector():
    return face_utils.get_face_detector()


def try_load_models():
    """
    Coba load kedua model. Kalau gagal (checkpoint belum ada / arsitektur
    tidak cocok), tampilkan pesan error yang jelas tapi jangan crash total.
    """
    ethnicity_model, emotion_model = None, None
    errors = []

    try:
        ethnicity_model = get_ethnicity_model()
    except models.ModelLoadError as e:
        errors.append(f"**Model Etnis (ResNet50):** {e}")

    try:
        emotion_model = get_emotion_model()
    except models.ModelLoadError as e:
        errors.append(f"**Model Emosi (EfficientNet):** {e}")

    return ethnicity_model, emotion_model, errors


# ---------------------------------------------------------------------------
# Core pipeline: 1 frame -> deteksi wajah -> prediksi 2 model per wajah
# ---------------------------------------------------------------------------
def run_pipeline_on_frame(frame_rgb, detector, ethnicity_model, emotion_model):
    """
    Return:
        annotated_bgr : frame dengan bounding box + label (siap ditampilkan/dikirim balik)
        results       : list of dict, satu entri per wajah, untuk ditabelkan
    """
    # --- PENGAMAN UTAMA AGAR MEDIAPIPE TIDAK KORUP ---
    if frame_rgb is None or not isinstance(frame_rgb, np.ndarray):
        return cv2.cvtColor(np.zeros((480, 640, 3), dtype=np.uint8), cv2.COLOR_RGB2BGR), []
    
    if len(frame_rgb.shape) != 3 or frame_rgb.shape[2] != 3:
        return cv2.cvtColor(np.zeros((480, 640, 3), dtype=np.uint8), cv2.COLOR_RGB2BGR), []
    # -----------------------------------------------
    frame_rgb = np.ascontiguousarray(frame_rgb)

    faces = face_utils.detect_faces(detector, frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    results = []
    for i, face in enumerate(faces):
        eth_label, eth_conf, eth_probs = models.predict(
            ethnicity_model, face.crop_rgb, config.ETHNICITY_CLASSES
        )
        emo_label, emo_conf, emo_probs = models.predict(
            emotion_model, face.crop_rgb, config.EMOTION_CLASSES
        )

        face_utils.draw_annotations(
            frame_bgr, face.box,
            ethnicity_text=f"{eth_label} ({eth_conf:.0%})",
            emotion_text=f"{emo_label} ({emo_conf:.0%})",
        )

        results.append({
            "Wajah": f"#{i + 1}",
            "Etnis": eth_label,
            "Confidence Etnis": f"{eth_conf:.1%}",
            "Emosi": emo_label,
            "Confidence Emosi": f"{emo_conf:.1%}",
            "_crop_rgb": face.crop_rgb,
            "box": face.box
        })

    return frame_bgr, results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Setting")
    st.caption(config.APP_SUBTITLE)

    mode = st.radio(
        "Sumber input",
        options=["📷 Live Webcam", "🖼️ Upload Gambar"],
        index=0,
    )
    
    st.divider()
    st.write("🔧 Penyesuaian Layar")
    rotasi_kamera = st.selectbox(
        "Rotasi Webcam (Bila Kamera Dimiringkan)",
        options=["Normal (Landscape)", "Putar Kanan 90° (Portrait)", "Putar Kiri 90° (Portrait)"],
        index=0
    )

    st.divider()
    st.caption(f"Device inferensi: `{config.DEVICE}`")

    with st.expander("ℹ️ Status model & konfigurasi"):
        st.write(f"- Checkpoint etnis: `{config.ETHNICITY_CHECKPOINT_PATH}`")
        st.write(f"- Checkpoint emosi: `{config.EMOTION_CHECKPOINT_PATH}`")
        st.write(f"- Jumlah kelas etnis: {config.NUM_ETHNICITY_CLASSES}")
        st.write(f"- Jumlah kelas emosi: {config.NUM_EMOTION_CLASSES}")

cv_rotation_flag = None
if "Kanan" in rotasi_kamera:
    cv_rotation_flag = cv2.ROTATE_90_CLOCKWISE
elif "Kiri" in rotasi_kamera:
    cv_rotation_flag = cv2.ROTATE_90_COUNTERCLOCKWISE

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(f"🎭 RupaNusantara AI")
st.caption(
    "Prototype deteksi 7 etnis & pengenalan emosi wajah: mendukung banyak wajah sekaligus dalam satu frame. "
    "Model ini masih dalam tahap prototype. Akurasi deteksi etnis dan emosi mungkin tidak sempurna "
    "terutama pada kondisi pencahayaan yang buruk atau wajah yang tidak jelas"
)

ethnicity_model, emotion_model, load_errors = try_load_models()
detector = get_face_detector()

if load_errors:
    st.error(
        "Model belum berhasil dimuat. Lengkapi terlebih dahulu:\n\n"
        + "\n\n".join(f"- {e}" for e in load_errors)
    )
    st.stop()

st.success("Model etnis & emosi berhasil dimuat. Siap digunakan.", icon="✅")

with st.expander("📖 Panduan Penggunaan", expanded=False):
    st.markdown(
        """
        Selamat datang di **RupaNusantara AI**! Ikuti langkah mudah ini untuk mencoba deteksi etnis dan emosi:
        
        1. **Pilih Mode Input:**
           - Buka **Sidebar (⚙️ Setting)** di sebelah kiri.
           - Pilih **📷 Live Webcam** untuk deteksi langsung secara *real-time*, atau **🖼️ Upload Gambar** untuk mencoba foto.
        
        2. **Atur Posisi Wajah:**
            - Pastikan pencahayaan cukup terang dan wajah menghadap lurus ke arah kamera agar sistem dapat mendeteksi dengan akurat.
        
        3. **Lihat Hasil Analisis:**
           - Kotak kuning (*bounding box*) akan muncul di wajah yang terdeteksi lengkap dengan label **Etnis** dan **Emosi** beserta persentase tingkat kepercayaan.

        4. **Unduh Hasil:**
            - Pada mode gambar, kamu bisa langsung mengunduh hasil jepretan yang sudah di-annotasi oleh AI!
            
        **Notes:**
        - Daftar etnis yang didukung: Jawa, Ambon, Toraja, Batak, Chinese, Papua, dan Kalimantan
        - Daftar emosi yang didukung: Happy, Sad, Angry, Suprised, Disgust, dan Fear
        """
    )

def render_results_table(results):
    if not results:
        st.info("Belum ada wajah terdeteksi.")
        return

    st.subheader(f"Hasil deteksi — {len(results)} wajah ditemukan")

    cols = st.columns(min(len(results), 4))
    for i, res in enumerate(results):
        with cols[i % len(cols)]:
            st.image(res["_crop_rgb"], caption=res["Wajah"], use_container_width=True)
            st.markdown(f"**Etnis:** {res['Etnis']} ({res['Confidence Etnis']})")
            st.markdown(f"**Emosi:** {res['Emosi']} ({res['Confidence Emosi']})")

    table_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["_crop_rgb", "box"]} for r in results
    ])
    with st.expander("Lihat sebagai tabel"):
        st.dataframe(table_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# MODE 1: Live Webcam (streamlit-webrtc)
# ---------------------------------------------------------------------------
if mode == "📷 Live Webcam":
    st.subheader("📷 Live Webcam")
    st.caption(
        "Klik START untuk mengaktifkan kamera. Bounding box, label etnis & "
        "emosi akan digambar langsung pada video."
    )

    class VideoProcessor:
        def __init__(self):
            self.detector = detector
            self.ethnicity_model = ethnicity_model
            self.emotion_model = emotion_model

            self.inference_interval = 0.2

            # Frame terbaru yang akan digunakan untuk inference
            self.latest_frame_rgb = None

            # Hasil inference terakhir
            self.last_results = []

            # Lock untuk mencegah race condition
            self.lock = threading.Lock()

            # Status inference
            self.inference_running = False
            
            # Variabel untuk menyimpan state rotasi
            self.cv_rotation = None

            # Thread inference
            self.inference_thread = threading.Thread(
                target=self._inference_loop,
                daemon=True
            )
            self.inference_thread.start()

        def _inference_loop(self):
            while True:

                time.sleep(self.inference_interval)

                # Ambil frame terbaru
                with self.lock:
                    if self.latest_frame_rgb is None:
                        continue

                    frame_rgb = self.latest_frame_rgb.copy()

                    # Jangan jalankan inference kalau inference sebelumnya
                    # masih berjalan
                    if self.inference_running:
                        continue

                    self.inference_running = True

                try:
                    _, results = run_pipeline_on_frame(
                        frame_rgb,
                        self.detector,
                        self.ethnicity_model,
                        self.emotion_model,
                    )

                    with self.lock:
                        self.last_results = results

                except Exception as e:
                    print(f"Inference error: {e}")
                    
                    # Auto-Reocvery: Jika graf mediapipe korup, buat ulang detektornya
                    with self.lock:
                        self.detector = get_face_detector()

                finally:
                    with self.lock:
                        self.inference_running = False
                        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # Frame kamera terbaru
            img_bgr = frame.to_ndarray(format="bgr24")
            
            if getattr(self, "cv_rotation", None) is not None:
                img_bgr = cv2.rotate(img_bgr, self.cv_rotation)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Simpan frame terbaru untuk inference thread dan ambil hasil terakhir
            with self.lock:
                self.latest_frame_rgb = img_rgb.copy()
                results = self.last_results.copy()

            # Frame yang akan ditampilkan ke layar
            annotated_bgr = img_bgr.copy()

            # --- TAMBAHKAN LOGIKA PENGGAMBARAN DISINI ---
            # Gambar bounding box dan label dari hasil inference terakhir
            for res in results:
                if "box" in res:
                    face_utils.draw_annotations(
                        annotated_bgr, 
                        res["box"],
                        ethnicity_text=f"{res['Etnis']} ({res['Confidence Etnis']})",
                        emotion_text=f"{res['Emosi']} ({res['Confidence Emosi']})",
                    )

            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")
    
    ctx = webrtc_streamer(
        key="roro-jonggrang-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        
        video_html_attrs = {
            "autoPlay": True,
            "controls": True,
            "style": {"width": "100%", "height": "100%", "objectFit": "cover"},
        }
    )
    
    if ctx.video_processor:
        # Kirim nilai rotasi dari sidebar ke object processor yang sedang berjalan
        ctx.video_processor.cv_rotation = cv_rotation_flag

    st.info(
        "Catatan: kalau video terasa lag di laptop yang kurang kuat, itu "
        "wajar karena 2 model deep learning dijalankan tiap frame untuk "
        "semua wajah yang terdeteksi. Untuk demo panggung, disarankan test "
        "dulu di device yang akan dipakai.",
        icon="💡",
    )


# ---------------------------------------------------------------------------
# MODE 2: Analisis Gambar (Upload / Kamera)
# ---------------------------------------------------------------------------
else:
    st.subheader("🖼️ Gambar")
    st.caption("Ambil foto atau upload gambar untuk melihat analisis etnis & emosi wajahmu!")
    
    # Membuat Tab agar UI lebih rapi
    tab_upload, tab_kamera = st.tabs(["📂 Upload Gambar", "📸 Ambil Foto Langsung"])
    
    img_buffer = None
    
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload foto (bisa berisi lebih dari satu wajah)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            img_buffer = uploaded_file

    with tab_kamera:
        camera_image = st.camera_input("Posisikan wajahmu di depan kamrea, lalu klik 'Take Photo'")
        if camera_image is not None:
            img_buffer = camera_image

    # Jika ada gambar yang masuk (entah dari upload atau kamera)
    if img_buffer is not None:
        try:
            # --- PERBAIKAN DI SINI (Konversi aman agar tidak kosong/tipe data valid) ---
            pil_image = Image.open(img_buffer).convert("RGB")
            frame_rgb = np.array(pil_image, dtype=np.uint8) # Pastikan tipe datanya uint8
            
            # Validasi tambahan: pastikan array gambar tidak kosong
            if frame_rgb.size == 0 or frame_rgb is None:
                st.error("Gagal membaca data gambar. File kosong atau rusak.")
                st.stop()
            # ------------------------------------------------------------------------

            with st.spinner("AI sedang mendeteksi dan menganalisis wajah..."):
                start = time.time()
                annotated_bgr, results = run_pipeline_on_frame(
                    frame_rgb, detector, ethnicity_model, emotion_model
                )
                elapsed = time.time() - start

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(frame_rgb, caption="Foto asli", use_container_width=True)
            with col2:
                st.image(annotated_rgb, caption="Hasil deteksi", use_container_width=True)

            st.caption(f"Waktu proses: {elapsed:.2f} detik")
            
            # Fitur tombol download
            result_pil = Image.fromarray(annotated_rgb)
            
            import io
            buf = io.BytesIO()
            result_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            
            st.markdown("---")
            col_dl1, col_dl2 = st.columns([2, 1])
            with col_dl1:
                st.success("🎉 Analisis selesai! Kamu bisa langsung mengunduh kartu hasil deteksi wajahmu di sini.")
            
            with col_dl2:
                st.download_button(
                    label="📥 Download Hasil Foto",
                    data=byte_im,
                    file_name="result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
                
            render_results_table(results)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
            
    else:
        st.info("Silakan upload gambar atau ambil foto menggunakan kamera untuk memulai deteksi.")