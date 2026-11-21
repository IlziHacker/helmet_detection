import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
from gtts import gTTS
import pygame

# =====================================================================
# 1. INITIALIZATION & SETUP
# =====================================================================
model = YOLO("best.pt")
os.makedirs("pelanggaran", exist_ok=True)
pygame.mixer.init()


# Fungsi cerdas untuk generate dan putar suara AI secara real-time
def putar_suara(teks_suara, filename):
    filepath = os.path.join("pelanggaran", filename)
    # Buat file audio jika belum ada
    if not os.path.exists(filepath):
        tts = gTTS(text=teks_suara, lang='id', slow=False)
        tts.save(filepath)

    # Putar Audio
    try:
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
    except Exception as e:
        print("Gagal memutar suara:", e)


print("\n[INFO] Menyiapkan Suara AI... Mohon tunggu sebentar...")
putar_suara("Sistem deteksi helm aktif", "aktif.mp3")

cap = cv2.VideoCapture(0)
last_save = 0
last_audio = 0  # Timer agar suara AI tidak tumpang tindih setiap detik
NAMA_JENDELA = "SISTEM DETEKSI HELM PREMIUM"

# =====================================================================
# 2. CORE LOOP DETEKSI
# =====================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.55, verbose=False)
    output = frame.copy()

    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_frame_ini = None

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:
            teks = f"MENGGUNAKAN HELM {conf:.2f}"
            warna = (0, 255, 0)  # Hijau
            status_frame_ini = "pakai"
        elif cls == 1:
            teks = f"TIDAK MENGGUNAKAN HELM {conf:.2f}"
            warna = (0, 0, 255)  # Merah
            status_frame_ini = "bolos"

            # FITUR AMBIL FOTO + WATERMARK TANGGAL WAKTU
            if time.time() - last_save > 5:  # Jeda 5 detik agar tidak spam foto
                foto_pelanggaran = frame.copy()

                # Tempel teks Tanggal & Waktu di pojok kiri bawah foto yang disimpan
                cv2.putText(
                    foto_pelanggaran,
                    f"Waktu Pelanggaran: {waktu_sekarang}",
                    (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                nama_file = datetime.now().strftime("pelanggaran/%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(nama_file, foto_pelanggaran)
                print(f"[TEREKAM] Bukti foto + tanggal berhasil disimpan: {nama_file}")
                last_save = time.time()

        # Menggambar Bounding Box dan Banner Teks
        cv2.rectangle(output, (x1, y1), (x2, y2), warna, 3)
        (text_w, text_h), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y1_box = max(y1, 25)
        cv2.rectangle(output, (x1, y1_box - text_h - 10), (x1 + text_w + 10, y1_box), (0, 0, 0), -1)
        cv2.putText(output, teks, (x1 + 5, y1_box - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)

    # 🔊 LOGIKA SUARA AI (Diberi jeda 4 detik agar suaranya selesai ngomong dulu)
    if time.time() - last_audio > 4:
        if status_frame_ini == "bolos":
            print("[AUDIO] Memperingatkan pelanggar...")
            putar_suara("Peringatan! Harap gunakan helm Anda.", "warning.mp3")
            last_audio = time.time()
        elif status_frame_ini == "pakai":
            print("[AUDIO] Mengucapkan terima kasih...")
            putar_suara("Terima kasih sudah menggunakan helm.", "thanks.mp3")
            last_audio = time.time()

    # Tampilkan teks waktu real-time juga di layar monitor webcam
    cv2.putText(output, waktu_sekarang, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(NAMA_JENDELA, output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(NAMA_JENDELA, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Sistem Premium Berhenti.")