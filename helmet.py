{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 383
        },
        "id": "BH1lpSTiUlyX",
        "outputId": "68ca7642-fd40-4c91-d4b5-283ffbe3be35"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'ultralytics'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_1913/2975640639.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtime\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0multralytics\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mYOLO\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0;31m# Load model\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'ultralytics'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
      "source": [
        "import cv2\n",
        "import os\n",
        "import time\n",
        "from ultralytics import YOLO\n",
        "\n",
        "# Load model\n",
        "model = YOLO(\"best.pt\")\n",
        "\n",
        "# Folder penyimpanan pelanggaran\n",
        "os.makedirs(\"pelanggaran\", exist_ok=True)\n",
        "\n",
        "# Kamera (ganti ke path video jika ingin uji coba file video)\n",
        "cap = cv2.VideoCapture(0)\n",
        "\n",
        "# Agar tidak menyimpan gambar berkali-kali dalam waktu singkat\n",
        "last_save = 0\n",
        "\n",
        "while True:\n",
        "    ret, frame = cap.read()\n",
        "    if not ret:\n",
        "        print(\"Gagal mengambil gambar dari kamera atau video selesai.\")\n",
        "        break\n",
        "\n",
        "    # verbose=False menyembunyikan log bawaan YOLO di terminal agar lebih bersih\n",
        "    results = model(frame, conf=0.7, verbose=False)\n",
        "\n",
        "    for r in results:\n",
        "        for box in r.boxes:\n",
        "            cls = int(box.cls[0])\n",
        "            conf = float(box.conf[0])\n",
        "            x1, y1, x2, y2 = map(int, box.xyxy[0])\n",
        "            label = model.names[cls]\n",
        "\n",
        "            # Inisialisasi variabel deteksi\n",
        "            is_pelanggaran = False\n",
        "\n",
        "            # --- PERBAIKAN LOGIKA LABEL ---\n",
        "            # Sesuaikan nama kelas \"no-helmet\" atau \"head\" dengan hasil training best.pt milikmu\n",
        "            if label == \"helmet\":\n",
        "                warna = (0, 255, 0)\n",
        "                teks = f\"MENGGUNAKAN HELM {conf:.2f}\"\n",
        "            elif label == \"no-helmet\" or label == \"head\":\n",
        "                warna = (0, 0, 255)\n",
        "                teks = f\"TIDAK MENGGUNAKAN HELM {conf:.2f}\"\n",
        "                is_pelanggaran = True\n",
        "            else:\n",
        "                # Lewati jika mendeteksi objek lain seperti 'motor', 'orang', dll\n",
        "                continue\n",
        "\n",
        "            # --- PROSES SIMPAN PELANGGARAN ---\n",
        "            if is_pelanggaran:\n",
        "                if time.time() - last_save > 3:\n",
        "                    nama_file = f\"pelanggaran/pelanggaran_{int(time.time())}.jpg\"\n",
        "                    # Disarankan clone frame asli agar gambar yang disimpan bersih tanpa bounding box\n",
        "                    cv2.imwrite(nama_file, frame)\n",
        "                    print(\"Pelanggaran tersimpan:\", nama_file)\n",
        "                    last_save = time.time()\n",
        "\n",
        "            # Bounding box objek\n",
        "            cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)\n",
        "\n",
        "            # --- PERBAIKAN BACKGROUND TEKS DINAMIS ---\n",
        "            (text_w, text_h), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)\n",
        "            # Batasi koordinat Y agar kotak teks tidak keluar dari layar atas frame\n",
        "            y1_box = max(y1, 20)\n",
        "            cv2.rectangle(frame, (x1, y1_box - text_h - 10), (x1 + text_w + 10), y1_box), warna, -1)\n",
        "\n",
        "            # Gambar Tulisan\n",
        "            cv2.putText(\n",
        "                frame,\n",
        "                teks,\n",
        "                (x1 + 5, y1_box - 5),\n",
        "                cv2.FONT_HERSHEY_SIMPLEX,\n",
        "                0.5,              # Ukuran font sedikit dikecilkan agar proporsional\n",
        "                (255, 255, 255),\n",
        "                2\n",
        "            )\n",
        "\n",
        "    cv2.imshow(\"Sistem Deteksi Helm\", frame)\n",
        "\n",
        "    if cv2.waitKey(1) & 0xFF == ord('q'):\n",
        "        break\n",
        "\n",
        "cap.release()\n",
        "cv2.destroyAllWindows()"
      ]
    }
  ]
}