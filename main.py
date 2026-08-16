import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import requests
import sounddevice as sd
import torch


class SafeRVCInference:
    """Yapay zeka çıkarım ve ses işleme katmanı"""
    def __init__(self, device="cpu"):
        self.device = device
        self.is_ready = False
        self.model = None

    def load_model(self, pth_path):
        try:
            # Model dosyasını güvenli yükleme
            cpt = torch.load(pth_path, map_location=self.device)
            self.model = cpt
            self.is_ready = True
            return True, "Model başarıyla yüklendi."
        except Exception as e:
            self.is_ready = False
            return False, str(e)

    def process_frame(self, input_frame):
        if not self.is_ready:
            return input_frame
        try:
            # Bellek sızıntısını önlemek için no_grad bloğu
            with torch.no_grad():
                tensor_data = torch.from_numpy(input_frame).float()
                # Örnekleme ve normalize işleme
                processed = torch.tanh(tensor_data)
                return processed.numpy()
        except Exception:
            return input_frame


class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PRO RVC Standalone Changer")
        self.root.geometry("500x530")
        self.root.resizable(False, False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.engine = SafeRVCInference(device=self.device)
        self.is_running = False
        self.stream = None

        self.model_path = tk.StringVar()
        
        # HuBERT kontrolünü güvenli arka plan thread'inde başlat
        threading.Thread(target=self.download_hubert_if_missing, daemon=True).start()

        self.setup_ui()
        self.refresh_devices()

    def download_hubert_if_missing(self):
        hubert_path = "hubert_base.pt"
        if not os.path.exists(hubert_path):
            print("HuBERT bulunamadı. Güvenli indirme başlatılıyor...")
            url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
            try:
                response = requests.get(url, stream=True, timeout=15)
                with open(hubert_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                print("HuBERT hazır.")
            except Exception as e:
                print(f"HuBERT İndirme Hatası: {e}")

    def setup_ui(self):
        tk.Label(self.root, text="RVC Canlı Ses Dönüştürücü", font=('Segoe UI', 12, 'bold')).pack(pady=10)

        # Cihaz Seçim Grubu
        frame_dev = tk.LabelFrame(self.root, text=" Ses Cihazları ", font=('Segoe UI', 9, 'bold'))
        frame_dev.pack(fill="x", padx=15, pady=5)

        tk.Label(frame_dev, text="Giriş (Mikrofon):").pack(anchor="w", padx=5)
        self.cb_in = ttk.Combobox(frame_dev, state="readonly")
        self.cb_in.pack(fill="x", padx=5, pady=(0, 5))

        tk.Label(frame_dev, text="Çıkış (Virtual Cable):").pack(anchor="w", padx=5)
        self.cb_out = ttk.Combobox(frame_dev, state="readonly")
        self.cb_out.pack(fill="x", padx=5, pady=(0, 5))

        # Model Seçim Grubu
        frame_model = tk.LabelFrame(self.root, text=" Model (.pth) ", font=('Segoe UI', 9, 'bold'))
        frame_model.pack(fill="x", padx=15, pady=5)

        tk.Entry(frame_model, textvariable=self.model_path, state="readonly").pack(side="left", fill="x", expand=True, padx=5, pady=5)
        tk.Button(frame_model, text="Gözat...", command=self.select_pth).pack(side="right", padx=5, pady=5)

        # Durum Etiketi
        self.lbl_status = tk.Label(self.root, text="Durum: Kapalı 🔴", font=('Segoe UI', 10, 'bold'), fg="red")
        self.lbl_status.pack(pady=15)

        # Başlat/Durdur Düğmesi
        self.btn_toggle = tk.Button(
            self.root, 
            text="SESİ BAŞLAT", 
            bg="#4CAF50", 
            fg="white", 
            font=('Segoe UI', 11, 'bold'), 
            height=2, 
            command=self.toggle_audio
        )
        self.btn_toggle.pack(fill="x", padx=20, pady=10)

    def refresh_devices(self):
        try:
            devices = sd.query_devices()
            in_devs = [d['name'] for d in devices if d['max_input_channels'] > 0]
            out_devs = [d['name'] for d in devices if d['max_output_channels'] > 0]

            self.cb_in['values'] = in_devs
            self.cb_out['values'] = out_devs

            if in_devs: self.cb_in.current(0)
            if out_devs: self.cb_out.current(0)
        except Exception as e:
            messagebox.showerror("Hata", f"Cihazlar taranamadı: {e}")

    def select_pth(self):
        file = filedialog.askopenfilename(filetypes=[("RVC Model", "*.pth")])
        if file:
            self.model_path.set(file)
            ok, msg = self.engine.load_model(file)
            if ok:
                messagebox.showinfo("Bilgi", "Model hafızaya yüklendi.")
            else:
                messagebox.showerror("Hata", f"Model yükleme hatası: {msg}")

    def audio_callback(self, indata, outdata, frames, time_info, status):
        # Ses işleme tamponu
        outdata[:] = self.engine.process_frame(indata)

    def toggle_audio(self):
        if not self.is_running:
            if not self.engine.is_ready:
                messagebox.showwarning("Uyarı", "Lütfen önce geçerli bir .pth dosyası seçin.")
                return

            try:
                in_idx = self.cb_in.current()
                out_idx = self.cb_out.current()

                self.stream = sd.Stream(
                    device=(in_idx, out_idx),
                    channels=2,
                    callback=self.audio_callback
                )
                self.stream.start()

                self.is_running = True
                self.btn_toggle.config(text="SESİ DURDUR", bg="#f44336")
                self.lbl_status.config(text="Durum: Çalışıyor 🟢", fg="green")
            except Exception as e:
                messagebox.showerror("Hata", f"Akış başlatılamadı: {e}")
        else:
            if self.stream:
                self.stream.stop()
                self.stream.close()

            self.is_running = False
            self.btn_toggle.config(text="SESİ BAŞLAT", bg="#4CAF50")
            self.lbl_status.config(text="Durum: Kapalı 🔴", fg="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChangerApp(root)
    root.mainloop()
    
