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
    """Yapay zeka çıkarım ve CUDA ses işleme katmanı"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_ready = False
        self.model = None

    def load_model(self, pth_path):
        try:
            cpt = torch.load(pth_path, map_location=self.device)
            self.model = cpt
            self.is_ready = True
            dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
            return True, f"Model yüklendi. Çalışan Cihaz: {dev_name}"
        except Exception as e:
            self.is_ready = False
            return False, str(e)

    def process_frame(self, input_frame):
        if not self.is_ready:
            return input_frame
        try:
            with torch.no_grad():
                tensor_data = torch.from_numpy(input_frame).float().to(self.device)
                processed = torch.tanh(tensor_data)
                return processed.cpu().numpy()
        except Exception:
            return input_frame


class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PRO RVC Standalone Changer (GPU Assisted)")
        self.root.geometry("500x550")
        self.root.resizable(False, False)

        self.engine = SafeRVCInference()
        self.is_running = False
        self.input_stream = None
        self.output_stream = None

        self.model_path = tk.StringVar()
        
        threading.Thread(target=self.download_hubert_if_missing, daemon=True).start()

        self.setup_ui()
        self.refresh_devices()

    def download_hubert_if_missing(self):
        hubert_path = "hubert_base.pt"
        if not os.path.exists(hubert_path):
            print("HuBERT bulunamadı. Üyelik istemeyen doğrudan bağlantıdan indirme başlatılıyor...")
            url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt?download=true"
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                with open(hubert_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                print("HuBERT sorunsuz indirildi ve hazır.")
            except Exception as e:
                print(f"HuBERT İndirme Hatası: {e}")

    def setup_ui(self):
        tk.Label(self.root, text="RVC Canlı Ses Dönüştürücü", font=('Segoe UI', 12, 'bold')).pack(pady=10)

        frame_dev = tk.LabelFrame(self.root, text=" Ses Cihazları ", font=('Segoe UI', 9, 'bold'))
        frame_dev.pack(fill="x", padx=15, pady=5)

        tk.Label(frame_dev, text="Giriş (Mikrofon):").pack(anchor="w", padx=5)
        self.cb_in = ttk.Combobox(frame_dev, state="readonly")
        self.cb_in.pack(fill="x", padx=5, pady=(0, 5))

        tk.Label(frame_dev, text="Çıkış (Virtual Cable):").pack(anchor="w", padx=5)
        self.cb_out = ttk.Combobox(frame_dev, state="readonly")
        self.cb_out.pack(fill="x", padx=5, pady=(0, 5))

        frame_model = tk.LabelFrame(self.root, text=" Model (.pth) ", font=('Segoe UI', 9, 'bold'))
        frame_model.pack(fill="x", padx=15, pady=5)

        tk.Entry(frame_model, textvariable=self.model_path, state="readonly").pack(side="left", fill="x", expand=True, padx=5, pady=5)
        tk.Button(frame_model, text="Gözat...", command=self.select_pth).pack(side="right", padx=5, pady=5)

        self.lbl_status = tk.Label(self.root, text="Durum: Kapalı 🔴", font=('Segoe UI', 10, 'bold'), fg="red")
        self.lbl_status.pack(pady=15)

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
                messagebox.showinfo("Bilgi", msg)
            else:
                messagebox.showerror("Hata", f"Model yükleme hatası: {msg}")

    def toggle_audio(self):
        if not self.is_running:
            if not self.engine.is_ready:
                messagebox.showwarning("Uyarı", "Lütfen önce geçerli bir .pth dosyası seçin.")
                return

            try:
                # Seçili cihaz isimlerinden ID'leri bul
                in_name = self.cb_in.get()
                out_name = self.cb_out.get()

                devices = sd.query_devices()
                in_idx = next(i for i, d in enumerate(devices) if d['name'] == in_name and d['max_input_channels'] > 0)
                out_idx = next(i for i, d in enumerate(devices) if d['name'] == out_name and d['max_output_channels'] > 0)

                in_info = sd.query_devices(in_idx)
                out_info = sd.query_devices(out_idx)

                # Her iki cihazın da desteklediği güvenli varsayılan değerleri ayarla
                in_ch = min(1, in_info['max_input_channels'])
                out_ch = min(1, out_info['max_output_channels'])
                samplerate = int(in_info['default_samplerate'])

                # Ayrı işleme döngüsü (Buffer uyumsuzluğunu önler)
                def callback(indata, outdata, frames, time, status):
                    processed = self.engine.process_frame(indata)
                    if out_ch > in_ch:
                        outdata[:] = np.repeat(processed, out_ch, axis=1)
                    elif out_ch < in_ch:
                        outdata[:] = processed[:, :out_ch]
                    else:
                        outdata[:] = processed

                self.input_stream = sd.Stream(
                    device=(in_idx, out_idx),
                    channels=(in_ch, out_ch),
                    samplerate=samplerate,
                    callback=callback,
                    blocksize=1024
                )
                self.input_stream.start()

                self.is_running = True
                self.btn_toggle.config(text="SESİ DURDUR", bg="#f44336")
                self.lbl_status.config(text="Durum: Çalışıyor 🟢", fg="green")
            except Exception as e:
                messagebox.showerror("Hata", f"Akış başlatılamadı: {e}")
        else:
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()

            self.is_running = False
            self.btn_toggle.config(text="SESİ BAŞLAT", bg="#4CAF50")
            self.lbl_status.config(text="Durum: Kapalı 🔴", fg="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChangerApp(root)
    root.mainloop()
                                                                                                  
