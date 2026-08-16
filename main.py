import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RVC Light Voice Changer")
        self.root.geometry("520x480")
        self.root.resizable(False, False)

        self.is_running = False
        self.model_path = tk.StringVar()
        self.hubert_path = tk.StringVar()
        self.pitch_shift = tk.IntVar(value=0)

        self.build_ui()

    def build_ui(self):
        # Header
        header = tk.Label(self.root, text="RVC Canlı Ses Dönüştürücü", font=('Segoe UI', 14, 'bold'))
        header.pack(pady=10)

        # 1. Model .pth Seçimi
        frame_pth = tk.LabelFrame(self.root, text=" Model Dosyası (.pth) ", font=('Segoe UI', 9, 'bold'))
        frame_pth.pack(fill="x", padx=15, pady=5)
        tk.Entry(frame_pth, textvariable=self.model_path, state="readonly").pack(side="left", fill="x", expand=True, padx=5, pady=5)
        tk.Button(frame_pth, text="Gözat...", command=self.select_model).pack(side="right", padx=5, pady=5)

        # 2. HuBERT Model Seçimi
        frame_hubert = tk.LabelFrame(self.root, text=" HuBERT Base Dosyası (.pt) ", font=('Segoe UI', 9, 'bold'))
        frame_hubert.pack(fill="x", padx=15, pady=5)
        tk.Entry(frame_hubert, textvariable=self.hubert_path, state="readonly").pack(side="left", fill="x", expand=True, padx=5, pady=5)
        tk.Button(frame_hubert, text="Gözat...", command=self.select_hubert).pack(side="right", padx=5, pady=5)

        # 3. Pitch (Ses Tonu Ayarı)
        frame_pitch = tk.LabelFrame(self.root, text=" Ses Tonu (Pitch Shift) ", font=('Segoe UI', 9, 'bold'))
        frame_pitch.pack(fill="x", padx=15, pady=5)
        tk.Scale(frame_pitch, from_=-12, to=12, orient="horizontal", variable=self.pitch_shift).pack(fill="x", padx=10, pady=5)

        # 4. Durum Etiketi
        self.lbl_status = tk.Label(self.root, text="Durum: Kapalı 🔴", font=('Segoe UI', 11, 'bold'), fg="red")
        self.lbl_status.pack(pady=15)

        # 5. AÇ / KAPA Butonu
        self.btn_toggle = tk.Button(self.root, text="SESİ BAŞLAT", bg="#4CAF50", fg="white", font=('Segoe UI', 12, 'bold'), height=2, command=self.toggle_engine)
        self.btn_toggle.pack(fill="x", padx=20, pady=5)

    def select_model(self):
        file = filedialog.askopenfilename(filetypes=[("RVC Model", "*.pth")])
        if file:
            self.model_path.set(file)

    def select_hubert(self):
        file = filedialog.askopenfilename(filetypes=[("HuBERT Model", "*.pt")])
        if file:
            self.hubert_path.set(file)

    def toggle_engine(self):
        if not self.is_running:
            # Doğrulama Kontrolleri
            if not self.model_path.get():
                messagebox.showwarning("Eksik Dosya", "Lütfen önce G_1840.pth model dosyasını seçin!")
                return
            if not os.path.exists(self.model_path.get()):
                messagebox.showerror("Hata", "Seçilen .pth dosyası bulunamadı!")
                return

            # Motoru Başlat
            self.is_running = True
            self.btn_toggle.config(text="SESİ DURDUR", bg="#f44336")
            self.lbl_status.config(text="Durum: Çalışıyor 🟢 (Canlı Aktarım)", fg="green")
            
            # Ses işleme döngüsünü arka planda (thread) başlat (Arayüz donmaması için)
            threading.Thread(target=self.audio_loop, daemon=True).start()
        else:
            # Motoru Durdur
            self.is_running = False
            self.btn_toggle.config(text="SESİ BAŞLAT", bg="#4CAF50")
            self.lbl_status.config(text="Durum: Kapalı 🔴", fg="red")

    def audio_loop(self):
        """Arka planda ses işleme döngüsü"""
        try:
            while self.is_running:
                # Buraya PyTorch / Sounddevice inference mantığı dahil olur
                pass
        except Exception as e:
            self.is_running = False
            self.root.after(0, lambda: messagebox.showerror("Ses Hatası", f"Ses motoru durdu: {str(e)}"))
            self.root.after(0, lambda: self.btn_toggle.config(text="SESİ BAŞLAT", bg="#4CAF50"))
            self.root.after(0, lambda: self.lbl_status.config(text="Durum: Hata Oluştu 🔴", fg="red"))

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChangerApp(root)
    root.mainloop()
            
