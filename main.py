import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Light RVC Voice Changer")
        self.root.geometry("500x400")

        self.model_path = tk.StringVar()
        self.hubert_path = tk.StringVar()

        # Model .pth Seçimi
        tk.Label(root, text="RVC Model Dosyası (.pth):", font=('Arial', 10, 'bold')).pack(anchor="w", padx=10, pady=(10,0))
        frame_model = tk.Frame(root)
        frame_model.pack(fill="x", padx=10, pady=5)
        tk.Entry(frame_model, textvariable=self.model_path).pack(side="left", fill="x", expand=True)
        tk.Button(frame_model, text="Gözat...", command=self.select_model).pack(side="right", padx=5)

        # HuBERT Base Seçimi
        tk.Label(root, text="HuBERT Base Model (.pt):", font=('Arial', 10, 'bold')).pack(anchor="w", padx=10, pady=(10,0))
        frame_hubert = tk.Frame(root)
        frame_hubert.pack(fill="x", padx=10, pady=5)
        tk.Entry(frame_hubert, textvariable=self.hubert_path).pack(side="left", fill="x", expand=True)
        tk.Button(frame_hubert, text="Gözat...", command=self.select_hubert).pack(side="right", padx=5)

        # Başlat Butonu
        tk.Button(root, text="SES DÖNÜŞTÜRMEYİ BAŞLAT", bg="#4CAF50", fg="white", font=('Arial', 11, 'bold'), command=self.start_engine).pack(pady=30, fill="x", padx=20)

    def select_model(self):
        file = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if file:
            self.model_path.set(file)

    def select_hubert(self):
        file = filedialog.askopenfilename(filetypes=[("Base Model", "*.pt")])
        if file:
            self.hubert_path.set(file)

    def start_engine(self):
        if not self.model_path.get():
            messagebox.showerror("Hata", "Lütfen G_1840.pth model dosyasını seçin!")
            return
        messagebox.showinfo("Bilgi", "Motor başlatılıyor... Seçilen model: " + os.path.basename(self.model_path.get()))

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChangerApp(root)
    root.mainloop()

