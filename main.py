"""
PRO RVC Standalone Changer (GPU Assisted)
==========================================
Gerçek zamanlıya yakın (chunk tabanlı) RVC ses dönüştürücü.

ÖNEMLİ - MİMARİ NOT:
Gerçek bir RVC dönüşümü (HuBERT içerik çıkarımı + F0/pitch tahmini + RVC
generator ağı) elle, sıfırdan yeniden yazılabilecek kadar basit bir şey
değildir; RVC-Project'in orijinal ağırlıkları ve mimarisiyle birebir
uyumlu olması gerekir. Bu yüzden burada, o mimariyi doğru şekilde implemente
eden ve bakımı yapılan resmi "rvc-python" (PyPI: rvc-python) paketi
kullanılıyor. Bu, `.pth` dosyasını gerçekten işleyen tek güvenilir yoldur.

Bu tasarımda ses, küçük parçalar (varsayılan 1 saniye) halinde toplanır,
her parça rvc-python ile dönüştürülür ve bir oynatma tamponuna eklenir.
Bu yüzden "örnek-örnek" değil, "parça-parça" gerçek zamanlıdır: toplam
gecikme yaklaşık olarak (parça süresi + o parçanın işlenme süresi) kadardır.
GPU'da bu genelde ~0.3-1.5 saniye, CPU'da çok daha fazla (birkaç saniye)
olabilir. Bu, ses kartı sürücü gecikmesi gibi donanım bağımlı bir durumdur
ve kodla "sıfıra" indirilemez.
"""

import os
import queue
import shutil
import sys
import tempfile
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    from rvc_python.infer import RVCInference
    RVC_IMPORT_ERROR = None
except Exception as _e:  # paket kurulu değilse uygulamanın açılmasını engelleme
    RVCInference = None
    RVC_IMPORT_ERROR = _e


def cuda_is_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class RVCEngine:
    """rvc-python üzerinden gerçek RVC modelini yükleyen/çalıştıran katman."""

    def __init__(self):
        self.device = "cuda:0" if cuda_is_available() else "cpu"
        self.rvc = None
        self.is_ready = False
        self.call_lock = threading.Lock()  # rvc-python nesnesi thread-safe değil

    def load_model(self, pth_path):
        if RVCInference is None:
            return False, (
                "rvc-python paketi kurulu değil.\n"
                f"Kurulum hatası: {RVC_IMPORT_ERROR}\n\n"
                "Kurmak için: pip install rvc-python"
            )
        try:
            with self.call_lock:
                self.rvc = RVCInference(device=self.device)
                self.rvc.load_model(pth_path)

                # Bilinen parametreleri ayarlamayı dene (rvc-python sürümüne
                # göre isim farklılık gösterebilir; bulunamazsa sessizce atla,
                # varsayılan değerlerle devam et).
                for attr, val in (
                    ("f0method", "rmvpe"),
                    ("index_rate", 0.5),
                    ("filter_radius", 3),
                    ("rms_mix_rate", 0.25),
                    ("protect", 0.33),
                ):
                    try:
                        setattr(self.rvc, attr, val)
                    except Exception:
                        pass

            self.is_ready = True
            dev_label = self.device
            if self.device.startswith("cuda"):
                try:
                    import torch
                    dev_label = torch.cuda.get_device_name(0)
                except Exception:
                    pass
            return True, f"Model yüklendi ({dev_label})"
        except Exception as e:
            self.is_ready = False
            self.rvc = None
            return False, f"{type(e).__name__}: {e}"

    def set_pitch(self, semitones):
        if self.rvc is None:
            return
        with self.call_lock:
            for attr in ("f0up_key", "pitch"):
                try:
                    setattr(self.rvc, attr, int(semitones))
                except Exception:
                    pass

    def convert_file(self, in_wav_path, out_wav_path):
        if not self.is_ready or self.rvc is None:
            raise RuntimeError("Model hazır değil.")
        with self.call_lock:
            self.rvc.infer_file(in_wav_path, out_wav_path)


class VoiceChangerApp:
    CHUNK_SECONDS = 1.0     # işlenecek ses parçası uzunluğu (sn)
    MAX_QUEUE = 4           # birikebilecek en fazla işlenmemiş parça (gecikmeyi sınırlar)

    def __init__(self, root):
        self.root = root
        self.root.title("PRO RVC Standalone Changer (GPU Assisted)")
        self.root.geometry("540x640")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.engine = RVCEngine()
        self.is_running = False

        self.input_stream = None
        self.output_stream = None

        self.model_path = tk.StringVar()
        self.pitch_var = tk.IntVar(value=0)

        self.work_queue = queue.Queue(maxsize=self.MAX_QUEUE)
        self.worker_thread = None
        self.worker_stop_event = threading.Event()

        self.capture_lock = threading.Lock()
        self.capture_accum = []       # bekleyen ham örnekler (mono, float32)
        self.capture_accum_len = 0
        self.capture_samplerate = None

        self.playback_lock = threading.Lock()
        self.playback_buffer = np.zeros((0, 2), dtype="float32")
        self.out_channels = 2

        self.temp_dir = tempfile.mkdtemp(prefix="rvc_rt_")

        self.setup_ui()
        self.refresh_devices()

        if RVCInference is None:
            self.set_status(f"rvc-python bulunamadı: {RVC_IMPORT_ERROR}", "red")

    # ------------------------------------------------------------------ UI
    def setup_ui(self):
        tk.Label(self.root, text="RVC Canlı Ses Dönüştürücü", font=("Segoe UI", 12, "bold")).pack(pady=10)

        frame_dev = tk.LabelFrame(self.root, text=" Ses Cihazları ", font=("Segoe UI", 9, "bold"))
        frame_dev.pack(fill="x", padx=15, pady=5)

        tk.Label(frame_dev, text="Giriş (Mikrofon):").pack(anchor="w", padx=5)
        self.cb_in = ttk.Combobox(frame_dev, state="readonly")
        self.cb_in.pack(fill="x", padx=5, pady=(0, 5))

        tk.Label(frame_dev, text="Çıkış (Virtual Cable):").pack(anchor="w", padx=5)
        self.cb_out = ttk.Combobox(frame_dev, state="readonly")
        self.cb_out.pack(fill="x", padx=5, pady=(0, 5))

        tk.Button(frame_dev, text="Cihazları Yenile", command=self.refresh_devices).pack(anchor="e", padx=5, pady=(0, 5))

        frame_model = tk.LabelFrame(self.root, text=" Model (.pth) ", font=("Segoe UI", 9, "bold"))
        frame_model.pack(fill="x", padx=15, pady=5)

        tk.Entry(frame_model, textvariable=self.model_path, state="readonly").pack(side="left", fill="x", expand=True, padx=5, pady=5)
        tk.Button(frame_model, text="Gözat...", command=self.select_pth).pack(side="right", padx=5, pady=5)

        frame_pitch = tk.LabelFrame(self.root, text=" Perde (Pitch, yarım ton) ", font=("Segoe UI", 9, "bold"))
        frame_pitch.pack(fill="x", padx=15, pady=5)

        self.lbl_pitch_val = tk.Label(frame_pitch, text="0")
        self.lbl_pitch_val.pack(side="right", padx=10)
        tk.Scale(
            frame_pitch, from_=-24, to=24, orient="horizontal", variable=self.pitch_var,
            command=self.on_pitch_change,
        ).pack(fill="x", padx=5, pady=5)

        frame_lat = tk.LabelFrame(self.root, text=" Gecikme / Parça Süresi ", font=("Segoe UI", 9, "bold"))
        frame_lat.pack(fill="x", padx=15, pady=5)
        tk.Label(
            frame_lat,
            text="Gerçek zamanlıya yakın çalışır; toplam gecikme ~ parça süresi\n"
                 "+ işlenme süresi kadardır (GPU'da daha kısa, CPU'da daha uzun).",
            justify="left", fg="#555555", font=("Segoe UI", 8),
        ).pack(anchor="w", padx=5, pady=5)

        self.lbl_status = tk.Label(self.root, text="Durum: Kapalı \U0001F534", font=("Segoe UI", 10, "bold"), fg="red")
        self.lbl_status.pack(pady=10)

        self.lbl_backlog = tk.Label(self.root, text="", font=("Segoe UI", 8), fg="#888888")
        self.lbl_backlog.pack()

        self.btn_toggle = tk.Button(
            self.root,
            text="SESİ BAŞLAT",
            bg="#4CAF50",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            height=2,
            command=self.toggle_audio,
        )
        self.btn_toggle.pack(fill="x", padx=20, pady=10)

    def set_status(self, text, color="black"):
        def _update():
            self.lbl_status.config(text=f"Durum: {text}", fg=color)
        self.root.after(0, _update)

    def on_pitch_change(self, _value):
        val = self.pitch_var.get()
        self.lbl_pitch_val.config(text=str(val))
        self.engine.set_pitch(val)

    # -------------------------------------------------------------- Cihaz
    def refresh_devices(self):
        try:
            all_devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            in_devs = []
            out_devs = []

            for idx, d in enumerate(all_devices):
                api_name = hostapis[d["hostapi"]]["name"]
                label = f"[{idx}] ({api_name}) {d['name']}"
                if d["max_input_channels"] > 0:
                    in_devs.append(label)
                if d["max_output_channels"] > 0:
                    out_devs.append(label)

            self.cb_in["values"] = in_devs
            self.cb_out["values"] = out_devs

            if in_devs:
                self.cb_in.current(0)
            if out_devs:
                self.cb_out.current(0)
        except Exception as e:
            messagebox.showerror("Hata", f"Cihazlar taranamadı: {e}")

    def select_pth(self):
        file = filedialog.askopenfilename(filetypes=[("RVC Model", "*.pth")])
        if not file:
            return
        self.model_path.set(file)
        self.set_status("Model yükleniyor... (ilk seferde gerekli bileşenler indirilebilir)", "orange")

        def _load():
            ok, msg = self.engine.load_model(file)
            if ok:
                self.root.after(0, lambda: messagebox.showinfo("Bilgi", msg))
                self.set_status("Kapalı \U0001F534 (model hazır)", "red")
                self.engine.set_pitch(self.pitch_var.get())
            else:
                self.root.after(0, lambda: messagebox.showerror("Hata", f"Model yükleme hatası:\n{msg}"))
                self.set_status("Model yüklenemedi \u26A0", "red")

        threading.Thread(target=_load, daemon=True).start()

    # ---------------------------------------------------------- Ses akışı
    def in_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Giriş Durumu: {status}", file=sys.stderr)

        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        with self.capture_lock:
            self.capture_accum.append(mono)
            self.capture_accum_len += len(mono)

            chunk_frames = int(self.capture_samplerate * self.CHUNK_SECONDS)
            if self.capture_accum_len >= chunk_frames:
                full = np.concatenate(self.capture_accum)
                ready = full[:chunk_frames]
                remainder = full[chunk_frames:]
                self.capture_accum = [remainder] if len(remainder) else []
                self.capture_accum_len = len(remainder)

                try:
                    self.work_queue.put_nowait((ready, self.capture_samplerate))
                except queue.Full:
                    # İşlenme hızı yetişemiyor; en eski parçayı atıp devam et.
                    try:
                        self.work_queue.get_nowait()
                        self.work_queue.put_nowait((ready, self.capture_samplerate))
                    except queue.Empty:
                        pass

    def out_callback(self, outdata, frames, time_info, status):
        if status:
            print(f"Çıkış Durumu: {status}", file=sys.stderr)

        with self.playback_lock:
            available = self.playback_buffer.shape[0]
            take = min(frames, available)
            if take > 0:
                outdata[:take] = self.playback_buffer[:take]
                self.playback_buffer = self.playback_buffer[take:]
            if take < frames:
                outdata[take:] = 0.0

    def worker_loop(self):
        while not self.worker_stop_event.is_set():
            try:
                mono, in_sr = self.work_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            in_path = os.path.join(self.temp_dir, "chunk_in.wav")
            out_path = os.path.join(self.temp_dir, "chunk_out.wav")

            try:
                sf.write(in_path, mono, in_sr, subtype="PCM_16")
                self.engine.convert_file(in_path, out_path)
                out_audio, out_sr = sf.read(out_path, dtype="float32", always_2d=True)

                out_audio = self._match_channels(out_audio, self.out_channels)
                out_audio = self._resample_if_needed(out_audio, out_sr, self.output_samplerate)

                with self.playback_lock:
                    self.playback_buffer = np.concatenate([self.playback_buffer, out_audio], axis=0)
                    # Aşırı gecikme birikmesini önlemek için tamponu sınırla.
                    max_samples = int(self.output_samplerate * 3.0)
                    if self.playback_buffer.shape[0] > max_samples:
                        self.playback_buffer = self.playback_buffer[-max_samples:]

                self.root.after(0, lambda: self.lbl_backlog.config(
                    text=f"İşlem kuyruğu: {self.work_queue.qsize()} parça bekliyor"
                ))
            except Exception:
                traceback.print_exc()
                self.set_status("İşleme hatası (konsola bakın) \u26A0", "red")
            finally:
                for p in (in_path, out_path):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    @staticmethod
    def _match_channels(audio, out_channels):
        # audio: (frames, ch)
        cur_ch = audio.shape[1]
        if cur_ch == out_channels:
            return audio
        if cur_ch == 1 and out_channels > 1:
            return np.repeat(audio, out_channels, axis=1)
        if cur_ch > out_channels:
            return audio[:, :out_channels]
        # cur_ch bir şekilde daha az ama 1 değilse: son kanalı tekrarla
        pad = np.repeat(audio[:, -1:], out_channels - cur_ch, axis=1)
        return np.concatenate([audio, pad], axis=1)

    @staticmethod
    def _resample_if_needed(audio, src_sr, dst_sr):
        if src_sr == dst_sr or audio.shape[0] == 0:
            return audio.astype("float32", copy=False)
        n_src = audio.shape[0]
        n_dst = int(round(n_src * dst_sr / src_sr))
        if n_dst <= 0:
            return np.zeros((0, audio.shape[1]), dtype="float32")
        x_src = np.linspace(0.0, 1.0, num=n_src, endpoint=False)
        x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
        out = np.empty((n_dst, audio.shape[1]), dtype="float32")
        for ch in range(audio.shape[1]):
            out[:, ch] = np.interp(x_dst, x_src, audio[:, ch])
        return out

    # ------------------------------------------------------------- Toggle
    def toggle_audio(self):
        if not self.is_running:
            self.start_audio()
        else:
            self.stop_audio()

    def start_audio(self):
        if not self.engine.is_ready:
            messagebox.showwarning("Uyarı", "Lütfen önce geçerli bir .pth dosyası seçin ve yüklenmesini bekleyin.")
            return
        if not self.cb_in.get() or not self.cb_out.get():
            messagebox.showwarning("Uyarı", "Lütfen giriş ve çıkış cihazlarını seçin.")
            return

        try:
            in_idx = int(self.cb_in.get().split("]")[0].replace("[", ""))
            out_idx = int(self.cb_out.get().split("]")[0].replace("[", ""))

            in_info = sd.query_devices(in_idx)
            out_info = sd.query_devices(out_idx)

            in_samplerate = int(in_info["default_samplerate"])
            out_samplerate = int(out_info["default_samplerate"])
            blocksize = 1024

            in_channels = min(1, in_info["max_input_channels"])
            out_channels = max(1, min(2, out_info["max_output_channels"]))

            if in_channels < 1:
                raise RuntimeError("Seçilen giriş cihazının giriş kanalı yok.")
            if out_channels < 1:
                raise RuntimeError("Seçilen çıkış cihazının çıkış kanalı yok.")

            self.capture_samplerate = in_samplerate
            self.capture_accum = []
            self.capture_accum_len = 0

            self.output_samplerate = out_samplerate
            self.out_channels = out_channels
            with self.playback_lock:
                self.playback_buffer = np.zeros((0, out_channels), dtype="float32")

            while not self.work_queue.empty():
                try:
                    self.work_queue.get_nowait()
                except queue.Empty:
                    break

            self.worker_stop_event.clear()
            self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
            self.worker_thread.start()

            self.input_stream = sd.InputStream(
                device=in_idx,
                channels=in_channels,
                samplerate=in_samplerate,
                callback=self.in_callback,
                blocksize=blocksize,
            )
            self.output_stream = sd.OutputStream(
                device=out_idx,
                channels=out_channels,
                samplerate=out_samplerate,
                callback=self.out_callback,
                blocksize=blocksize,
            )

            self.input_stream.start()
            self.output_stream.start()

            self.is_running = True
            self.btn_toggle.config(text="SESİ DURDUR", bg="#f44336")
            self.set_status("Çalışıyor \U0001F7E2", "green")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Hata", f"Akış başlatılamadı: {e}")
            self.stop_audio(silent=True)

    def stop_audio(self, silent=False):
        self.worker_stop_event.set()

        for stream_attr in ("input_stream", "output_stream"):
            stream = getattr(self, stream_attr)
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                setattr(self, stream_attr, None)

        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

        while not self.work_queue.empty():
            try:
                self.work_queue.get_nowait()
            except queue.Empty:
                break

        with self.playback_lock:
            self.playback_buffer = np.zeros((0, self.out_channels), dtype="float32")

        self.is_running = False
        self.btn_toggle.config(text="SESİ BAŞLAT", bg="#4CAF50")
        if not silent:
            self.set_status("Kapalı \U0001F534", "red")
        self.lbl_backlog.config(text="")

    def on_close(self):
        try:
            self.stop_audio(silent=True)
        finally:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChangerApp(root)
    root.mainloop()
