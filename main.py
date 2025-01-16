import sys
import os
import json
import threading
import time
import queue
import datetime
import wave
import shutil
import numpy as np
import cv2
import textwrap

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QFileDialog, QComboBox, QTextEdit, QSpinBox, QDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage

import pyaudio
import mss  # ← mssを使用
from PIL import Image as PILImage  # mss から取り出した画像を PIL へ変換

try:
    import whisper
except ImportError:
    print("openai-whisper がインストールされていません。 pip install openai-whisper を実行してください。")
    sys.exit(1)

# ReportLab / Platypus 関連
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, PageBreak, Spacer, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


class StereoMixerRecorder(threading.Thread):
    """
    ステレオミキサーデバイス (または WASAPIループバック/ステレオミックス) 1つで録音。
    """
    def __init__(self, device_index, log_queue, chunk=1024, rate=44100, channels=2):
        super().__init__()
        self.device_index = device_index
        self.log_queue = log_queue
        self.chunk = chunk
        self.rate = rate
        self.channels = channels
        
        self.p = None
        self.stream = None
        self.frames = []
        
        self.is_recording = False
        self.lock = threading.Lock()
    
    def run(self):
        self.p = pyaudio.PyAudio()
        try:
            self.log_queue.put("StereoMixerRecorder: 録音開始")
            
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk
            )
            
            self.is_recording = True
            while self.is_recording:
                data = self.stream.read(self.chunk)
                with self.lock:
                    self.frames.append(data)
        except Exception as e:
            self.log_queue.put(f"StereoMixerRecorderエラー: {e}")
        finally:
            self.log_queue.put("StereoMixerRecorder: スレッド終了")
    
    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
    
    def pop_frames(self):
        with self.lock:
            frames_copy = self.frames[:]
            self.frames = []
        return frames_copy


class TranscriptionThread(threading.Thread):
    """
    Whisperで文字起こしを別スレッドで実行
    """
    def __init__(self, whisper_model, wav_path, txt_path, ffmpeg_exists, log_queue):
        super().__init__()
        self.whisper_model = whisper_model
        self.wav_path = wav_path
        self.txt_path = txt_path
        self.ffmpeg_exists = ffmpeg_exists
        self.log_queue = log_queue
    
    def run(self):
        self.log_queue.put(f"文字起こしスレッド開始: {self.wav_path}")
        
        if not self.ffmpeg_exists:
            text_content = "[ffmpegが見つかりません。文字起こしをスキップしました。]"
        else:
            try:
                result = self.whisper_model.transcribe(self.wav_path)
                text_content = result["text"]
            except Exception as e:
                text_content = f"Whisperエラー: {e}"
                self.log_queue.put(text_content)
        
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        
        self.log_queue.put(f"文字起こし完了 -> {self.txt_path}")


class AutoCaptureThread(threading.Thread):
    """
    mss を使った自動キャプチャスレッド (画面変化検知)
    """
    def __init__(self, save_dir, threshold, monitor_index, capture_callback, log_queue):
        super().__init__()
        self.save_dir = save_dir
        self.threshold = threshold
        self.monitor_index = monitor_index  # mss.monitors の index
        self.capture_callback = capture_callback
        self.log_queue = log_queue
        
        self.running = True
        self.paused = False
        self.prev_frame = None

    def run(self):
        import mss
        self.log_queue.put("自動キャプチャスレッド: 開始")

        with mss.mss() as sct:
            while self.running:
                if self.paused:
                    time.sleep(0.5)
                    continue
                
                mon = sct.monitors[self.monitor_index]
                
                # スクリーンショット
                img_sct = sct.grab(mon)
                frame = np.array(img_sct)[:,:,:3]  # BGRA→BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                if self.prev_frame is not None:
                    diff = cv2.absdiff(self.prev_frame, frame)
                    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    change_value = np.sum(gray)
                    
                    if change_value > self.threshold:
                        # キャプチャ実行
                        self.capture_callback(is_auto=True)
                        self.log_queue.put(f"画面変化検知: {change_value}, キャプチャ実行")
                
                self.prev_frame = frame
                time.sleep(1)
        
        self.log_queue.put("自動キャプチャスレッド: 終了")
    
    def pause_capture(self):
        self.paused = True
        self.log_queue.put("自動キャプチャ: 一時停止")
    
    def resume_capture(self):
        self.paused = False
        self.log_queue.put("自動キャプチャ: 再開")
    
    def stop_capture(self):
        self.running = False


class PDFPlatypusGenerationThread(threading.Thread):
    """
    ReportLabのPlatypusを用いて、画像と文字起こしを丁寧に折り返してPDF出力するスレッド。
    """
    def __init__(self, save_dir, log_queue):
        super().__init__()
        self.save_dir = save_dir
        self.log_queue = log_queue
        self.pdf_filename = os.path.join(self.save_dir, "result.pdf")
        
        # 日本語フォント登録 (Windowsなら msゴシックや BIZ-UDGothicR など)
        try:
            pdfmetrics.registerFont(TTFont('NotoSans', 'C:/Windows/Fonts/BIZ-UDGothicR.ttc'))
        except Exception as e:
            self.log_queue.put(f"フォント登録エラー: {e} (文字化けの可能性あり)")

        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            name='Japanese',
            fontName='NotoSans',
            fontSize=12,
            leading=15,
        ))
        self.styles.add(ParagraphStyle(
            name='Heading',
            fontName='NotoSans',
            fontSize=12,
            leading=15,
            textColor='#333333',
            spaceAfter=5,
            bulletIndent=0,
            leftIndent=0,
        ))
    
    def run(self):
        self.log_queue.put("PDF生成スレッド開始(Platypus版)")
        
        # 画像を作成日時順にソート
        image_files = [f for f in os.listdir(self.save_dir) if f.lower().endswith(".png")]
        image_files.sort(key=lambda x: os.path.getctime(os.path.join(self.save_dir, x)))
        
        from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, PageBreak, Spacer, Image
        
        story = []
        for i, img_file in enumerate(image_files, start=1):
            base_name, _ = os.path.splitext(img_file)
            txt_file = f"{base_name}.txt"
            
            # 見出し
            heading = Paragraph(f"画像名: {img_file}", self.styles["Heading"])
            story.append(heading)
            story.append(Spacer(1, 0.2*cm))
            
            # 画像を指定幅で挿入
            from PIL import Image as PILImage
            img_path = os.path.join(self.save_dir, img_file)
            try:
                MAX_WIDTH = 20*cm
                pil_img = PILImage.open(img_path)
                orig_w, orig_h = pil_img.size
                if orig_w>0 and orig_h>0:
                    aspect = orig_h / orig_w
                    target_width = MAX_WIDTH
                    target_height = target_width * aspect
                    flow_img = Image(img_path, width=target_width, height=target_height)
                    story.append(flow_img)
                else:
                    story.append(Paragraph("<font color=red>画像破損</font>", self.styles["Japanese"]))
            except Exception as e:
                story.append(Paragraph(f"<font color=red>画像読み込み失敗: {e}</font>", self.styles["Japanese"]))

            story.append(Spacer(1, 0.5*cm))
            
            # 文字起こし
            txt_path = os.path.join(self.save_dir, txt_file)
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            else:
                text_content = f"{img_file} に対応する文字起こしがありません。"

            text_para = Paragraph(text_content, self.styles["Japanese"])
            story.append(text_para)
            
            story.append(PageBreak())
        
        self.build_document(story)
        self.log_queue.put(f"PDF生成完了: {self.pdf_filename}")
    
    def build_document(self, story):
        doc = BaseDocTemplate(self.pdf_filename, pagesize=A4)
        
        margin_left = 2*cm
        margin_right = 2*cm
        margin_top = 2*cm
        margin_bottom = 2*cm
        
        frame = Frame(
            margin_left,
            margin_bottom,
            A4[0] - margin_left - margin_right,
            A4[1] - margin_top - margin_bottom,
            showBoundary=0
        )
        
        def on_page(canvas, doc):
            page_num = doc.page
            canvas.setFont("NotoSans", 8)
            canvas.drawRightString(A4[0] - margin_right, 1*cm, f"{page_num}")
        
        page_template = PageTemplate(id='Main', frames=[frame], onPage=on_page)
        doc.addPageTemplates([page_template])
        doc.build(story)


class MirroringWindow(QDialog):
    """
    MSSを使って選択したモニタをリアルタイムでプレビュー表示するウィンドウ。
    """
    def __init__(self, get_monitor_index_callback, parent=None):
        """
        Args:
            get_monitor_index_callback: 呼び出すと現在選択されているモニタのIndex (mss.monitorsのインデックス) を返す関数
        """
        super().__init__(parent)
        self.setWindowTitle("画面ミラー (MSS)")
        self.setFixedSize(400, 300)

        self.get_monitor_index_callback = get_monitor_index_callback
        
        # ラベルに表示
        self.label = QLabel("プレビュー中...", self)
        self.label.setGeometry(0, 0, 400, 300)
        self.label.setAlignment(Qt.AlignCenter)

        # MSSオブジェクトを開く (closeEvent でクローズ)
        import mss
        self.sct = mss.mss()

        # タイマーで更新
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_mirror)
        self.timer.start(300)  # 0.3秒ごと
    
    def update_mirror(self):
        """
        MSSでモニタをキャプチャ → QLabelに表示
        """
        mon_idx = self.get_monitor_index_callback()
        if mon_idx is None:
            return
        
        # mss.monitors[0]は全画面、1,2..が個別モニタ
        if mon_idx < 0 or mon_idx >= len(self.sct.monitors):
            return
        
        mon = self.sct.monitors[mon_idx]
        
        # 画面取得
        img_sct = self.sct.grab(mon)
        frame = np.array(img_sct)  # BGRA
        # BGRA → BGR
        frame = frame[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # OpenCV(numpy配列) → QImage → QPixmap
        qimg = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.shape[1]*3,
            QImage.Format_BGR888
        )
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.label.width(),
            self.label.height(),
            Qt.KeepAspectRatio
        )
        self.label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        # ウィンドウを閉じるときに sct をクローズ
        if self.sct:
            self.sct.close()
        self.timer.stop()
        super().closeEvent(event)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("スライドキャプチャツール")
        self.resize(1000, 750)

        self.log_queue = queue.Queue()
        
        # 設定ファイル
        self.config_path = os.path.join(os.getcwd(), "settings.json")
        self.config_data = self.load_config()  # 前回設定復元

        self.save_dir = self.config_data.get("save_dir", os.path.join(os.getcwd(), "captures"))
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.whisper_model = whisper.load_model("base")
        self.ffmpeg_exists = bool(shutil.which("ffmpeg"))
        if not self.ffmpeg_exists:
            self.log_queue.put("警告: ffmpeg が見つかりません。")
        
        # 連番管理 (フォルダ内走査して max+1 とする例)
        self.manual_capture_index = 0
        self.auto_capture_index = 0
        self._init_capture_indices_from_folder()

        self.last_capture_prefix = None
        self.auto_capture_thread = None
        self.audio_recorder = None
        self.transcription_threads = []

        self.mirroring_window = None

        # GUI
        self.init_ui()
        self.restore_settings_from_config()  # UI反映

        # ログ更新タイマー
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_log)
        self.log_timer.start(500)

    # =========================================
    # 設定ファイル (JSON)
    # =========================================
    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
                return {}
        else:
            return {}

    def save_config(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            self.log_queue.put(f"設定ファイルに保存しました: {self.config_path}")
        except Exception as e:
            self.log_queue.put(f"設定ファイル保存エラー: {e}")

    def store_settings_to_config(self):
        # 保存先フォルダ
        self.config_data["save_dir"] = self.save_dir
        # モニタ選択
        self.config_data["monitor_idx"] = self.monitor_combo.currentIndex()
        # しきい値
        self.config_data["threshold"] = self.threshold_spin.value()
        # 入力デバイス
        self.config_data["input_device_idx"] = self.input_device_combo.currentIndex()
        
        self.save_config()

    def restore_settings_from_config(self):
        # save_dir は __init__ で読み込み済み
        if "monitor_idx" in self.config_data:
            idx = self.config_data["monitor_idx"]
            if 0 <= idx < self.monitor_combo.count():
                self.monitor_combo.setCurrentIndex(idx)
        if "threshold" in self.config_data:
            self.threshold_spin.setValue(self.config_data["threshold"])
        if "input_device_idx" in self.config_data:
            iidx = self.config_data["input_device_idx"]
            if 0 <= iidx < self.input_device_combo.count():
                self.input_device_combo.setCurrentIndex(iidx)

    def closeEvent(self, event):
        self.store_settings_to_config()
        super().closeEvent(event)

    # =========================================
    # フォルダ内走査で連番を続きから
    # =========================================
    def _init_capture_indices_from_folder(self):
        max_manual = 0
        max_auto = 0
        files = os.listdir(self.save_dir)
        for f in files:
            if f.startswith("manualcap_") and f.lower().endswith(".png"):
                num_str = f.replace("manualcap_", "").replace(".png", "")
                if num_str.isdigit():
                    max_manual = max(max_manual, int(num_str))
            if f.startswith("autocap_") and f.lower().endswith(".png"):
                num_str = f.replace("autocap_", "").replace(".png", "")
                if num_str.isdigit():
                    max_auto = max(max_auto, int(num_str))
        self.manual_capture_index = max_manual + 1
        self.auto_capture_index = max_auto + 1

    # =========================================
    # GUI
    # =========================================
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 保存先フォルダ
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel(f"保存先: {self.save_dir}")
        self.folder_button = QPushButton("保存先選択")
        self.folder_button.clicked.connect(self.select_save_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_button)
        main_layout.addLayout(folder_layout)

        # MSSモニタ一覧を取得しComboBoxへ
        self.monitor_combo = QComboBox()
        self.mss_monitors = []
        self.fetch_mss_monitors()

        mon_layout = QHBoxLayout()
        mon_layout.addWidget(QLabel("モニタ:"))
        mon_layout.addWidget(self.monitor_combo)
        main_layout.addLayout(mon_layout)

        # しきい値
        thr_layout = QHBoxLayout()
        thr_label = QLabel("画面変化の閾値:")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 999999999)
        self.threshold_spin.setValue(500000)
        thr_layout.addWidget(thr_label)
        thr_layout.addWidget(self.threshold_spin)
        main_layout.addLayout(thr_layout)

        # 音声デバイス
        audio_layout = QHBoxLayout()
        self.input_device_combo = QComboBox()
        self.set_audio_device_list()
        audio_layout.addWidget(QLabel("入力デバイス:"))
        audio_layout.addWidget(self.input_device_combo)
        main_layout.addLayout(audio_layout)

        # ボタン群
        btn_layout = QHBoxLayout()

        self.auto_start_btn = QPushButton("自動キャプチャ開始")
        self.auto_start_btn.clicked.connect(self.start_auto_capture)
        btn_layout.addWidget(self.auto_start_btn)

        self.auto_pause_btn = QPushButton("一時停止/再開")
        self.auto_pause_btn.clicked.connect(self.toggle_pause_auto_capture)
        btn_layout.addWidget(self.auto_pause_btn)

        self.auto_stop_btn = QPushButton("停止")
        self.auto_stop_btn.clicked.connect(self.stop_auto_capture)
        btn_layout.addWidget(self.auto_stop_btn)

        self.manual_cap_btn = QPushButton("手動キャプチャ")
        self.manual_cap_btn.clicked.connect(lambda: self.capture_and_transcribe(is_auto=False))
        btn_layout.addWidget(self.manual_cap_btn)

        self.audio_start_btn = QPushButton("録音開始")
        self.audio_start_btn.clicked.connect(self.start_audio_recording)
        btn_layout.addWidget(self.audio_start_btn)

        self.audio_stop_btn = QPushButton("録音停止")
        self.audio_stop_btn.clicked.connect(self.stop_audio_recording)
        btn_layout.addWidget(self.audio_stop_btn)

        self.pdf_btn = QPushButton("PDF生成")
        self.pdf_btn.clicked.connect(self.generate_pdf)
        btn_layout.addWidget(self.pdf_btn)

        self.mirror_btn = QPushButton("画面ミラー")
        self.mirror_btn.clicked.connect(self.open_mirroring_window)
        btn_layout.addWidget(self.mirror_btn)

        main_layout.addLayout(btn_layout)

        # ログ表示
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

    def fetch_mss_monitors(self):
        """
        mssを使ってモニタ一覧を取得し、コンボボックスに表示
        """
        self.monitor_combo.clear()
        self.mss_monitors = []
        import mss
        with mss.mss() as sct:
            for i, mon in enumerate(sct.monitors):
                desc = f"モニタ{i}: (left={mon['left']}, top={mon['top']}, w={mon['width']}, h={mon['height']})"
                self.monitor_combo.addItem(desc, i)
                self.mss_monitors.append(mon)

    def set_audio_device_list(self):
        p = pyaudio.PyAudio()
        dev_count = p.get_device_count()
        for i in range(dev_count):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels",0)>0:
                name = info["name"]
                self.input_device_combo.addItem(name, i)
        p.terminate()

    # =========================================
    # ボタンハンドラ
    # =========================================
    def select_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "保存先選択", self.save_dir)
        if folder:
            self.save_dir = folder
            os.makedirs(self.save_dir, exist_ok=True)
            self.folder_label.setText(f"保存先: {self.save_dir}")
            self.log_queue.put(f"保存先フォルダ変更: {folder}")
            self._init_capture_indices_from_folder()

    def start_auto_capture(self):
        # いきなり1枚撮影
        self.capture_and_transcribe(is_auto=True)

        if self.auto_capture_thread and self.auto_capture_thread.is_alive():
            self.log_queue.put("自動キャプチャは既に実行中です。")
            return
        
        threshold_val = self.threshold_spin.value()
        mon_idx = self.monitor_combo.currentData()

        self.auto_capture_thread = AutoCaptureThread(
            save_dir=self.save_dir,
            threshold=threshold_val,
            monitor_index=mon_idx,
            capture_callback=self.capture_and_transcribe,
            log_queue=self.log_queue
        )
        self.auto_capture_thread.start()
        self.log_queue.put("自動キャプチャ開始。")

    def toggle_pause_auto_capture(self):
        if self.auto_capture_thread and self.auto_capture_thread.is_alive():
            if not self.auto_capture_thread.paused:
                self.auto_capture_thread.pause_capture()
            else:
                self.auto_capture_thread.resume_capture()
        else:
            self.log_queue.put("自動キャプチャは開始されていません。")

    def stop_auto_capture(self):
        if self.auto_capture_thread:
            self.auto_capture_thread.stop_capture()
            self.auto_capture_thread = None
            self.log_queue.put("自動キャプチャ停止。")
        else:
            self.log_queue.put("自動キャプチャは実行中ではありません。")

    def capture_and_transcribe(self, is_auto=False):
        # 1) 前回キャプチャに対する音声処理
        if self.last_capture_prefix is not None:
            self.handle_audio_for_prefix(self.last_capture_prefix)

        # 2) 今回の画像を mss でキャプチャ
        if is_auto:
            prefix = f"autocap_{self.auto_capture_index}"
            self.auto_capture_index += 1
        else:
            prefix = f"manualcap_{self.manual_capture_index}"
            self.manual_capture_index += 1

        mon_idx = self.monitor_combo.currentData()
        import mss
        with mss.mss() as sct:
            mon = sct.monitors[mon_idx]
            img_sct = sct.grab(mon)
            # mss -> PIL
            img_pil = PILImage.frombytes("RGB", img_sct.size, img_sct.bgra, "raw", "BGRX")
            
            filename = f"{prefix}.png"
            filepath = os.path.join(self.save_dir, filename)
            img_pil.save(filepath)

        self.log_queue.put(f"{'自動' if is_auto else '手動'}キャプチャ保存: {filepath}")

        self.last_capture_prefix = prefix

    def handle_audio_for_prefix(self, prefix):
        # 録音していなければスキップ
        if not self.audio_recorder or not self.audio_recorder.is_recording:
            self.log_queue.put(f"{prefix}: 音声録音なし。")
            return
        frames = self.audio_recorder.pop_frames()
        if not frames:
            self.log_queue.put(f"{prefix}: 音声フレーム空。")
            return
        
        # WAV保存
        import wave
        p = pyaudio.PyAudio()
        wav_path = os.path.join(self.save_dir, f"{prefix}.wav")
        wf = wave.open(wav_path, 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
        wf.close()
        p.terminate()

        self.log_queue.put(f"WAV保存完了: {wav_path}")

        # Whisperで文字起こし
        txt_path = os.path.join(self.save_dir, f"{prefix}.txt")
        try:
            result = self.whisper_model.transcribe(wav_path)
            text_content = result["text"]
        except Exception as e:
            text_content = f"Whisperエラー: {e}"
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        self.log_queue.put(f"文字起こし完了: {txt_path}")

    def start_audio_recording(self):
        if self.audio_recorder and self.audio_recorder.is_recording:
            self.log_queue.put("音声録音は既に開始されています。")
            return
        
        idx = self.input_device_combo.currentData()
        if idx is None:
            self.log_queue.put("入力デバイスを選択してください。")
            return

        self.audio_recorder = StereoMixerRecorder(
            device_index=idx,
            log_queue=self.log_queue
        )
        self.audio_recorder.start()

    def stop_audio_recording(self):
        if self.audio_recorder and self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording()
            self.audio_recorder = None
            self.log_queue.put("音声録音を停止しました。")
        else:
            self.log_queue.put("音声録音は開始されていません。")

    def generate_pdf(self):
        pdf_thread = PDFPlatypusGenerationThread(self.save_dir, self.log_queue)
        pdf_thread.start()

    def open_mirroring_window(self):
        """
        ここでは PyAutoGUI でのミラーを例示。
        mss でプレビュー表示したい場合は別途実装が必要。
        """
        if self.mirroring_window and self.mirroring_window.isVisible():
            self.mirroring_window.close()
            self.mirroring_window = None
            return
        
        def get_monitor_index():
            # monitor_combo などでユーザーが選んだ index (mss.monitors) を返す
            return self.monitor_combo.currentData()  # 例: 0,1,2...
        self.mirroring_window = MirroringWindow(get_monitor_index_callback=get_monitor_index, parent=self)
        self.mirroring_window.show()

    # =========================================
    # ログ更新
    # =========================================
    def update_log(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.append(msg)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
