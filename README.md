# スライドキャプチャツール

**AI（Whisper）を活用して、スライド画面の自動キャプチャ＆文字起こしができるツールです。**

自動、手動でのディスプレイキャプチャ、音声録音と文字起こし、PDF出力など、プレゼンや講義記録に便利な機能をオールインワンで提供します。

（このプロジェクトは、ChatGPT O1を多大に用いて作成されたものです。）

## 特徴

- **mss** を用いたマルチディスプレイ対応キャプチャ
    - **自動キャプチャ（画面変化検知）** と **手動キャプチャ**に対応
- **AI (Whisper)** を活用した文字起こし
    - 自動 or 手動キャプチャごとに区切られた音声を**分割＆文字起こし**
    - 文字起こし結果を `.txt` として保存
- **PDF生成** (ReportLab / Platypus)
    - 画像と文字起こし結果を丁寧に折り返してレイアウト
    - 日本語フォントを埋め込むことで文字化け回避
- **GUI**: PyQt5
    - 簡単操作で任意のフォルダにスクリーンショットを保存

---

## インストール

1. Python 3.8+ 環境を用意してください
2. リポジトリをクローンまたはダウンロードし、ルートディレクトリへ移動
3. 依存ライブラリをインストールしてください:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
    - あるいは、主要なライブラリを個別に:
        
        ```bash
        pip install pyqt5 pyaudio mss opencv-python pillow reportlab openai-whisper
        
        ```
        
    - **ffmpeg** がインストール＆Path設定されている必要があります (Whisperが内部で使用)

---

## 使い方

1. **起動**
    
    ```bash
    python main.py
    
    ```
    
2. **保存先フォルダ選択**
    - GUI 上部の「保存先を選択」ボタンで、スクリーンショットや音声ファイルを保存するフォルダを指定
3. **モニタ選択**
    - プルダウンでキャプチャ対象モニタを選択 (mss.monitors[0] = 全画面, 1.. = 個別モニタ)
4. **音声デバイス選択**
    - 音声デバイスをプルダウンで選択
5. **録音開始**
    - 「録音開始」ボタンで音声録音を開始 (必要に応じて)
6. **キャプチャ**
    - **手動キャプチャ**: 「手動キャプチャ」ボタンを押すと即時にスクリーンショットを撮り、音声の区間を分割 → Whisper文字起こし
    - **自動キャプチャ**: 「自動キャプチャ開始」ボタンを押すと、画面変化検知により自動でキャプチャ → 音声区間分割＆文字起こし
    - しきい値(変化の閾値)をスピンボックスで調整
7. **PDF生成**
    - キャプチャされた画像と文字起こし結果をレイアウトした PDF を作成します。
    - 「PDF生成」ボタンを押すと、`result.pdf`が保存先フォルダに作られます
8. **終了**
    - ウィンドウを閉じると、設定（フォルダやしきい値、デバイス選択など）が `settings.json` に保存され、次回起動時に復元されます。

---


## 依存ライブラリ / バージョン情報

- Python 3.8+
- [PyQt5](https://pypi.org/project/PyQt5/)
- [pyaudio](https://pypi.org/project/PyAudio/)
- [mss](https://pypi.org/project/mss/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [Pillow](https://pypi.org/project/Pillow/)
- [reportlab](https://pypi.org/project/reportlab/)
- [openai-whisper](https://github.com/openai/whisper)
- ffmpeg (外部ツール)





---

## 開発者向け (任意)

- **AI (Whisper) の精度向上**: モデルサイズを大きいものに変えたい場合、`whisper.load_model("large")` などに書き換えてください(動作が重くなる可能性あり)。
- **画面変化検知アルゴリズム**: 現在は `absdiff` → `sum()` の単純比較です。領域指定やヒストグラム比較などに変更すれば精度を上げられます。


---
https://opensource.org/license/mit