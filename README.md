# Real-Time Automatic Number Plate Recognition (ANPR) & Character Logger

A production-quality ANPR application written in Python. It detects vehicle license plates using YOLOv8, performs character recognition using EasyOCR, and logs unique recognized registration numbers to structured CSV and text databases.

---

## 🚀 Key Features

* **30 FPS Real-Time Webcam Feed**: Localization of plates runs continuously at high frame rates.
* **5-Frame Temporal Majority Voting**: Captures 5 frames sequentially upon vehicle lock, runs OCR on each, and computes a majority vote to pick the most accurate reading, eliminating blur/glare.
* **Advanced Preprocessing Pipeline**: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE), Bilateral Filter denoising, and unsharp masking (sharpening kernel) to characters to improve OCR readability.
* **Indian Alphanumeric Corrector & Regex**: Corrects letter/number confusion based on index positions (e.g. State Code must be letters, unique ID must be digits). Rejects any plate that doesn't match standard Indian registration formats.
* **Global Duplicate Prevention**: Syncs logs on launch and ensures each plate is recorded exactly once.
* **Dual Output Database**: Saves timestamps, plate text, confidence, and snapshot paths to both `detected_numbers.txt` and `detected_numbers.csv`.
* **Complete Training Pipeline**: Includes scripts to merge multiple Kaggle datasets and train/fine-tune custom YOLOv8 models.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Dattu47/number_plate_detection_app.git
   cd number_plate_detection_app
   ```

2. **Activate the Virtual Environment**:
   * On Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

3. **Install Requirements**:
   ```bash
   pip install ultralytics easyocr opencv-python numpy kagglehub
   ```

4. **Prepare Model Weights**:
   Ensure you have `license_plate_yolov8.pt` in the project root folder. (If you don't have custom weights, the script automatically falls back to standard `yolov8n.pt`).

---

## 💻 How to Use

### 1. Run Real-Time Camera Detection & Logging
Run the detector application:
```bash
python detect.py
```
* Present a license plate in front of the camera. The status will transition from `Scanning...` to `Stabilizing vehicle` (tracks for 5 stable frames).
* It will automatically capture 5 snapshots in `images/`, run OCR on each, vote on the final winner, log it to `detected_numbers.txt` and `detected_numbers.csv`, and lock the detection state until the vehicle leaves.
* Press **`SPACE`** or **`C`** to manually force a capture.
* Press **`Q`** to exit the application window.

### 2. Multi-Dataset Training Pipeline
If you want to train your own YOLO detection model using custom Kaggle datasets:
```bash
# 1. Download, convert XMLs, and merge 2 datasets into YOLO format
python prepare_merged_dataset.py

# 2. Train/Fine-tune YOLOv8 on the merged dataset
python train_model.py
```
The trained model weights will be saved under `runs/detect/train/weights/best.pt`.
