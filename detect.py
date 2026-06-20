import os
import re
import cv2
import csv
import time
import logging
import datetime
import collections
import numpy as np
import easyocr
from ultralytics import YOLO

# ==========================================
# CONSTANTS & CONFIGURATIONS
# ==========================================
MODEL_PATH = "license_plate_yolov8.pt"
LOG_FILE_TXT = "detected_numbers.txt"
LOG_FILE_CSV = "detected_numbers.csv"
SYSTEM_LOG_FILE = "anpr_system.log"
IMAGES_DIR = "images"

# Inference / Processing thresholds
YOLO_CONF_THRESHOLD = 0.50     # Avoid false positive bounding boxes
MIN_OCR_CONF_THRESHOLD = 0.40  # Reject low-confidence characters
OCR_COOLDOWN = 2.0             # Cool-down to wait after vehicle capture (seconds)
STABLE_DETECTION_FRAMES = 5    # Frames needed before triggering stable capture
VOTE_FRAME_COUNT = 5           # Number of frames captured for majority voting
CROP_PADDING = 15              # Bounding box padding in pixels to prevent cutoffs

# Character mappings to correct common OCR classification mistakes
CHAR_TO_NUM = {'O': '0', 'I': '1', 'J': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '1', 'D': '0'}
NUM_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}

# ==========================================
# SYSTEM LOGGER SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(SYSTEM_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ANPR_System")

# Global variables
logged_plates = set()
last_recognized_plate_text = "None"
last_recognized_plate_conf = 0.0

# State machine variables
consecutive_detection_frames = 0
plate_captured_this_session = False
frames_without_plate = 0

# ==========================================
# HELPER FUNCTIONS & LOGIC
# ==========================================
def load_existing_plates():
    """Loads previously logged plate strings from TXT and CSV to prevent duplicates globally."""
    global logged_plates
    logged_plates = set()
    
    # Read from TXT
    if os.path.exists(LOG_FILE_TXT):
        try:
            with open(LOG_FILE_TXT, "r") as f:
                for line in f:
                    match = re.search(r'\]\s*(.*)', line)
                    if match:
                        plate = match.group(1).strip()
                        if plate:
                            logged_plates.add(plate)
        except Exception as e:
            logger.error(f"Failed loading plates from TXT: {e}")

    # Read from CSV
    if os.path.exists(LOG_FILE_CSV):
        try:
            with open(LOG_FILE_CSV, "r") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row and len(row) >= 2:
                        plate = row[1].strip()
                        if plate:
                            logged_plates.add(plate)
        except Exception as e:
            logger.error(f"Failed loading plates from CSV: {e}")
            
    logger.info(f"Duplicate prevention cache initialized with {len(logged_plates)} unique plates.")

def preprocess_plate_image(img):
    """Applies advanced Computer Vision preprocessing to maximize text readability for the CNN OCR."""
    try:
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize by 3x using bicubic interpolation (makes character edges smooth for networks)
        resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        
        # 3. Apply Bilateral Filter (reduces camera sensor noise while preserving text edges)
        denoised = cv2.bilateralFilter(resized, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 4. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Prevents overexposure/underexposure under extreme lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 5. Apply unsharp mask (sharpening kernel) to sharpen character borders
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        return gray

def correct_and_validate_indian_plate(text):
    """Cleans, corrects, and validates Indian license plate formats using regex and position mappings."""
    # Force uppercase and remove all non-alphanumeric noise characters
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Strip common prefix noise (such as 'IND' on High Security Registration Plates)
    if cleaned.startswith("IND") and len(cleaned) >= 7:
        test_strip = cleaned[3:]
        if len(test_strip) >= 5:
            cleaned = test_strip
            
    n = len(cleaned)
    if n < 5 or n > 11:
        return cleaned, False

    corrected = list(cleaned)

    # 1. Check for Bharat (BH) series format: [2 digits] [BH] [4 digits] [2 letters]
    is_bh = False
    if n == 10:
        c2 = NUM_TO_CHAR.get(corrected[2], corrected[2])
        c3 = NUM_TO_CHAR.get(corrected[3], corrected[3])
        if c2 + c3 == "BH" or cleaned[2:4] == "BH":
            corrected[2] = 'B'
            corrected[3] = 'H'
            is_bh = True

    if is_bh:
        # Year digits (0-1) must be digits
        for i in range(2):
            if corrected[i].isalpha():
                corrected[i] = CHAR_TO_NUM.get(corrected[i], corrected[i])
        # Unique registration number (4-7) must be digits
        for i in range(4, 8):
            if corrected[i].isalpha():
                corrected[i] = CHAR_TO_NUM.get(corrected[i], corrected[i])
        # Series letters (8-9) must be letters
        for i in range(8, 10):
            if corrected[i].isdigit():
                corrected[i] = NUM_TO_CHAR.get(corrected[i], corrected[i])
        candidate = "".join(corrected)
        if re.match(r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$', candidate):
            return candidate, True
        return candidate, False

    # 2. Standard Indian License Plate Format:
    # [State Code: 2 letters] [District Code: 1-2 digits] [Series: 0-3 letters] [Number: 1-4 digits]
    # State code (Indices 0 and 1) must be letters
    for i in range(2):
        if corrected[i].isdigit():
            corrected[i] = NUM_TO_CHAR.get(corrected[i], corrected[i])

    # Index 2 must be a digit (District Code)
    if corrected[2].isalpha():
        corrected[2] = CHAR_TO_NUM.get(corrected[2], corrected[2])

    # Calculate expected length of digits at the end
    num_digits = 0
    for i in range(n - 1, 2, -1):
        if corrected[i].isdigit() or corrected[i] in CHAR_TO_NUM:
            num_digits += 1
        else:
            break
    num_digits = min(4, max(1, num_digits))
    start_num_idx = n - num_digits

    # Unique number digits must be digits
    for i in range(start_num_idx, n):
        if corrected[i].isalpha():
            corrected[i] = CHAR_TO_NUM.get(corrected[i], corrected[i])

    # Decide if index 3 is part of District code (digit) or Series (letter)
    if start_num_idx > 4:
        if corrected[3].isalpha():
            corrected[3] = CHAR_TO_NUM.get(corrected[3], corrected[3])
    else:
        if corrected[3].isdigit():
            corrected[3] = NUM_TO_CHAR.get(corrected[3], corrected[3])

    # Middle Series letters (Index 4 up to start_num_idx) must be letters
    for i in range(4, start_num_idx):
        if corrected[i].isdigit():
            corrected[i] = NUM_TO_CHAR.get(corrected[i], corrected[i])

    candidate = "".join(corrected)
    # Match standard Indian registration number regex
    if re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$', candidate):
        return candidate, True
    return candidate, False

def save_plate_record(plate_text, ocr_conf, image_path):
    """Saves the verified ANPR detection to TXT and CSV formats with headers."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Append to TXT
    try:
        with open(LOG_FILE_TXT, "a") as f:
            f.write(f"[{now_str}] {plate_text}\n")
        logger.info(f"Logged to TXT: {plate_text}")
    except Exception as e:
        logger.error(f"Failed writing plate to TXT log file: {e}")

    # 2. Append to CSV
    csv_exists = os.path.exists(LOG_FILE_CSV)
    try:
        with open(LOG_FILE_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(["Timestamp", "Plate", "Confidence", "Image_Path"])
            writer.writerow([now_str, plate_text, f"{ocr_conf:.2%}", image_path])
        logger.info(f"Logged to CSV: {plate_text} ({ocr_conf:.2%})")
    except Exception as e:
        logger.error(f"Failed writing plate to CSV log file: {e}")

def run_temporal_analysis(cap, model, reader):
    """Captures 5 frames sequentially, saves them to disk, processes OCR, and votes on the winner."""
    global last_recognized_plate_text, last_recognized_plate_conf
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    candidates = []
    confidences = {}  # Map plate string -> list of confidences
    saved_paths = []
    
    logger.info("Starting temporal multi-frame collection (5 frames)...")
    
    for i in range(VOTE_FRAME_COUNT):
        time.sleep(0.1)  # Brief delay to capture slight variations in angles/focus
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Frame {i+1} capture failed.")
            continue
            
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(IMAGES_DIR, f"captured_{timestamp_str}_f{i+1}.jpg")
        cv2.imwrite(image_path, frame)
        saved_paths.append(image_path)
        
        # Plate Detection
        results = model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
        best_coords = None
        best_box_conf = 0.0
        
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf > best_box_conf:
                    best_box_conf = conf
                    best_coords = box.xyxy[0].tolist()
                    
        if best_coords is not None:
            h, w, _ = frame.shape
            xmin, ymin, xmax, ymax = map(int, best_coords)
            
            # Apply crop padding to prevent character cutoff
            xmin = max(0, xmin - CROP_PADDING)
            ymin = max(0, ymin - CROP_PADDING)
            xmax = min(w, xmax + CROP_PADDING)
            ymax = min(h, ymax + CROP_PADDING)
            
            cropped = frame[ymin:ymax, xmin:xmax]
            if cropped.size > 0:
                # Preprocess image (Bilateral filter + CLAHE + Sharpening)
                preprocessed = preprocess_plate_image(cropped)
                
                # Perform OCR with strict capital alphanumeric allowlist
                ocr_results = reader.readtext(preprocessed, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                
                if ocr_results:
                    avg_conf = sum([r[2] for r in ocr_results]) / len(ocr_results)
                    raw_text = "".join([r[1] for r in ocr_results]).strip().upper()
                    
                    # Apply correction mapping and validate formatting
                    corrected_text, is_valid = correct_and_validate_indian_plate(raw_text)
                    
                    logger.info(f"Frame {i+1} - Raw: '{raw_text}', Corrected: '{corrected_text}' (Valid: {is_valid}, Conf: {avg_conf:.1%})")
                    
                    # Store if passes confidence threshold
                    if avg_conf >= MIN_OCR_CONF_THRESHOLD:
                        if corrected_text not in confidences:
                            confidences[corrected_text] = []
                        confidences[corrected_text].append(avg_conf)
                        candidates.append(corrected_text)
                    else:
                        logger.warning(f"Frame {i+1} OCR rejected: Confidence {avg_conf:.1%} below threshold.")
                else:
                    logger.warning(f"Frame {i+1}: No characters detected.")
            else:
                logger.warning(f"Frame {i+1}: Cropped plate was empty.")
        else:
            logger.warning(f"Frame {i+1}: YOLO failed to detect number plate.")

    # 4. Perform Majority Voting
    if candidates:
        vote_counter = collections.Counter(candidates)
        # Choose the most frequent candidate
        voted_plate, count = vote_counter.most_common(1)[0]
        avg_voted_conf = sum(confidences[voted_plate]) / len(confidences[voted_plate])
        
        # Check if the voted plate is already logged (Global Duplicate Prevention)
        if voted_plate in logged_plates:
            logger.info(f"Voted Plate: {voted_plate} is already logged. Skipping duplicate.")
            last_recognized_plate_text = voted_plate
            last_recognized_plate_conf = avg_voted_conf
            return
            
        # Determine the primary image path associated with the voted plate
        primary_image = saved_paths[0] if saved_paths else "unknown.jpg"
        
        # Save record to CSV and TXT
        save_plate_record(voted_plate, avg_voted_conf, primary_image)
        
        # Update overlay values
        last_recognized_plate_text = voted_plate
        last_recognized_plate_conf = avg_voted_conf
        
        logger.info(f"🎯 Final Voted Plate: {voted_plate} (Agreed frames: {count}/{len(candidates)}, Avg Conf: {avg_voted_conf:.2%})")
    else:
        logger.error("Temporal Analysis Failed: No valid plates detected across the 5 frames.")

# ==========================================
# MAIN APPLICATION LOOP
# ==========================================
def main():
    global consecutive_detection_frames, plate_captured_this_session, frames_without_plate
    
    # Load previously logged plates into global cache
    load_existing_plates()
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"YOLO Weights file not found at '{MODEL_PATH}'")
        return

    logger.info("Initializing YOLOv8 model...")
    model = YOLO(MODEL_PATH)

    logger.info("Initializing EasyOCR Engine...")
    reader = easyocr.Reader(['en'])

    logger.info("Opening webcam capture...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam could not be opened. Check camera hardware connection.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logger.info("\n-----------------------------------------------------")
    logger.info("ANPR Production System is running successfully!")
    logger.info("Press 'Q' inside the camera feed to quit.")
    logger.info("-----------------------------------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera.")
            break

        # Fast localization inference
        results = model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)

        plate_detected = False
        best_box_coords = None
        best_conf = 0.0

        # Extract plate bounding box with highest confidence
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf > best_conf:
                    best_conf = conf
                    best_box_coords = box.xyxy[0].tolist()
                    plate_detected = True

        # Handle keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        force_ocr = (key == ord(' ') or key == ord('c'))

        # State-Machine Controller
        if plate_detected and best_box_coords is not None:
            frames_without_plate = 0
            xmin, ymin, xmax, ymax = map(int, best_box_coords)
            
            # Draw visual tracking box (Green) on preview feed
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Plate Loc: {best_conf:.1%}", (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)

            if not plate_captured_this_session:
                consecutive_detection_frames += 1
                progress_dots = "." * consecutive_detection_frames
                cv2.putText(frame, f"Stabilizing vehicle{progress_dots}", (xmin, ymax + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)
                
                # Check stable frame threshold (5 consecutive frames) or manual override
                if consecutive_detection_frames >= STABLE_DETECTION_FRAMES or force_ocr:
                    plate_captured_this_session = True
                    run_temporal_analysis(cap, model, reader)
            elif force_ocr:
                logger.info("Manual override triggered during active session.")
                run_temporal_analysis(cap, model, reader)
        else:
            consecutive_detection_frames = 0
            frames_without_plate += 1
            
            # If no plate is seen for 30 consecutive frames (approx 1 sec), reset lock
            if frames_without_plate >= 30:
                if plate_captured_this_session:
                    logger.info("Vehicle left the frame. Resetting state-machine.")
                plate_captured_this_session = False

        # If manually forced with no plate detected in frame
        if force_ocr and not plate_detected:
            logger.info("Manual capture triggered without auto-locked plate.")
            run_temporal_analysis(cap, model, reader)

        # ==========================================
        # VIDEO HUD SCREEN OVERLAYS
# ==========================================
        # 1. Overlay Last Read Plate Number
        cv2.putText(frame, f"Last Read: {last_recognized_plate_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        
        # 2. Overlay Last Read OCR Confidence Score
        conf_color = (0, 255, 0) if last_recognized_plate_conf >= 0.70 else (0, 165, 255)
        cv2.putText(frame, f"OCR Conf: {last_recognized_plate_conf:.2%}", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2, lineType=cv2.LINE_AA)
        
        # 3. Overlay Total Logged Plate Count
        cv2.putText(frame, f"Logged Plates: {len(logged_plates)}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, lineType=cv2.LINE_AA)

        # 4. Overlay Status indicators
        status_text = "Status: Locked & Monitoring" if plate_captured_this_session else "Status: Scanning..."
        status_color = (255, 0, 0) if plate_captured_this_session else (0, 255, 255)
        cv2.putText(frame, status_text, (20, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, lineType=cv2.LINE_AA)

        # Flash visual capture effect (screen border flash)
        if force_ocr:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 20)

        # Show live feed
        cv2.imshow("Production ANPR System", frame)

        # Q to quit safely
        if key == ord('q'):
            logger.info("User requested exit.")
            break

    # Release resources safely
    cap.release()
    cv2.destroyAllWindows()
    logger.info("System clean shutdown completed. Camera released.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled system crash: {e}", exc_info=True)
