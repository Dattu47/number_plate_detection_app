import os
import shutil
import random
import xml.etree.ElementTree as ET
import kagglehub

# Set random seed for reproducibility
random.seed(42)

# Configurations
OUTPUT_DIR = os.path.abspath("datasets/merged_dataset")
TRAIN_RATIO = 0.80

def convert_coordinates(size, box):
    """Converts PASCAL VOC bounding box coordinates to normal YOLO coordinates."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

def parse_xml_to_yolo(xml_path):
    """Parses Pascal VOC XML labels and formats them under class '0' (license_plate)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size_elem = root.find("size")
        if size_elem is None:
            return []
        width = float(size_elem.find("width").text)
        height = float(size_elem.find("height").text)
        
        if width <= 0 or height <= 0:
            return []
            
        yolo_boxes = []
        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            # Accept any plate-related classifications (like number_plate, license, licence, etc.)
            if "plate" in name or "licence" in name or "license" in name or name == "number_plate":
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                # Clamp coordinates to image boundaries
                xmin = max(0.0, xmin)
                ymin = max(0.0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                    
                yolo_box = convert_coordinates((width, height), (xmin, ymin, xmax, ymax))
                yolo_boxes.append(f"0 {' '.join([f'{coord:.6f}' for coord in yolo_box])}")
        return yolo_boxes
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return []

def scan_dataset(root_dir):
    """Scans dataset directories recursively to find and match images and XML labels."""
    samples = []
    xmls = {}
    images = {}
    
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            base = os.path.splitext(f)[0]
            full_path = os.path.join(dirpath, f)
            if ext == ".xml":
                xmls[base] = full_path
            elif ext in [".jpg", ".jpeg", ".png"]:
                images[base] = full_path
                
    # Match them by base filename
    for base, xml_path in xmls.items():
        if base in images:
            samples.append((images[base], xml_path))
            
    return samples

def main():
    print("--------------------------------------------------")
    print("Downloading Dataset 1: dataclusterlabs/indian-number-plates-dataset...")
    d1_path = kagglehub.dataset_download("dataclusterlabs/indian-number-plates-dataset")
    print(f"Dataset 1 downloaded to: {d1_path}")
    
    print("\nDownloading Dataset 2: andrewmvd/car-plate-detection...")
    d2_path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
    print(f"Dataset 2 downloaded to: {d2_path}")
    print("--------------------------------------------------\n")
    
    # Scan both datasets
    print("Scanning datasets and matching images with labels...")
    d1_samples = scan_dataset(d1_path)
    d2_samples = scan_dataset(d2_path)
    
    print(f"Dataset 1: Found {len(d1_samples)} valid image-label pairs.")
    print(f"Dataset 2: Found {len(d2_samples)} valid image-label pairs.")
    
    # Merge samples with prefix identifiers to prevent filename collisions
    all_samples = []
    for img, xml in d1_samples:
        all_samples.append((img, xml, "d1_"))
    for img, xml in d2_samples:
        all_samples.append((img, xml, "d2_"))
        
    total_samples = len(all_samples)
    print(f"\nTotal merged dataset size: {total_samples} samples.")
    if total_samples == 0:
        print("Error: No samples found. Cannot prepare dataset.")
        return

    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(total_samples * TRAIN_RATIO)
    train_samples = all_samples[:split_idx]
    valid_samples = all_samples[split_idx:]
    
    print(f"Split: {len(train_samples)} training samples, {len(valid_samples)} validation samples.")

    # Create folder structure
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    def process_split(samples, split_name):
        print(f"Processing and converting {split_name} split...")
        count = 0
        for img_path, xml_path, prefix in samples:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            new_base = prefix + base_name
            
            # Destination paths
            dest_img = os.path.join(OUTPUT_DIR, split_name, "images", new_base + ".jpg")
            dest_lbl = os.path.join(OUTPUT_DIR, split_name, "labels", new_base + ".txt")
            
            # Parse XML and convert
            yolo_lines = parse_xml_to_yolo(xml_path)
            if not yolo_lines:
                # Skip samples with empty/invalid bounds
                continue
                
            # Copy image
            try:
                shutil.copy(img_path, dest_img)
                # Write YOLO label txt file
                with open(dest_lbl, "w") as f:
                    f.write("\n".join(yolo_lines))
                count += 1
            except Exception as e:
                print(f"Error copying {img_path}: {e}")
        print(f"Successfully processed {count} samples in {split_name} split.")

    process_split(train_samples, "train")
    process_split(valid_samples, "valid")

    # Generate data.yaml file
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: valid/images

nc: 1
names: ['license_plate']
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"\n[SUCCESS] Merged dataset preparation complete!")
    print(f"Dataset configurations saved to: {yaml_path}")

if __name__ == "__main__":
    main()
