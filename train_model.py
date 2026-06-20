import os
from ultralytics import YOLO

def main():
    # Path to the data yaml configuration file
    data_yaml_path = os.path.abspath("datasets/merged_dataset/data.yaml")
    if not os.path.exists(data_yaml_path):
        print(f"Error: Dataset yaml configuration not found at '{data_yaml_path}'")
        print("Please run 'prepare_merged_dataset.py' first to prepare the dataset.")
        return

    # Path to the pre-trained license plate detection model weights
    pretrained_weights = "license_plate_yolov8.pt"
    if not os.path.exists(pretrained_weights):
        print(f"Warning: Pre-trained weights not found at '{pretrained_weights}'")
        print("Falling back to standard YOLOv8n model...")
        pretrained_weights = "yolov8n.pt"

    print("--------------------------------------------------")
    print(f"Loading weights from: {pretrained_weights}")
    model = YOLO(pretrained_weights)

    print(f"Starting YOLOv8 training on merged custom dataset...")
    print(f"Dataset Config: {data_yaml_path}")
    print("Training settings: 5 epochs (CPU test run)")
    print("--------------------------------------------------\n")

    # Start training
    # For CPU testing, we limit to 5 epochs. You can increase this to 50 or 100 
    # if you have a GPU or want to train longer.
    model.train(
        data=data_yaml_path,
        epochs=5,
        imgsz=640,
        batch=16,
        device="cpu"  # Change to '0' or 'cuda' if GPU is available
    )

    print("\n✅ Training execution complete!")
    print("Trained model weights are saved in: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
