# 1. Import
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 2.EDA and Cleaning
train_images = "C:/Users/suhaimi/Desktop/PPE_PROJECT/dataset/train/images"
train_labels = "C:/Users/suhaimi/Desktop/PPE_PROJECT/dataset/train/labels"
# Get filenames without extensions
image_files = {os.path.splitext(f)[0] for f in os.listdir(train_images) if f.endswith('.jpg')}
label_files = {os.path.splitext(f)[0] for f in os.listdir(train_labels) if f.endswith('.txt')}
# Find missing labels
missing_labels = image_files - label_files

if missing_labels:
    print(f"Missing label files for {len(missing_labels)} images!")
    for img in missing_labels:
        print(f"- {img}.jpg")
else:
    print("All images have corresponding label files.")

label_path = "C:/Users/suhaimi/Desktop/PPE_PROJECT/dataset/train/labels"

empty_files = [
    "construction-1-_mp4-115_jpg.rf.7afe87b728c11bfa2d6616ef75081e4c.txt",
    "construction-4-_mp4-19_jpg.rf.b39073990b499b99dfe415a49fdb90bb.txt",
    "construction-5-_mp4-30_jpg.rf.f671ac8915234f2ca8b9246d095055e9.txt",
    "construction-821-_jpg.rf.098b834846b3be5f1b24ef6dbc3a30a4.txt",
    "youtube-196_jpg.rf.d9d5913fb05f3dfcc17d19e54e9ecbf6.txt"
]

for file in empty_files:
    file_path = os.path.join(label_path, file)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted {file}")

label_path_empty = "C:/Users/suhaimi/Desktop/PPE_PROJECT/dataset/train/labels/youtube-197_jpg.rf.84e48afd1eb2f504aecfb6456a196950.txt"
# Check if the file exists before deleting
if os.path.exists(label_path_empty):
    os.remove(label_path_empty)
    print(f"🗑 Deleted {label_path_empty}")
else:
    print("No empty label files found.")

# Check each label file
empty_files = []
incorrect_format = []
for file in os.listdir(label_path):
    if file.endswith(".txt"):
        file_path = os.path.join(label_path, file)
        with open(file_path, "r") as f:
            lines = f.readlines()
            if not lines:
                empty_files.append(file)
            else:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:  # Each line should have 5 values
                        incorrect_format.append(file)
                        break

# Print results
if empty_files:
    print(f"{len(empty_files)} label files are empty:")
    for f in empty_files[:5]:  # Show only first 5
        print(f"- {f}")
else:
    print("✅ No empty label files found.")

if incorrect_format:
    print(f"{len(incorrect_format)} label files have incorrect format")
    for f in incorrect_format[:5]:  # Show only first 5
        print(f"- {f}")
else:
    print("✅ All label files have the correct format.")

# Check for empty label files
empty_files = [f for f in os.listdir(label_path) if os.path.getsize(os.path.join(label_path, f)) == 0]

if empty_files:
    print(f"⚠️ {len(empty_files)} label files are STILL empty!")
    for f in empty_files:
        print(f"- {f}")
else:
    print("✅ All labels contain annotations. Good to go!")

# 3. Load Model and Training
# Load the model
model = YOLO("yolov8n.pt")
# Print dataset configuration
print(model.yaml)
# Train on PPE dataset
model.train(data="C:/Users/suhaimi/Desktop/FINALCAPSTONE/dataset/yamldata.yaml", epochs=50, imgsz=640)
plt.switch_backend("TkAgg")
# Load the trained model
model = YOLO("C:/Users/suhaimi/Desktop/FINALCAPSTONE/runs/detect/train4/weights/best.pt")
# Define the path to your sample image
image_path = "C:/Users/suhaimi/Desktop/FINALCAPSTONE/female.jpg"  # Change to your actual image filename
# Perform inference (prediction)
results = model(image_path)
# Plot the results
img = results[0].plot()  # Draw results on the image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()