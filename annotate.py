
import cv2
import os
import shutil

# Input image
image_path = "original_images/pancard1.jpg"

# Output folders
image_output = "dataset/train/images"
label_output = "dataset/train/labels"

os.makedirs(image_output, exist_ok=True)
os.makedirs(label_output, exist_ok=True)

# Read image
img = cv2.imread(image_path)

if img is None:
    print("Image not found! Check image path.")
    exit()

print("Drag your mouse around the complete PAN card.")
print("Press ENTER or SPACE to confirm.")
print("Press ESC to cancel.")

# Draw bounding box using mouse
box = cv2.selectROI(
    "PAN Card Annotation",
    img,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

x, y, w, h = box

if w == 0 or h == 0:
    print("No rectangle selected.")
    exit()

# Convert to YOLO format
height, width = img.shape[:2]

x_center = (x + w / 2) / width
y_center = (y + h / 2) / height
box_width = w / width
box_height = h / height

# Save image
filename = os.path.basename(image_path)
shutil.copy2(
    image_path,
    os.path.join(image_output, filename)
)

# Save YOLO label
label_name = os.path.splitext(filename)[0] + ".txt"
label_path = os.path.join(label_output, label_name)

with open(label_path, "w") as f:
    f.write(
        f"0 {x_center:.6f} {y_center:.6f} "
        f"{box_width:.6f} {box_height:.6f}\n"
    )

print("PAN card annotation completed!")
print("Image saved:", filename)
print("Label saved:", label_path)
print("YOLO coordinates saved successfully!")
