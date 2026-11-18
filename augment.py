import cv2
import os
import random

input_folder = "original_images"
output_folder = "augmented_images"

os.makedirs(output_folder, exist_ok=True)
count = 0

for img_name in os.listdir(input_folder):

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w = img.shape[:2]

    for i in range(10):

        angle = random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)

        rotated = cv2.warpAffine(img, M, (w, h))

        filename = f"aug_{count}.jpg"
        save_path = os.path.join(output_folder, filename)

        cv2.imwrite(save_path, rotated)

        count += 1

print("Total images created:", count)