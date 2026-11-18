from ultralytics import YOLO

model = YOLO("runs/detect/train9/weights/best.pt")

results = model.predict(
    source="original_images/pancard1.jpg",
    conf=0.50,
    show=True
)