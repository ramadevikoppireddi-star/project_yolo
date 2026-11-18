# PAN Card Detection Using YOLO and OpenCV

## Project Overview

This project uses YOLO object detection and OpenCV to detect PAN cards in real time through a webcam.

The model is trained using only one class: `pancard`. It identifies PAN cards and displays the detection status based on the webcam input.

## Technologies Used

- Python
- YOLO (Ultralytics)
- OpenCV
- Computer Vision
- Deep Learning

## Features

- Real-time PAN card detection using a webcam.
- Trained using only PAN card images.
- Detects PAN cards using a custom-trained YOLO model.
- Displays bounding boxes around detected PAN cards.
- Displays **PAN CARD DETECTED** when a PAN card is identified.
- Displays **PAN CARD NOT DETECTED** when another card, such as an Aadhaar card, is detected but no PAN card is identified.
- Displays no status message when no card is present.
- Press Q or ESC to close the webcam.
- Automatically releases the webcam when the application exits.

## Model Training

The YOLO model is trained using a single detection class:

- Class 0: `pancard`

No separate Aadhaar detection class is used.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python detect.py
```

## Expected Output

| Webcam Input | Expected Output |
|---|---|
| PAN Card | PAN