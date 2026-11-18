
import cv2
from ultralytics import YOLO

# Configuration
MODEL_PATH = "runs/detect/train9/weights/best.pt"
CONFIDENCE = 0.60
WINDOW_NAME = "PAN Card Detection System"

# Card shape configuration
MIN_AREA = 10000
MAX_AREA = 300000
MIN_ASPECT_RATIO = 1.35
MAX_ASPECT_RATIO = 2.10

# Load trained PAN card model
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to access webcam")


def detect_card_shape(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    card_detected = False

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # Check rectangular shape
        if len(approx) == 4:

            x, y, w, h = cv2.boundingRect(approx)

            if h == 0:
                continue

            ratio = max(w, h) / min(w, h)

            if MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:

                card_detected = True

                cv2.drawContours(
                    frame,
                    [approx],
                    -1,
                    (0, 165, 255),
                    2
                )

                break

    return card_detected


print("PAN Card Detection System Started")
print("Press Q or ESC to exit")

try:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Unable to read webcam frame")
            break

        # YOLO PAN card detection
        results = model.predict(
            frame,
            conf=CONFIDENCE,
            verbose=False
        )

        pan_detected = False

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            if class_name.lower() == "pancard":

                pan_detected = True

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 200, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"PAN CARD ({confidence:.2f})",
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 0),
                    2
                )

        # Detect generic rectangular card shapes
        card_detected = detect_card_shape(frame)

        # Status logic
        if pan_detected:

            status = "PAN CARD DETECTED"
            color = (0, 200, 0)

        elif card_detected:

            status = "PAN CARD NOT DETECTED"
            color = (0, 0, 255)

        else:

            status = None

        # Display status only when required
        if status:

            cv2.rectangle(
                frame,
                (10, 10),
                (550, 65),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                frame,
                status,
                (25, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )

        # Display webcam
        cv2.imshow(WINDOW_NAME, frame)

        # Exit using Q or ESC
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        # Exit when window is closed
        if cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("PAN Card Detection System Stopped")


