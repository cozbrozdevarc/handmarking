import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === Model Path ===
model_path = "/Users/adamafaal-refaee/Desktop/Desktop-Win/codeing/models/hand_landmarker.task"

# === Callback Function ===
def print_result(result, output_image, timestamp_ms):
    print("Timestamp:", timestamp_ms)
    print("Detected hands:", len(result))

# === Create Options ===
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# === Open Camera ===
cap = cv2.VideoCapture(0)

# === Create Landmarker ONCE (not inside loop) ===
with HandLandmarker.create_from_options(options) as landmarker:

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height,width,channels = frame.shape
        cv2.imshow("wideo", frame)

        # Convert BGR (OpenCV) to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Generate proper increasing timestamp
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Send frame asynchronously
        detection_result = landmarker.detect_async(mp_image, timestamp_ms)

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) == 27:
            break
        
cap.release()
cv2.destroyAllWindows()
