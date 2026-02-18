import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python import vision
import pandas as pd
import os

model_path = "/Users/adamafaal-refaee/Desktop/Desktop-Win/codeing/projects/machine_learning/handmarks/models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Global variable to store latest results
latest_result = None

def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result
)

cap = cv2.VideoCapture(0)

filename = "/Users/adamafaal-refaee/Desktop/Desktop-Win/codeing/projects/machine_learning/handmarks/data2.csv"

if not os.path.exists(filename):
    df = pd.DataFrame(columns=["i", "j", "x", "y"])
    df.to_csv(filename, index=False)

time.sleep(3)

with HandLandmarker.create_from_options(options) as landmarker:

    start_time = time.time()
    
    j = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.time() - start_time) * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)
        
        # === DRAW LANDMARKS ===
        if latest_result and latest_result.hand_landmarks:
            height, width, _ = frame.shape
            i = 0
            for hand_landmarks in latest_result.hand_landmarks:
                for landmark in hand_landmarks:
                    i+=1
                    if i == 1:
                        j += 1
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
                    new_row = pd.DataFrame([[j, i, x, y]], columns=["Iteration #", "Landmark #", "x", "y"])
                    new_row.to_csv(filename, mode="a", header=False, index=False)
                    print(f"Iteration {j}: Landmark {i}: ({x}, {y})")

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

print()
