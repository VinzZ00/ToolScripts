import cv2
import mediapipe as mp
import os

def extract_hand_keypoints(video_path: str, output_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(video_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_count = 0

    with open(output_path, "w") as file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    frame_data = []
                    for landmark in landmarks.landmark:
                        frame_data.extend([landmark.x, landmark.y, landmark.z])

                    # Ensure exactly 63 values for one frame (21 keypoints * 3)
                    if len(frame_data) == 63:
                        file.write(" ".join(map(str, frame_data)) + "\n")
                    else: 
                        print(f"Invalid frame data length: {len(frame_data)}")
                        break
            frame_count += 1

    cap.release()
    print(f"Keypoints saved to: {output_path}")
