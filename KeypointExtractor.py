import cv2
import mediapipe as mp
import os

def extract_hand_keypoints(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.3)

    with open(output_path, "w") as file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processFrame(frame_rgb, file, hands)

    cap.release()
    hands.close()

    print(f"Keypoints saved to: {output_path}")


def processFrame(frame_rgb, file, hands):
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            frame_data = []
            for landmark in landmarks.landmark:
                frame_data.append([round(landmark.x, 4), round(landmark.y, 4)])
            if len(frame_data) == 21:
                file.write(",".join([f'"{point}"' for point in frame_data]) + "\n")
            else:
                print(f"Invalid frame data length: {len(frame_data)}")