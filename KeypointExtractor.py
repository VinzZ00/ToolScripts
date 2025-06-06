import cv2
import mediapipe as mp
import os

def extract_hand_keypoints(video_path: str, output_path, isFlipped: bool = False):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize MediaPipe Hands **once**
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    flipped_csv_path = output_path.removesuffix(".csv") + "_flipped.csv" if isFlipped else None

    with open(output_path, "w") as file:
        flipped_file = open(flipped_csv_path, "w") if isFlipped else None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if isFlipped and flipped_file:
                flipped_frame = cv2.flip(frame, 1)
                flipped_rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
                processFrame(flipped_rgb, flipped_file, hands)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processFrame(frame_rgb, file, hands)

        if flipped_file:
            flipped_file.close()

    cap.release()
    hands.close()

    print(f"Keypoints saved to: {output_path}")
    if isFlipped:
        print(f"Flipped keypoints saved to: {flipped_csv_path}")


def processFrame(frame_rgb, file, hands):
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            frame_data = []
            for landmark in landmarks.landmark:
                frame_data.append([round(landmark.x, 4), round(landmark.y, 4)])
            if len(frame_data) == 21:
                file.write(",".join(map(str, frame_data)) + "\n")
            else:
                print(f"Invalid frame data length: {len(frame_data)}")
