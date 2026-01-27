import cv2
import mediapipe as mp
import os
import time

# Function to crop the hand from the frame with padding
def crop_hand_with_padding(frame, landmarks, padding=20):
    x_min = min([landmark.x for landmark in landmarks]) * frame.shape[1]
    x_max = max([landmark.x for landmark in landmarks]) * frame.shape[1]
    y_min = min([landmark.y for landmark in landmarks]) * frame.shape[0]
    y_max = max([landmark.y for landmark in landmarks]) * frame.shape[0]

    x_min = max(0, int(x_min - padding))
    x_max = min(frame.shape[1], int(x_max + padding))
    y_min = max(0, int(y_min - padding))
    y_max = min(frame.shape[0], int(y_max + padding))

    cropped_hand = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
    return cropped_hand

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

dataset_path = "dataset"
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for letter in letters:
    os.makedirs(f"{dataset_path}/{letter}", exist_ok=True)

index = 1
letter_index = 0
current_letter = letters[letter_index]
recording = False
cropped_frames = []
fps = 30
frame_time = 1.0 / fps

start_time = 0
countdown_time = 5

while cap.isOpened():
    start_frame_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            cropped_hand = crop_hand_with_padding(frame, landmarks.landmark)
            cropped_hand_resized = cv2.resize(cropped_hand, (224, 224))

            if recording:
                cropped_frames.append(cropped_hand_resized)

            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Cropped Hand", cropped_hand_resized)

    if recording:
        elapsed_time = time.time() - start_time
        remaining_time = max(0, countdown_time - int(elapsed_time))
        cv2.putText(frame, f"Recording {current_letter}... {remaining_time}s left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if elapsed_time >= countdown_time:
            recording = False
            video_filename = f"{dataset_path}/{current_letter}/{current_letter}_{index}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(video_filename, fourcc, 30, (224, 224))
            for frame in cropped_frames:
                out.write(frame)
            out.release()
            index += 1

            if index % 5 == 0:
                letter_index += 1
                if letter_index < len(letters):
                    current_letter = letters[letter_index]
                else:
                    break

    if not recording:
        cv2.putText(frame, f"Press 'C' to start recording for {current_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recorder", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and not recording:
        recording = True
        cropped_frames = []
        start_time = time.time()

    if key == ord('q'):
        break
    
    elapsed_time = time.time() - start_frame_time
    if elapsed_time < frame_time:
        time.sleep(frame_time - elapsed_time)

cap.release()
cv2.destroyAllWindows()
