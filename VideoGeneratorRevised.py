import cv2
import mediapipe as mp
import os
import KeypointExtractor as kpExtract


listOfPrimeVideoPath = []

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

def generateVideo():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    dataset_path = "dataset-Elvin"
    letters = 'A'
    
    for letter in letters:
        os.makedirs(f"{dataset_path}/{letter}", exist_ok=True)

    index = 1
    letter_index = 0
    current_letter = letters[letter_index]
    recording = False
    cropped_frames = []
    full_frame = []

    cropped_frames_orig = []
    full_frame_orig = []
    
    frame_count = 0
    TARGET_FRAMES = 150  # 5 seconds at 30 FPS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        frame_orig = frame.copy()
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        rgb_frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        results_orig = hands.process(rgb_frame_orig)

        

        if recording:
            if results.multi_hand_landmarks:
                for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
                    # --- Flipped frame ---
                    full_frame.append(frame.copy())
                    cropped_hand = crop_hand_with_padding(frame, landmarks.landmark)
                    cropped_hand_resized = cv2.resize(cropped_hand, (224, 224))
                    cropped_frames.append(cropped_hand_resized.copy())
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                    # --- Original (non-flipped) frame ---
                    if results_orig.multi_hand_landmarks and hand_idx < len(results_orig.multi_hand_landmarks):
                        landmarks_orig = results_orig.multi_hand_landmarks[hand_idx]
                        full_frame_orig.append(frame_orig.copy())
                        cropped_hand_orig = crop_hand_with_padding(frame_orig, landmarks_orig.landmark)
                        cropped_hand_orig_resized = cv2.resize(cropped_hand_orig, (224, 224))
                        cropped_frames_orig.append(cropped_hand_orig_resized.copy())

                cv2.imshow("Cropped Hand (Flipped)", cropped_frames[-1] if cropped_frames else frame)
                
                if cropped_frames_orig:
                    cv2.imshow("Cropped Hand (Original)", cropped_frames_orig[-1])
                    
                frame_count += 1
                remaining_frames = TARGET_FRAMES - frame_count
            else:         
                cropped_frames, full_frame = [], []
                cropped_frames_orig, full_frame_orig = [], []
                frame_count = 0
                remaining_frames = TARGET_FRAMES - frame_count

            cv2.putText(frame, f"Recording {current_letter}... {remaining_frames} frames left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if frame_count >= TARGET_FRAMES:
                recording = False
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                video_filename = f"{dataset_path}/{current_letter}/{current_letter}_{index}"

                # --- Flipped versions ---
                out = cv2.VideoWriter(f'{video_filename}-flipped.mp4', fourcc, 30, (224, 224))
                prime_out = cv2.VideoWriter(f'{video_filename}-flipped-prime.mp4', fourcc, 30, (1280, 720))
                for f in cropped_frames: out.write(f)
                for f in full_frame: prime_out.write(f)
                out.release()
                prime_out.release()
                listOfPrimeVideoPath.append(f'{video_filename}-flipped-prime.mp4')

                # --- Original (non-flipped) versions ---
                out_orig = cv2.VideoWriter(f'{video_filename}.mp4', fourcc, 30, (224, 224))
                prime_out_orig = cv2.VideoWriter(f'{video_filename}-prime.mp4', fourcc, 30, (1280, 720))
                for f in cropped_frames_orig: out_orig.write(f)
                for f in full_frame_orig: prime_out_orig.write(f)
                out_orig.release()
                prime_out_orig.release()
                listOfPrimeVideoPath.append(f'{video_filename}-prime.mp4')

                # Reset all buffers
                cropped_frames, full_frame = [], []
                cropped_frames_orig, full_frame_orig = [], []

                index += 1
                if index % 3 == 0:
                    letter_index += 1
                    if letter_index < len(letters):
                        current_letter = letters[letter_index]
                    else:
                        break

        if not recording:
            cv2.putText(frame, f"will take photo-{index} Press 'C' to start recording for {current_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and not recording:
            recording = True
            cropped_frames = []
            full_frame = []
            frame_count = 0

        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generateVideo()

    print('list of primeVideoPath available:')
    for video in listOfPrimeVideoPath:
        print(video)
        kpExtract.extract_hand_keypoints(video, video.replace("-prime.mp4", "-prime.csv"))