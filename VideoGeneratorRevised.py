import cv2
import mediapipe as mp
import os
import KeypointExtractor as kpExtract


listOfPrimeVideoPath = []
listOfCroppedVideoPath = []
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

    dataset_path = "dataset-26-May-2025"
    letters = 'ABJZ'
    
    for letter in letters:
        os.makedirs(f"{dataset_path}/{letter}", exist_ok=True)

    index = 0
    letter_index = 0
    current_letter = letters[letter_index]
    recording = False
    cropped_frames = []
    full_frame = []

    frame_count = 0
    TARGET_FRAMES = 150  # 5 seconds at 30 FPS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:

                if recording: full_frame.append(frame.copy())

                cropped_hand = crop_hand_with_padding(frame, landmarks.landmark)
                cropped_hand_resized = cv2.resize(cropped_hand, (224, 224))

                if recording: cropped_frames.append(cropped_hand_resized.copy())

                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow("Cropped Hand", cropped_hand_resized)

        if recording:
            frame_count += 1
            remaining_frames = TARGET_FRAMES - frame_count
            cv2.putText(frame, f"Recording {current_letter}... {remaining_frames} frames left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if frame_count >= TARGET_FRAMES:
                recording = False

                video_filename = f"{dataset_path}/{current_letter}/{current_letter}_{index}"
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out = cv2.VideoWriter(f'{video_filename}.mp4', fourcc, 30, (224, 224))
                primeOut = cv2.VideoWriter(f'{video_filename}-prime.mp4', fourcc, 30, (1280, 720))
                listOfCroppedVideoPath.append(f'{video_filename}.mp4')
                listOfPrimeVideoPath.append(f'{video_filename}-prime.mp4')

                for frame in cropped_frames: out.write(frame)
                for frame in full_frame: primeOut.write(frame)

                primeOut.release()
                out.release()
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

def AugmentFlipVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Could not open video.")
    
    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

    # Generate output path by appending '_flipped' before extension
    base, ext = os.path.splitext(videoPath)
    output_path = f"{base}_flipped{ext}"

    # Create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video...\nInput: {videoPath}\nOutput: {output_path}")

    # Read and flip each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        flipped_frame = cv2.flip(frame, 1)  # Flip horizontally
        out.write(flipped_frame)

    # Release everything
    cap.release()
    out.release()
    
    print("Video processing completed.")

if __name__ == "__main__":
    generateVideo()
    isAugment = True
    for croppedVideo in listOfCroppedVideoPath:
        AugmentFlipVideo(croppedVideo)

    for video in listOfPrimeVideoPath:
        kpExtract.extract_hand_keypoints(video, video.replace("-prime.mp4", "-prime.csv"), isFlipped=isAugment)