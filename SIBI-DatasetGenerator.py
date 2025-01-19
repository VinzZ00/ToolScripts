import cv2
import mediapipe as mp
import os
import time

# Set up Mediapipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create folders to save the dataset
dataset_path = "dataset"
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for letter in letters:
    if not os.path.exists(f"{dataset_path}/{letter}"):
        os.makedirs(f"{dataset_path}/{letter}")

# Function to crop the hand from the frame with padding
def crop_hand_with_padding(frame, landmarks, padding=20):
    # Get the bounding box of the hand
    x_min = min([landmark.x for landmark in landmarks]) * frame.shape[1]
    x_max = max([landmark.x for landmark in landmarks]) * frame.shape[1]
    y_min = min([landmark.y for landmark in landmarks]) * frame.shape[0]
    y_max = max([landmark.y for landmark in landmarks]) * frame.shape[0]

    # Add padding around the hand (optional)
    x_min = max(0, int(x_min - padding))
    x_max = min(frame.shape[1], int(x_max + padding))
    y_min = max(0, int(y_min - padding))
    y_max = min(frame.shape[0], int(y_max + padding))

    # Crop the hand from the frame
    cropped_hand = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
    return cropped_hand, (x_min, y_min, x_max, y_max)  # Return cropped hand and the bounding box

# Start video capture loop
index = 1
letter_index = 0
current_letter = letters[letter_index]
recording = False
cropped_frames = []

start_time = 0
countdown_time = 5  # Set to 5 seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view (selfie view)
    frame = cv2.flip(frame, 1)

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand landmarks if hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks (optional, can remove)
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Crop the hand with padding
            cropped_hand, (x_min, y_min, x_max, y_max) = crop_hand_with_padding(frame, landmarks.landmark)

            # Resize the cropped hand to a fixed size (e.g., 224x224 for input to a model)
            cropped_hand_resized = cv2.resize(cropped_hand, (224, 224))

            # Show cropped hand preview (during both preview and recording)
            cv2.imshow("Cropped Hand", cropped_hand_resized)

            # Draw bounding box with padding on the original frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Store cropped frames for recording
            if recording:
                cropped_frames.append(cropped_hand_resized)

                # Save the landmarks to a text file
                keypoints_file = f"{dataset_path}/{current_letter}/{current_letter}_{index}_keypoints.txt"
                with open(keypoints_file, "a") as file:
                    # Write x, y, z coordinates of each hand landmark
                    for landmark in landmarks.landmark:
                        file.write(f"{landmark.x} {landmark.y} {landmark.z}\n")

    # Countdown logic when recording
    if recording:
        elapsed_time = time.time() - start_time
        remaining_time = max(0, countdown_time - int(elapsed_time))

        # Display the countdown
        cv2.putText(frame, f"Recording {current_letter}... {remaining_time}s left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Stop recording after 5 seconds
        if elapsed_time >= countdown_time:
            recording = False

            # Save video after recording
            video_filename = f"{dataset_path}/{current_letter}/{current_letter}_{index}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(video_filename, fourcc, 30, (224, 224))
            for frame in cropped_frames:
                out.write(frame)
            out.release()

            index += 1

            # Move to the next letter after 5 gestures for the current letter
            if index % 5 == 0:
                letter_index += 1
                if letter_index < len(letters):
                    current_letter = letters[letter_index]
                else:
                    break

    # If not recording, display instructions
    if not recording:
        cv2.putText(frame, f"Press 'C' to start recording for {current_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recorder", frame)

    # Wait for user input to capture the gesture
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and not recording:  # Press 'C' to start recording
        recording = True
        cropped_frames = []  # Reset frames for the current letter
        start_time = time.time()  # Start the timer

    if key == ord('q'):  # Press 'Q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
