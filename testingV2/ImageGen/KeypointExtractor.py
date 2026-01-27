import cv2
import string
import time

def capture_videos():
    # Set up video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Ensure the camera is open
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define video properties
    frame_width = 1280
    frame_height = 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print("Press 'R' to start recording and 'Q' to quit.")

    recording = False

    for letter in string.ascii_uppercase:  # A to Z
        for copy in range(1, 5):  # 1 to 4
            filename = f"{letter}-{copy}.mp4"

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to capture frame.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                # Resize frame to 520x520
                frame = cv2.resize(frame, (520, 520))

                cv2.imshow('Recording', frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('r') and not recording:
                    print(f"Recording: {filename}")
                    out = cv2.VideoWriter(filename, fourcc, 30, (frame_width, frame_height))
                    start_time = time.time()
                    recording = True

                if key == ord('q'):
                    print("Recording stopped by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if recording:
                    out.write(frame)

                    if time.time() - start_time >= 4:  # Record for 4 seconds
                        out.release()
                        recording = False
                        break

    cap.release()
    cv2.destroyAllWindows()
    print("All videos captured successfully.")

if __name__ == "__main__":
    capture_videos()
