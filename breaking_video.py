import cv2
from pathlib import Path

frame_dir = Path('Frame')
frame_dir.mkdir(exist_ok=True)

capture = cv2.VideoCapture('hola.mp4')  # Input video path here
frameNr = 0

while True:
    success, frame = capture.read()
    if success:
        cv2.imwrite(f'Frame\\frame_{frameNr}.jpg', frame)
    else:
        break
    frameNr += 1

capture.release()
