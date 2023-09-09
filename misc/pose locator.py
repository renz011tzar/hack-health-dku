import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Constants
SECONDS = 10
FRAME_RATE = 30
IMAGE_SIZE = 256
KEYPOINT_CONNECTIONS = [
    [16, 14], [14, 12], [6, 8], [8, 10], [15, 13], [13, 11], 
    [5, 7], [7, 9], [11, 12], [5, 6], [2, 2], [1, 1], 
    [0, 0], [11, 5], [12, 6]
]
COLOR = (0, 255, 0)
THICKNESS = 7

# Load the pre-trained model
model = hub.load('./movenet_singlepose_thunder_4')  # Replace with the actual path to your model
movenet = model.signatures['serving_default']

def bigplotter(frame, i, h, w):
    """
    This function takes a frame, processes it, performs pose detection, and visualizes the keypoints.
    Args:
    frame: np.array, the input frame.
    i: int, the frame index.
    h: int, the height of the frame.
    w: int, the width of the frame.
    """
    buffer = abs(w-h)
    if h > w:
        img = frame[int(buffer/2):h-int(buffer/2), 0:w]
        h = w
    else:
        img = frame[0:h, int(buffer/2):w-int(buffer/2)]
        w = h

    # Preprocess the image for model inference
    image = tf.convert_to_tensor([img])
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    # Perform model inference
    outputs = movenet(image)
    keypoints = outputs['output_0']

    # Visualize keypoints
    for start_idx, end_idx in KEYPOINT_CONNECTIONS:
        if keypoints[0][0][start_idx][2] < 0.25 or keypoints[0][0][end_idx][2] < 0.25:
            continue

        start = (int(w*keypoints[0][0][start_idx][1]), int(h*keypoints[0][0][start_idx][0]))
        end = (int(w*keypoints[0][0][end_idx][1]), int(h*keypoints[0][0][end_idx][0]))
        cv2.line(img, start, end, COLOR, THICKNESS)

    cv2.imwrite(f'docs/extra/frame_{i}.jpeg', img)

# Main script
if __name__ == "__main__":
    video = cv2.VideoCapture('/path/to/your/video.mp4')  # Replace with the actual path to your video

    for i in range(SECONDS * FRAME_RATE):
        ret, frame = video.read()
        if not ret or frame is None:
            print(f"Could not read frame at index {i}. Exiting...")
            break

        h, w = frame.shape[:2]
        bigplotter(frame, i, h, w)
