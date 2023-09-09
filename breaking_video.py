import cv2 

capture = cv2.VideoCapture('video_preview_h264.mp4') ## input video path here 
frameNr = 0
# os.system("rm -rf Frame")
while (True):
    success, frame = capture.read()
    if success:
        cv2.imwrite(f'Frame/frame_{frameNr}.jpg', frame)
    else:
        break
    frameNr = frameNr+1
capture.release()