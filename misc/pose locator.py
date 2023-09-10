import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob
import os
from PIL import Image


seconds = 3

model = hub.load('./movenet_singlepose_thunder_4')
movenet = model.signatures['serving_default']

h = 1
w = 1

color = (0, 255, 0)

thick = 7

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

arr = [[16, 14], [14, 12], [6, 8], [8, 10], [15, 13], [13, 11], [5, 7], [7, 9], [11, 12], [5, 6], [2, 2], [1, 1], [0, 0], [11, 5], [12, 6]]

def bigplotter(frame, i, opt):

    image_path = './Frame/frame_' + str(i) + '.jpeg'
    
    img1 = cv2.imread(image_path)

    dimensions = img1.shape

    h = img1.shape[0]
    w = img1.shape[1]
    
    buffer = abs(w-h)

    if h > w :
        img = img1[int(buffer/2):h-int(buffer/2), 0:w]
        h = w
    else :
        img = img1[0:h, int(buffer/2):w-int(buffer/2)]
        w = h
 
    cv2.imwrite(image_path, img)

    # cv2.imwrite('docs/' + name, img)

    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)

    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

    outputs = movenet(image)
    keypoints = outputs['output_0']

    #return vars
    return_list = []
    for j in range(15):
        starter = arr[j][0];
        ender = arr[j][1]

        if (keypoints[0][0][starter][2] < 0.25 or keypoints[0][0][ender][2] < 0.25) : continue
        
        start = (int(w*keypoints[0][0][starter][1]), int(h*keypoints[0][0][starter][0]))
        end = (int(w*keypoints[0][0][ender][1]), int(h*keypoints[0][0][ender][0]))
        cv2.line(img, start, end, color, thick)
        cv2.imwrite('./Output/frame_' + str(i) + '.jpg', img)
        if opt == 1:    
            #get vercor data here:
            if keypoints[0][0][starter][2] >= 0.25 and keypoints[0][0][ender][2] >= 0.25 and j == 4:
                return_list.append((-int(end[0] - start[0]), -int(end[1] - start[1]), 0))
            if keypoints[0][0][starter][2] >= 0.25 and keypoints[0][0][ender][2] >= 0.25 and j == 5:
                return_list.append((int(end[0] - start[0]), int(end[1] - start[1]), 1)) 
                #reverse the vectors
    return return_list
    



def angleCalc(vect1, vect2):
    x = np.array(vect1)
    y = np.array(vect2)

    len_x = np.sqrt(x.dot(x))
    len_y = np.sqrt(y.dot(y))

    xy = x.dot(y)
    cos_ = xy/(len_x * len_y)
    return cos_

def exampleSquat():
    flag_list = []
    video = cv2.VideoCapture('hola.mp4')
    for i in range(seconds*30):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        cv2.imwrite('./Frame/frame_' + str(i) + '.jpeg', frame)
        #Collect the cordinates for the two variables
        return_list = bigplotter(frame, i, 1)
        #Both variables exist
        if len(return_list) == 2:
            angle_cos = angleCalc(return_list[0], return_list[1])
            if angle_cos < 0.05:
                flag_list.append("OK!")
            else:
                flag_list.append("Uh-oh! Knees behind feet!")
        #whether the two cordinates appeared on the same frame?
    video.release()

    #Video output
    file_dir = "./Output/frame_"
    path_list = []
    video = cv2.VideoWriter('vid.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(720,720))
    for i in range(seconds*30):
        path = str(i) + '.jpg'
        img = cv2.imread(file_dir + path)
        img = cv2.resize(img, (720,720))
        try:
            cv2.putText(img, str(flag_list[i]),(100,200),cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 5)
            video.write(img)
        except:
            break
    
    video.release() 

#video = cv2.VideoCapture('hola.mp4')

"""
for i in range(seconds*30):
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = video.read()
    cv2.imwrite('./Frame/frame_' + str(i) + '.jpg', frame)
    bigplotter(frame, i)
"""
exampleSquat()