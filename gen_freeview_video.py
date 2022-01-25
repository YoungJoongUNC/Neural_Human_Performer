import cv2
import numpy as np
import glob
import os

path = 'data/perform/demo/epoch_-1/debug/0/'
vid_filename = 'subject_0'

files = os.listdir(path)
files.sort()

speed = 30
img_array = []
for idx in range(len(files)):

    img = cv2.imread(path + str(idx) + '.png')
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('videos/' + vid_filename + '.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), speed, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
