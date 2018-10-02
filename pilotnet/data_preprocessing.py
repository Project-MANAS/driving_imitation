import os
import cv2
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity

filenames = np.array(pd.read_csv("dataset/interpolated.csv", usecols=["filename", "angle"], index_col=None, dtype='O'))

steering_angles = []
steering = []

for i in range(0, filenames.shape[0] - 4):
    if filenames[i][0].find("center") == 0 and os.path.exists("dataset/" + str(filenames[i][0])):
        print("dataset/" + filenames[i][0])
        steering_angles.append("dataset/" + filenames[i][0])
        steering.append(float(filenames[i][1]))


steering_angles = np.array(steering_angles)
np.save("processed/steering_angles", np.array(steering))


mean = np.zeros((192, 256, 4))
std = np.zeros((192, 256, 4))


for i in range(0, steering_angles.size - 4):
    images= []
    for j in range(0, 4):
        image1 = cv2.resize(cv2.imread(steering_angles[i + j], cv2.IMREAD_GRAYSCALE), (256, 192))
        image2 = cv2.resize(cv2.imread(steering_angles[i + j + 1], cv2.IMREAD_GRAYSCALE), (256, 192))
        image_diff = image1 - image2
        image_diff = rescale_intensity(image_diff, in_range=(-255, 255), out_range=(0, 255))
        images.append(image_diff)

    image_combined = np.array(images)
    image_combined = np.transpose(image_combined, [1, 2, 0])

    if i == 0:
        mean = image_combined
    mean_old = mean

    mean = mean + (image_combined - mean)/(i + 1)
    std = std + (image_combined - mean_old) * (image_combined - mean)

    print("Processing image: ", i)
    np.save("processed/images/" + str(i), image_combined)

std = np.sqrt(std/(steering_angles.size - 4 -1))

np.save("processed/mean", mean)
np.save("processed/std", std)
