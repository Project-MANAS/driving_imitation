from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
from skimage.exposure import rescale_intensity
import cv2

def make_grayscale_diff_data(path, num_channels=2):
    df = pd.read_csv(path)
    num_rows = df.shape[0]
    data_path = path[:-4] + '/'
    row, col = 192, 256

    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print "Processed " + str(i) + " images out of " + str(num_rows) + " images"
        for j in range(num_channels):
            path0 = df['filename'].iloc[i - j - 1]
            path1 = df['filename'].iloc[i - j]
            img0 = load_img(data_path + path0, grayscale=True, target_size=(row, col))
            img1 = load_img(data_path + path1, grayscale=True, target_size=(row, col))
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img = img1 - img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[i - num_channels, :, :, j] = img[:, :, 0]
    return X, np.array(df["angle"].iloc[num_channels:])


if __name__ == '__main__':
    print "Pre-processing udacity data..."
    # for i in [1,2,4,5,6]:
    i = 3
    X_train, y_train = make_grayscale_diff_data('/home/naivehobo/Desktop/udacity/'+str(i)+'.csv', 4)
    np.save("/home/naivehobo/Desktop/Rambo/data/X_train_part{}".format(i), X_train)
    np.save("/home/naivehobo/Desktop/Rambo/data/y_train_part{}".format(i), y_train)
