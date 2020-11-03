import cv2
import os

IMG_WIDTH = 30
IMG_HEIGHT = 30

for file_1 in os.listdir('gtsrb'):
    for file_2 in os.listdir(os.path.join('gtsrb', file_1)):
        img = cv2.imread(os.path.join(os.path.join('gtsrb', file_1), file_2))
        img.resize((IMG_WIDTH, IMG_HEIGHT, 3))
        print(img)
        print(img.shape)
        print(int(file_1))
        break
    break