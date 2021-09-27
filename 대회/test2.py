import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

paths = glob.glob('사람/김인환/*.jpg')
paths = np.array([plt.imread(paths[i]) for i in range(len(paths))])
size = (640, 480)
frame_resized = []
for i in range(len(paths)):
    paths[i] = paths[i].astype(np.int16)

    frame_resized += cv2.resize(paths[i], size, interpolation=cv2.INTER_AREA)
    print(paths[i])

for i in range(len(paths)):
        cv2.imshow('rgb_image', frame_resized[i])
        cv2.imwrite("김인환.jpg", frame_resized[i])


