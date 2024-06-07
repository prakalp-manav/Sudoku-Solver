from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
filename = askopenfilename()
print(filename)
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
img = cv.imread(filename)

while True:
    numpy_horizontal = np.hstack((img, img))
    numpy_horizontal_concat = np.concatenate((img, img), axis=1)

    cv.imshow("a",numpy_horizontal_concat)
    k = cv.waitKey(0) & 0xFF
    print(k)
    if k == 27:
        cv.destroyAllWindows()
        break