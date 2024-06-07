import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
model = tf.keras.models.load_model('/Model_saves/first.keras')

def is_valid(board, row, col, num):
    if num in board[row]:
        return False
    
    for i in range(9):
        if board[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    
    return True

def solve_sudoku(board):
    empty_cell = find_empty_cell(board)
    if not empty_cell:
        return True
    
    row, col = empty_cell
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = -1
    return False

def find_empty_cell(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == -1:
                return (i, j)
    return None

def lf(i, tl, br):
	img = i.copy()
	h, w = img.shape[:2]
	mxa = 0
	cd = (None, None)
	if tl is None:
		tl = [0, 0]
	if br is None:
		br = [w, h]	
	for x in range(tl[0], br[0]):
		for y in range(tl[1], br[1]):
			if img.item(y,x) == 255 and x < w and y < h:
				area = cv2.floodFill(img, None, (x,y), 64)
				if area[0] > mxa:
					mxa = area[0]
					cd = (x, y)
	for x in range(w):
		for y in range(h):
			if img.item(y,x) == 255 and x < w and y < h:
				cv2.floodFill(img, None, (x,y), 64)		
	cvr = np.zeros(((h + 2), (w + 2)), np.uint8)
	if all([p is not None for p in cd]):
		cv2.floodFill(img, cvr, cd, 255)
	tp, btm, lft, rgt = h, 0, w, 0
	
	for x in range(w):
		for y in range(h):
			if img.item(y, x) == 64:
				cv2.floodFill(img, cvr, (x, y), 0)
			
			if img.item(y, x) == 255:
				tp = min(y, tp)
				btm = max(y, btm)
				lft = min(x, lft)
				rgt = max(x, rgt)
			
	box = [[lft, tp], [rgt, btm]]
	return img, np.array(box, dtype='float32'), cd

def alg(img, size, mgn=0, bgd=0):
	h, w = img.shape[:2]
	def ctr(lgth):
		if lgth % 2 == 0:
			s1 = int((size - lgth) / 2)
			s2 = s1
		else:
			s1 = int((size - lgth) / 2)
			s2 = s1 + 1
		return s1, s2
	if h > w:
		t_pad = int(mgn / 2)
		b_pad = t_pad
		ratio = (size - mgn) / h
		w, h = int(ratio *  w), int(ratio * h)
		l_pad, r_pad = ctr(w)
	else:
		l_pad = int(mgn / 2)
		r_pad = l_pad
		ratio = (size - mgn) / w
		w, h = int(ratio * w), int(ratio * h)
		t_pad, b_pad = ctr(h)
	img = cv2.resize(img, (w,h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, bgd)
	return cv2.resize(img, (size, size))

def drecog(image, model):
	h, w = image.shape[:2]
	r = 28/w
	dim = (28, int(r*h))
	image = cv2.resize(image, dim)
	for i in range(28):
		for j in range(28):
			if image[i][j] < 150:	
				image[i][j] = 255
			else:
				image[i][j] = 0
	
	image = image.reshape((1, 784)).astype('float32')
	image = image / 255
	predc = model.predict(image)
	return predc.argmax()

def sudoku_to_image(board):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-0.5, 9, 1))
    ax.set_yticks(np.arange(-0.5, 9, 1))
    ax.grid(which="both", color="black", linestyle="-", linewidth=2)
    ax.imshow(board, cmap="Blues", vmin=0, vmax=1000)
    for i in range(9):
        for j in range(9):
            if board[i][j] != -1:
                ax.text(j, i, str(board[i][j]), ha="center", va="center", fontsize=12)
    plt.savefig("tempsave.jpg",bbox_inches='tight')



Tk().withdraw()
filename = askopenfilename()
print(filename)
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
image = cv.imread(filename)
h, w, d = image.shape
r = 300/w
dim = (300, int(r*h))
image = cv2.resize(image, dim)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(grey,(9,9),0)
thres = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

cnts,hie = cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

bgt = None
mxa = 0
for i in cnts:
    area = cv2.contourArea(i)
    peri = cv2.arcLength(i,True)
    app = cv2.approxPolyDP(i,0.02*peri,True)
    if area>mxa and len(app)==4:
        mxa = area
        bgt = app
		
big = bgt.reshape((4,2))
coord = np.zeros((4,2),np.float32)
sum = np.sum(big,axis=1)
diff = np.diff(big,axis=1)
coord[0] = big[sum.argmin()]
coord[2] = big[sum.argmax()]
coord[1] = big[diff.argmin()]
coord[3] = big[diff.argmax()]

pts = np.float32([[0,0], [450,0], [450,450], [0,450]])
pers = cv2.getPerspectiveTransform(coord, pts)

pers = cv2.warpPerspective(grey, pers, (450,450))

sq = []
sd = 450/9

for i in range(9):
    for j in range(9):
        tl = (i*sd,j*sd)
        br = ((i+1)*sd,(j+1)*sd)
        sq.append((tl,br))

digits = []
ind = 0
pers = cv2.adaptiveThreshold(pers,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

for i in sq:
    cell = pers[int(i[0][0]):int(i[1][0]),int(i[0][1]):int(i[1][1])]
    h, w = cell.shape[:2]
    mgn = int(np.mean([h,w])/2.5)
    _, box, seed = lf(cell, [mgn, mgn], [w - mgn, h - mgn])
    cell = cell[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]
    w = box[1][0] - box[0][0]
    h = box[1][1] - box[0][1]
    if w > 0 and h > 0 and (w*h) > 100 and len(i) > 0:
        digits.append(alg(cell,28,4))
    else:
        digits.append(np.zeros((28,28), np.uint8))
		
preds = []
ind = 0
for i in range(9):
    te = []
    for j in range(9):
        res = cv2.countNonZero(digits[ind])
        if res>=150:
            te.append(drecog(digits[ind],model))
        elif res<150 and res>=100:
            te.append(1)
        else:
            te.append(' ')
        ind = ind+1
    preds.append(te)
	
df = pd.DataFrame(preds)

temp = df.to_numpy()
for i in range(len(temp)):
      for j in range(len(temp[0])):
            if temp[i][j]==' ':
                  temp[i][j] = -1
solve_sudoku(temp)
sudoku_to_image(temp.astype(int))

temp = cv2.imread('tempsave.jpg')

resize = cv2.resize(image, (temp.shape[1], temp.shape[0]))

while True:
    numpy_horizontal = np.hstack((resize, temp))
    numpy_horizontal_concat = np.concatenate((resize, temp), axis=1)

    cv2.imshow("a",numpy_horizontal_concat)
    k = cv2.waitKey(0) & 0xFF
    print(k)
    if k == 255:
        cv2.destroyAllWindows()
        break