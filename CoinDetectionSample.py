import cv2
import numpy as np
import matplotlib.pyplot as plt

barya = cv2.imread("BARYA_9.jpg")
cv2.namedWindow('COINS', cv2.WINDOW_NORMAL)
cv2.imshow('COINS', barya)
#Equalize:
gray_barya = cv2.cvtColor(barya,cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
eqlzd_barya = clahe.apply(gray_barya)

#Thresholding:
v = 20  #   any pixelvalue in the gray image less than v are filtered out.
ret1, th1 = cv2.threshold(gray_barya, v, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations = 9)
final_thresh = opening
#   Contouring: before watershed
# https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
_, contours, _ = cv2.findContours(final_thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(barya, contours, -1, (0, 0, 255), 3)
plt.imshow(final_thresh,cmap='gray'),plt.axis('off'),plt.title('Global thresholding v='+str(v))

# cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
# cv2.imshow('Threshold', th1)

# selecting contour to be used
canvas = np.zeros(final_thresh.shape, np.uint8)
dst25=0
maxr_25 = 0
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    maxr_old = maxr_25
    maxr_25 = cv2.contourArea(cnt)
    if radius > 160:
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(canvas, center, radius, (255, 255, 255), -1)
        # cv2.drawContours(barya, contours, -1, (0,0,255), 3)
        if maxr_25<maxr_old:
            maxr_25 = cv2.contourArea(cnt)
print(maxr_25)
_, contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea)
#Algo for 25 cents:
dst25=0
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    if radius > 160 and radius < 200:
        mask = np.zeros(barya.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        dst25_old = dst25
        dst25 = cv2.bitwise_and(barya, barya, mask=mask)
        dst25_new = cv2.bitwise_or(dst25,dst25_old)
        dst25=dst25_new

cv2.namedWindow('25C COINS', cv2.WINDOW_NORMAL)
cv2.imshow('25C COINS', dst25_new)

plt.show()
cv2.waitKey()

# _, contours, _ = cv2.findContours(dst25_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea)
# dst25=0
# for contour in contours:
#
# cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres Act5\\result\\25C COINS.jpg",dst25_new)
#
# #Algo for One peso:
# dst01=0
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     if w < 450 and h < 450 and w > 400 and h > 400:
#         mask = np.zeros(barya.shape[:2], np.uint8)
#         cv2.drawContours(mask, [contour], -1, 255, -1)
#         dst01_old = dst01
#         dst01 = cv2.bitwise_and(barya, barya, mask=mask)
#         dst01_new = cv2.bitwise_or(dst01,dst01_old)
#         dst01=dst01_new
# cv2.namedWindow('1Peso COINS', cv2.WINDOW_NORMAL)
# cv2.imshow('1Peso COINS', dst01_new)
# # cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres Act5\\result\\1Peso COINS.jpg",dst01_new)
#
# dst05=0
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     if w < 550 and h < 550 and w > 450 and h > 450:
#         mask = np.zeros(barya.shape[:2], np.uint8)
#         cv2.drawContours(mask, [contour], -1, 255, -1)
#         dst05_old = dst05
#         dst05 = cv2.bitwise_and(barya, barya, mask=mask)
#         dst05_new = cv2.bitwise_or(dst05,dst05_old)
#         dst05=dst05_new
# cv2.namedWindow('5Peso COINS', cv2.WINDOW_NORMAL)
# cv2.imshow('5Peso COINS', dst05_new)
# cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres Act5\\result\\5Peso COINS.jpg",dst05_new)
#
# dst10=0
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     if w < 460 and h < 460 and w > 435 and h > 430:
#         mask = np.zeros(barya.shape[:2], np.uint8)
#         cv2.drawContours(mask, [contour], -1, 255, -1)
#         dst10_old = dst10
#         dst10 = cv2.bitwise_and(barya, barya, mask=mask)
#         dst10_new = cv2.bitwise_or(dst10,dst10_old)
#         dst10=dst10_new
# cv2.namedWindow('10Peso COINS', cv2.WINDOW_NORMAL)
# cv2.imshow('10Peso COINS', dst10_new)
# # cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres Act5\\result\\10Peso COINS.jpg",dst10_new)




