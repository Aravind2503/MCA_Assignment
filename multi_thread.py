import cv2
import numpy as np
import math
import convolution
import threading
import time

img = cv2.imread("Resources/mountain_image.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (500, 500))
img = np.pad(img, [(1, 1), (1, 1)], mode='constant', constant_values=0)

# cv2.imshow("img", img)
# cv2.waitKey(0)
final_image = np.zeros(shape=(img.shape[0], img.shape[1]))


num_threads = 20
rows_n = int(img.shape[0]/num_threads)

thread_pool = []

laplace = [[0, -1, 0],
           [-1, 4, -1],
           [0, -1, 0]]


for i in range(num_threads):
    thread_pool.append(threading.Thread(target=convolution.convolve_multi_thread, args=(
        img, i*rows_n, (i+1)*rows_n, 1, img.shape[1]-1, laplace, final_image)))

start = time.time()
for i in thread_pool:
    i.start()

for i in thread_pool:
    i.join()

end = time.time()
print("time taken: ", end-start)

cv2.imshow("img", final_image/255)
cv2.waitKey(0)
