import cv2
import numpy as np
import math
import convolution
# import future
import time


img = cv2.imread("Resources/salt_pepper.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (1000, 1000))
img = np.pad(img, [(1, 1), (1, 1)], mode='constant', constant_values=0)

cv2.imshow("img", img)
cv2.waitKey(0)
imgy = np.zeros(shape=(img.shape[0], img.shape[1]))


# cv2.imshow("img", imgy)
# cv2.waitKey(0)

imgx = np.zeros(shape=(img.shape[0], img.shape[1]))


laplace = [[0, -1, 0],
           [-1, 4, -1],
           [0, -1, 0]]
# final_image = np.zeros(shape=(img.shape[0], img.shape[1]))


# for i in range(1, img.shape[0]-1):
#     for j in range(1, img.shape[1]-1):
#         imgy[i][j] = 2*img[i-1][j] + 1*img[i-1][j-1] + 1*img[i-1][j+1] - \
#             (2*img[i+1][j] + 1*img[i+1][j-1] + 1*img[i+1][j+1])

#         imgx[i][j] = 1*img[i-1][j-1] + 2*img[i][j-1] + 1*img[i+1][j-1] - \
#             (2*img[i+1][j+1] + 1*img[i][j+1] + 1*img[i+1][j+1])

#         final_image[i][j] = math.sqrt(imgy[i][j]**2+imgx[i][j]**2)/1.414129/255

#         # imgx[i][j] = (4/16*img[i][j]+2/16*img[i-1][j]+2/16*img[i][j+1]+2/16*img[i][j-1] +
#         #               2/16*img[i+1][j]+img[i-1][j-1]/16+img[i-1][j+1]/16 +
#         #   img[i+1][j-1]/16+img[i+1][j+1]/16)/150
# # cv2.normalize(imgx, img)
# cv2.imshow("imgx", imgx)
# cv2.waitKey(0)

# cv2.imshow("imgy", imgy)
# cv2.waitKey(0)

start = time.time()
final = convolution.median_s(
    img, 1, img.shape[0]-1, 1, img.shape[1]-1)

end = time.time()
print("time taken: ", end-start)
cv2.imshow("img", final/255)
cv2.waitKey(0)
