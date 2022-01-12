import os
import multiprocessing
from typing import final
import cv2
import numpy as np
import math
import convolution
import threading
import time
from multiprocessing import Process, shared_memory




if __name__ == "__main__":
    img = cv2.imread("Resources/dark1.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (500, 500))
    img = np.pad(img, [(1, 1), (1, 1)], mode='constant', constant_values=0)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    print(os.getpid())

    final_image = np.zeros(shape=(img.shape[0], img.shape[1]))

    num_threads = 8
    rows_n = int(img.shape[0]/num_threads)

    process_pool = []

    
    shm = shared_memory.SharedMemory(
        create=True, size=final_image.nbytes, name="shr_mem")

    final_image = np.ndarray(
        img.shape, dtype=np.float32, buffer=shm.buf)
    # b[:] = final_image[:]

    for i in range(num_threads):
        process_pool.append(Process(target=convolution.log_transform_p, args=(
            img, i*rows_n, (i+1)*rows_n, 1, img.shape[1]-1)))

    start = time.time()
    for i in process_pool:
        i.start()

    for i in process_pool:
        i.join()

    end = time.time()
    print("time taken: ", end-start)

    cv2.imshow("img", final_image/255)
    cv2.waitKey(0)

    cv2.imwrite("Resources/log_transform.png",final_image)
