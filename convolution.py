import cv2
import numpy as np
import math
import multi_process
from multiprocessing import shared_memory
from numpy.fft import fft2,ifft2


def convolve(image=[], row_start=0, row_end=0, col_start=0, col_end=0, kernel=[]):
    final_image = np.zeros(shape=(image.shape[0], image.shape[1]))
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            final_image[i][j] = image[i-1][j-1]*kernel[0][0]+image[i-1][j]*kernel[0][1]+image[i-1][j+1]*kernel[0][2]+image[i][j-1]*kernel[1][0] + \
                image[i][j]*kernel[1][1]+image[i][j+1]*kernel[1][2]+image[i+1][j-1] * \
                kernel[2][0]+image[i+1][j]*kernel[2][1] + \
                image[i+1][j+1]*kernel[2][2]
    return final_image


def convolve_multi_thread(image=[], row_start=0, row_end=0, col_start=0, col_end=0, kernel=[], final_image=[]):

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            final_image[i][j] = image[i-1][j-1]*kernel[0][0]+image[i-1][j]*kernel[0][1]+image[i-1][j+1]*kernel[0][2]+image[i][j-1]*kernel[1][0] + \
                image[i][j]*kernel[1][1]+image[i][j+1]*kernel[1][2]+image[i+1][j-1] * \
                kernel[2][0]+image[i+1][j]*kernel[2][1] + \
                image[i+1][j+1]*kernel[2][2]


def convolve_multi_process(image=[], row_start=0, row_end=0, col_start=0, col_end=0, kernel=[]):

    existing_shm = shared_memory.SharedMemory(name='shr_mem')

    final_image = np.ndarray(
        (image.shape[0], image.shape[1]), dtype=np.float32, buffer=existing_shm.buf)

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            final_image[i][j] = image[i-1][j-1]*kernel[0][0]+image[i-1][j]*kernel[0][1]+image[i-1][j+1]*kernel[0][2]+image[i][j-1]*kernel[1][0] + \
                image[i][j]*kernel[1][1]+image[i][j+1]*kernel[1][2]+image[i+1][j-1] * \
                kernel[2][0]+image[i+1][j]*kernel[2][1] + \
                image[i+1][j+1]*kernel[2][2]

def convolve_multi_process1(image=[], row_start=0, row_end=0, col_start=0, col_end=0, kernel=[]):

    existing_shm = shared_memory.SharedMemory(name='shr_mem')

    final_image = np.ndarray(
        (image.shape[0], image.shape[1]), dtype=np.float32, buffer=existing_shm.buf)

    cv2.filter2D(image[row_start:row_end],final_image[row_start:row_end],-1,kernel)
    # cv2.filter2D(image,final_image,-1,kernel)


def log_transform_p(image=[], row_start=0, row_end=0, col_start=0, col_end=0):

    existing_shm = shared_memory.SharedMemory(name='shr_mem')

    final_image = np.ndarray(
        (image.shape[0], image.shape[1]), dtype=np.float32, buffer=existing_shm.buf)

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            final_image[i][j] = 255/(1+math.log(1+400))*(math.log(1+image[i][j]))


def log_transform_s(image=[], row_start=0, row_end=0, col_start=0, col_end=0):

    final_image = np.zeros(shape=(image.shape[0], image.shape[1]))

    c = 255 / np.log(1 + np.max(image))

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            final_image[i][j] = c*(np.log(1+image[i][j]))
    return final_image


def median_p(image=[], row_start=0, row_end=0, col_start=0, col_end=0, kernel=[]):

    existing_shm = shared_memory.SharedMemory(name='shr_mem')

    final_image = np.ndarray(
        (image.shape[0], image.shape[1]), dtype=np.float32, buffer=existing_shm.buf)

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            final_image[i][j] = np.median([image[i-1][j-1],image[i-1][j],image[i-1][j+1],image[i][j-1],
                image[i][j],image[i][j+1],image[i+1][j-1] ,
                image[i+1][j],
                image[i+1][j+1]])

def median_s(image=[], row_start=0, row_end=0, col_start=0, col_end=0, kernel=[]):

    final_image = np.zeros(shape=(image.shape[0], image.shape[1]))

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            final_image[i][j] = np.median([image[i-1][j-1],image[i-1][j],image[i-1][j+1],image[i][j-1],
                image[i][j],image[i][j+1],image[i+1][j-1] ,
                image[i+1][j],
                image[i+1][j+1]])
    
    return final_image