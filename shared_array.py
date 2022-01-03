import os 
import numpy as np
import multiprocessing as mp
import cv2
from contextlib import closing

def _init(shared_arr_):
    # The shared array pointer is a global variable so that it can be accessed by the
    # child processes. It is a tuple (pointer, dtype, shape).
    global shared_arr
    shared_arr = shared_arr_


def shared_to_numpy(shared_arr, dtype, shape):
    """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
    No copy is involved, the array reflects the underlying shared buffer."""
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)


def create_shared_array(dtype, shape):
    """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
    Note that the buffer values are not initialized.
    """
    dtype = np.dtype(dtype)
    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    # Create the RawArray instance.
    shared_arr = mp.RawArray(cdtype, sum(shape))
    # Get a NumPy array view.
    arr = shared_to_numpy(shared_arr, dtype, shape)
    return shared_arr, arr


def parallel_function(image,row_start,row_end,kernel):
    """This function is called in parallel in the different processes. It takes an index range in
    the shared array, and fills in some values there."""
    
    arr = shared_to_numpy(*shared_arr)
    # WARNING: you need to make sure that two processes do not write to the same part of the array.
    # Here, this is the case because the ranges are not overlapping, and all processes receive
    # a different range.
    # arr[i0:i1] = np.arange(i0, i1)
    cv2.filter2D(image[row_start:row_end],arr[row_start:row_end],-1,kernel)


def main():

    laplace = [[0, -1, 0],
               [-1, 4, -1],
               [0, -1, 0]]
    
    image = cv2.imread("Resources/mountain_image.jpg",cv2.IMREAD_GRAYSCALE)
    # For simplicity, make sure the total size is a multiple of the number of processes.
    n_processes = os.cpu_count()
    N = 100
    N = N - (N % n_processes)
    assert N % n_processes == 0

    # Initialize a shared array.
    dtype = np.float32
    shape = (N,)
    shared_arr, arr = create_shared_array(dtype, shape)
    # arr.flat[:] = np.zeros(N)
    # Show [0, 0, 0, ...].
    # print(arr)

    # Create a Pool of processes and expose the shared array to the processes, in a global variable
    # (_init() function).
    with closing(mp.Pool(
            n_processes, initializer=_init, initargs=((shared_arr, dtype, shape),))) as p:
        n = N // n_processes
        # Call parallel_function in parallel.
        p.map(parallel_function(), [(k * n, (k + 1) * n) for k in range(n_processes)])
    # Close the processes.
    p.join()
    # Show [0, 1, 2, 3...]
    cv2.imshow("final_image",arr)
    cv2.waitkey(0)

if __name__ == '__main__':
    mp.freeze_support()
    main()