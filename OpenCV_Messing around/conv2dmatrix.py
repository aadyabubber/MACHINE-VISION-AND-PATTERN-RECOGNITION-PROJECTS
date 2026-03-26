import numpy as np
import time

def conv2dmatrix(image, H):
    image = np.asarray(image)
    H = np.asarray(H)

    img_vec = image.reshape(-1, 1)

    start = time.perf_counter()
    out_vec = H @ img_vec
    end = time.perf_counter()

    latency = end - start

    P = H.shape[0]
    out_size = int(np.sqrt(P))
    out_img = out_vec.reshape(out_size, out_size)

    return out_img, latency
