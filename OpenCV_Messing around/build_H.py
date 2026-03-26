import numpy as np
from conv2dmatrix import conv2dmatrix

def build_H(image_shape, kernel):
    """ Builds the convolution (correlation) matrix H for full zero-padding. """

    M, N = image_shape
    k = kernel.shape[0]

    out_h = M + k - 1   
    out_w = N + k - 1

    H = np.zeros((out_h * out_w, M * N))
    pad = k - 1

    for i in range(out_h):
        for j in range(out_w):
            row = i * out_w + j

            for u in range(k):
                for v in range(k):
                    x = i - pad + u
                    y = j - pad + v

                    if 0 <= x < M and 0 <= y < N:
                        col = x * N + y
                        H[row, col] += kernel[u, v]

    return H
