import numpy as np
import cv2

from utilities import ImageUtilities
from config import Config


class Filters:
    @staticmethod
    def convolve2d(img, kernel):
        """
        Performs convolution in 2D array.
        :param img: numpy 2D array: input array
        :param kernel: array 3x3: convolution kernel
        :return: numpy 2D array: result of convolution, image has same shape as input
        """

        kernel = np.flip(kernel, 0)
        kernel = np.flip(kernel, 1)

        output = np.zeros(img.shape)

        window_size = 3

        padded_input = ImageUtilities.pad_image(img, 1)

        h, w = img.shape
        for x in range(0, w):
            for y in range(0, h):
                window = padded_input[y:y+window_size, x:x+window_size]
                window = np.multiply(window, kernel)
                output[y, x] = window.sum()

        return output


    @staticmethod
    def median_filter(img, window_size):
        """
        Filters image using median filter.
        :param img: numpy 2D array: array to filter
        :param window_size: int: size of filter
        :return: numpy 2D array: filtered image, same shape as input
        """

        if Config.USE_CV2_FILTERS:
            return cv2.medianBlur(img, 3)

        output = np.zeros(img.shape)
        padded_input = ImageUtilities.pad_image(img, 1)

        half_win_size = window_size // 2

        h, w = img.shape

        mid = (window_size**2) // 2

        for x in range(0, w):
            for y in range(0, h):
                window = padded_input[y:y+window_size, x:x+window_size]
                window = window.flatten()
                window.sort()
                val = window[mid]
                output[y][x] = val

        return output

    @staticmethod
    def scharr(img):
        if Config.USE_CV2_FILTERS:
            grad_x = cv2.Scharr(img, cv2.CV_64F, 1, 0) / 16
            grad_y = cv2.Scharr(img, cv2.CV_64F, 0, 1) / 16
            return np.sqrt(grad_x ** 2 + grad_y ** 2)

        horizontal_k = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        vertical_k = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

        grad_x = Filters.convolve2d(img, horizontal_k) / 16
        grad_y = Filters.convolve2d(img, vertical_k) / 16
        return np.sqrt(grad_x ** 2 + grad_y ** 2)

    @staticmethod
    def threshold(img, value, invert=False):
        thresh_max = 255
        if Config.USE_CV2_FILTERS:
            ret, thresh = cv2.threshold(img, value, thresh_max,
                                        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)
            return thresh

        if invert:
            thresh = np.full(img.shape, thresh_max)
            thresh[img > value] = 0
            return thresh
        else:
            thresh = np.zeros(img.shape)
            thresh[img > value] = thresh_max
            return thresh

# usage:
#output = median_filter(input, 3)
#print(output)
