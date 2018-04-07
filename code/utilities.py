import cv2
import numpy as np
from matplotlib import pyplot as plt
#from skimage.draw import line_aa


class ImageUtilities:
    @staticmethod
    def pad_image(img, padding_size, zeros=True):
        """
        Creates new image with specified padding
        :param img: Image to pad
        :param padding_size: Size of pad
        :param zeros: If true, padded with zeros, else with ones
        :return: New, padded image
        """
        fun = np.zeros if zeros else np.ones
        padded = fun((img.shape[0] + 2 * padding_size, img.shape[1] + 2 * padding_size))
        padded[padding_size:-padding_size, padding_size:-padding_size] = img
        return padded

    @staticmethod
    def show_contours_in_image(input_path, contours, lines):
        img = cv2.imread(input_path, 0)
        for contour in contours:
            ImageUtilities.draw_contour(img, contour)

        for line in lines:
            l, t = line
            x1, y1, x2, y2 = l
            xx1, yy1, xx2, yy2 = t

            #rr2, cc2, val2 = line_aa(y1, x1, y2, x2)
            #img[rr2, cc2] = val2 * 125

            #rr, cc, val = line_aa(yy1, xx1, yy2, xx2)
            #img[rr, cc] = val * 125

        plt.title("Particle Contours")
        plt.imshow(img, cmap='gray')
        plt.show()
        cv2.waitKey()

    @staticmethod
    def draw_contour(img, contour):
        for x, y in contour:
            img[y, x] = 255

    @staticmethod
    def normal_hist(image):
        # calculates normalized histogram of an image
        h, w = image.shape
        hist = [0.0] * 256
        for y in range(h):
            for x in range(w):
                hist[image[y, x]] += 1
        return np.array(hist) / (h * w)

    @staticmethod
    def prefixsum(hist):
        # prefix sum of a numpy array, list
        return [sum(hist[:i + 1]) for i in range(len(hist))]

    @staticmethod
    def histeq(image):
        # calculate Histogram
        hist = ImageUtilities.normal_hist(image)
        cdf = np.array(ImageUtilities.prefixsum(hist))  # cumulative distribution function
        sk = np.uint8(255 * cdf)  # finding transfer function values
        s1, s2 = image.shape
        new_file = np.zeros_like(image)
        # applying transfered values for each pixels
        for i in range(0, s1):
            for j in range(0, s2):
                new_file[i, j] = sk[image[i, j]]
        H = ImageUtilities.normal_hist(new_file)
        # return transformed image
        return new_file