import cv2
import numpy as np
from matplotlib import pyplot as plt
import collections
from filters import Filters

from utilities import ImageUtilities

BG_COLOR = 0
UNKNOWN_COLOR = 1
FG_COLOR = 2
EDGE_COLOR = 3
TRACK_START_COLOR = 4
TRACK_END_COLOR = 10


def flood_fill(seeds, img, background_val, fill_val):
    """
    Fills a continuous area of image with flooding algorithm
    from starting position (sx,sy). Flood only fills pixels
    with value equal to starting position pixel. If it finds
    pixel with `fg_val` value, it changes it to `edge_val`.
    :param seeds: Starting positions as tuples (x,y)
    :param img: Image to fill (will be modified in place)
    :param background_val: Color that should be flooded
    :param fill_val: Value with which to fill
    """
    q = collections.deque()
    for s in seeds:
        q.append(s)

    min_x = 0
    min_y = 0
    max_x = img.shape[1] - 1
    max_y = img.shape[0] - 1

    while len(q) > 0:
        x, y = q.pop()
        val = img[y, x]

        if val != background_val:
            continue

        img[y, x] = fill_val
        if x < max_x:
            q.append((x+1, y))
        if x > min_x:
            q.append((x-1, y))
        if y < max_y:
            q.append((x, y+1))
        if y > min_y:
            q.append((x, y-1))


def pad_image(img, padding_size, zeros=True):
    """
    Creates new image with specified padding
    :param img: Image to pad
    :param padding_size: Size of pad
    :param zeros: If true, padded with zeros, else with ones
    :return: New, padded image
    """
    fun = np.zeros if zeros else np.ones
    padded = fun((img.shape[0] + 2*padding_size, img.shape[1] + 2*padding_size))
    padded[padding_size:-padding_size, padding_size:-padding_size] = img
    return padded


def track_edge(sx, sy, img, fill_val, edge_val, k_size, k_edge_max=8):
    """
    Tracks the edge by using flooding algorithm, starting at
    position (sx, sy) and stopping at `edge_val` pixels.
    :param sx: Starting position x-coordinate (should be inside particle)
    :param sy: Starting position y-coordinate (should be inside particle)
    :param img: Image to fill (will be modified in place)
    :param fill_val: Value with which to fill
    :param edge_val: Value which is considered as edge
    :param k_size: Size of kernel, must be odd number
    :param k_edge_max: Maximum number of edge pixels that may be present
                       in kernel so that pixel is still considered as particle
    :return: Tuple of edge pixels as list [(x1,y1),(x2,y2),..] and boolean if the
             particle is touching edge.
    """
    output = []
    q = collections.deque()
    q.append((sx, sy))
    half_k_size = k_size // 2
    inv_edge_condition = k_size**2 - k_edge_max

    # Pad the image
    padded = pad_image(img, half_k_size, False)

    min_x = 0
    min_y = 0
    max_x = img.shape[1] - 1
    max_y = img.shape[0] - 1
    touching_limits = False

    while len(q) > 0:
        x, y = q.pop()
        val = img[y, x]

        if val == edge_val:
            output.append((x, y))
            continue

        non_zero = np.count_nonzero(padded[y:y+k_size, x:x+k_size])
        if non_zero <= inv_edge_condition:
            output.append((x,y))
            continue

        if val == fill_val:
            continue

        img[y, x] = fill_val
        if x < max_x:
            q.append((x + 1, y))
        else:
            touching_limits = True
        if x > min_x:
            q.append((x - 1, y))
        else:
            touching_limits = True
        if y < max_y:
            q.append((x, y + 1))
        else:
            touching_limits = True
        if y > min_y:
            q.append((x, y - 1))
        else:
            touching_limits = True

    return output, touching_limits


def track_all_particles(sure_fg, bg_flood):
    """
    Finds contours for all particles in image.
    :param sure_fg: Image with areas where we are sure foreground is
    :param bg_flood: Image with background flooded with BG_COLOR
    :return: List of contours, where each particle has one. Each
             contour is list of tuples (x,y)
    """
    contours = []
    bg_flood[sure_fg > BG_COLOR] = TRACK_START_COLOR
    for (y, x), value in np.ndenumerate(bg_flood):
        if value == TRACK_START_COLOR:
            contour, ignore = track_edge(x, y, bg_flood, TRACK_END_COLOR, BG_COLOR, 5)
            if ignore or len(contour) == 0:
                continue
            contours.append(contour)
    return contours


def find_seeds(background_before_fill):
    """
    Find seeds for background filling.
    :param background_before_fill: Background to use.
    :return: List of seeds array (x,y) tuples.
    """
    seeds = []
    width = background_before_fill.shape[1]
    height = background_before_fill.shape[0]
    max_x = width - 1
    max_y = height - 1

    for x in range(width):
        if background_before_fill[0, x] == UNKNOWN_COLOR:
            seeds.append((x, 0))

    for x in range(width):
        if background_before_fill[max_y, x] == UNKNOWN_COLOR:
            seeds.append((x, max_y))

    for y in range(height):
        if background_before_fill[y, 0] == UNKNOWN_COLOR:
            seeds.append((0, y))

    for y in range(height):
        if background_before_fill[y, max_x] == UNKNOWN_COLOR:
            seeds.append((max_x, y))

    return seeds


def flood_image(original_image_path, flooded_output_path):
    """
    Floods background of the image, starting at seeds that
    are obtained at the edges in unknown areas. After
    filling, unknown areas should only be inside particles.
    :param original_image_path: Path to input image
    :param flooded_output_path: Path to flooded image
    :return: Image where background is BG_COLOR and rest is
             either UNKNOWN_COLOR or FG_COLOR.
    """
    img = cv2.imread(original_image_path, 0)
    sure_fg = Filters.threshold(img, 130, False)

    #equalize histogram
    # equalized_img = ImageUtilities.histeq(img)
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('img'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(equalized_img, cmap='gray')
    # plt.title('equalized_img'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # Scharr filter
    grad = Filters.scharr(img)
    grad_8 = np.uint8(grad)

    grad_bg = Filters.threshold(grad_8, 4, True)
    grad_bg[np.all([grad_bg == 0, img < 55], axis=0)] = 255
    sure_bg = grad_bg - sure_fg

    sure_bg_filtered = Filters.median_filter(sure_bg, 3)

    cv2.imwrite('Gradient.tif', grad_bg)
    cv2.imwrite('Gradient_Background.tif', grad_bg)
    cv2.imwrite('Sure_Background_Filtered.tif', sure_bg_filtered)

    sure_bg_inv = Filters.threshold(sure_bg_filtered, 2, True)
    sure_bg_inv = (sure_bg_inv // 255) + 1
    sure_bg_flood = np.copy(sure_bg_inv)

    seeds = find_seeds(sure_bg_flood)

    flood_fill(seeds, sure_bg_flood, UNKNOWN_COLOR, BG_COLOR)

    cv2.imwrite(flooded_output_path, sure_bg_flood)

    return sure_bg_flood


def track_image(original_image_path, flooded_background):
    img = cv2.imread(original_image_path, 0)
    sure_fg = Filters.threshold(img, 180, False)

    tracked_edges = np.copy(flooded_background)
    contours = track_all_particles(sure_fg, tracked_edges)

    nparr = np.asarray(contours)
    np.save("contours.npy", nparr)

    return contours


class FloodEdgeDetector:
    @staticmethod
    def find_contours(input_path):
        flood_path = 'flood_output.tif'

        bg_flood = flood_image(input_path, flood_path)
        contours = track_image(input_path, bg_flood)
        return contours
