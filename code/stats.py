import numpy as np
import time
import math


def get_distance(x1, y1, x2, y2):
    """
    Computes distance between two 2D points
    :param x1: x coord of first point
    :param y1: y coord of first point
    :param x2: x coord of second point
    :param y2: y coord of second point
    :return: float: distance between points
    """
    return np.linalg.norm(np.array([x1-x2, y1-y2]))  # point distance

def find_max_length_base(xs, ys):
    max_len = 0
    max_x1 = -1
    max_x2 = -1
    max_y1 = -1
    max_y2 = -1
    for x1, y1 in zip(xs, ys):
        for x2, y2 in zip(xs, ys):
            dist = get_distance(x1, y1, x2, y2) # point distance
            if (dist >= max_len):
                max_len = dist
                max_x1 = x1
                max_y1 = y1
                max_x2 = x2
                max_y2 = y2

    return max_x1, max_y1, max_x2, max_y2

def find_max_length_base2(xs, ys):
    """
    Function finds max length line inside object. Object is defined by border points.
    :param xs: x coords of border point of object
    :param ys: y coords of border point of object
    :return: float: maximum distance between 2 border points
    """
    max_len = 0
    max_i = -1
    max_j = -1
    point_cnt = len(xs)
    a = 0
    for i in range(point_cnt):
        x1 = xs[i]
        y1 = ys[i]
        for j in range(a, point_cnt):
            x2 = xs[j]
            y2 = ys[j]
            dx = x1 - x2
            dy = y1 - y2
            dist = math.sqrt(dx*dx + dy*dy)

            if dist >= max_len:
                max_len = dist
                max_i = i
                max_j = j
        a += 1

    return xs[max_i], ys[max_i], xs[max_j], ys[max_j]

def line_between_points(x1, y1, x2, y2):
    """
    Function returns coeficients of function defined by two 2D points given in parameters
    :param x1: x coord of first point
    :param y1: y coord of first point
    :param x2: x coord of second point
    :param y2: y coord of second point
    :return: (float, float): first componet is slope, second is shift
    """
    coefficients = np.polyfit([x1, x2], [y1, y2], 1)
    return coefficients[0], coefficients[1]

def lower_to_higher(p1, p2, c):
    if c == 1:
        a = p1[0]
        b = p2[0]
    else:
        a = p1[1]
        b = p2[1]

    if (a > b):
        return p2, p1
    else:
        return p1, p2

def get_perpendicular_line(x, y, m):
    """
    Function computes coeficients (slope, shift) of a line which is perpendicular to line with slope m, in point [x, y]
    :param x:  x coord of point
    :param y:  y coord of point
    :param m:  slope
    :return: (float, float): slope and shift of perpendicular line
    """
    b = y + 1/m * x
    return -(1/m), b

def filter_res(res):
    max_dist = 0
    max_x1 = -1
    max_x2 = -1
    max_y1 = -1
    max_y2 = -1
    a = 0
    for i in range(len(res)):
        for j in range(a, len(res)):
            item = res[i]
            item2 = res[j]
            dist = get_distance(item[0], item[1], item2[0], item2[1])

            if (dist >= max_dist):
                max_dist = dist
                max_x1 = item[0]
                max_y1 = item[1]
                max_x2 = item2[0]
                max_y2 = item2[1]
        a += 1

    #print(max_x1, max_y1, max_x2, max_y2)
    return max_dist, (max_x1, max_y1, max_x2, max_y2)

def check_validity(p, res):
    """
    This function checks whether candidate thickness line point does not go outside of the point
    :param p: candidate point
    :param res: rest of the points (array of 2 item arrays)
    :return: True if line is valid
    """
    res_new = []
    x1, y1, x2, y2 = p
    for item in res:
        d1 = get_distance(x1, y1, item[0], item[1])
        d2 = get_distance(x2, y2, item[0], item[1])
        if d1 > 2 and d2 > 2:
            res_new.append(item)

    return res_new == []


def iterate_x(p1, p2, m, b, xs, ys, radius):
    """
    This function computes ax thickness line. It must be perpendicular to max len line (defined by parameters p1, p2 and slope m and shift b).
    Edge points of max thickness line will be on border point, which are saved in xs and ys.
    :rtype: returns maximum thickness computes, parameter of said line (coords + slope + shift)
    """
    p1, p2 = lower_to_higher(p1, p2, 1)

    max_dist = 0
    max_m = 0
    max_b = 0
    max_p = ()
    for i in range(p1[0], p2[0]+1):
        x = i
        y = x * m + b

        pm, pb = get_perpendicular_line(x, y, m)

        res = []
        for xx, yy in zip(xs, ys):
            tmp_yy = pm * xx + pb
            if np.abs(yy - tmp_yy) < radius:
                res.append([xx, yy])

        if len(res) >= 2:
            dist, p = filter_res(res)
            check = check_validity(p, res)
            # if (dist >= max_dist and check == False):
            #     print(res)
            #     print("PROBLEM")
            if (dist >= max_dist and check):
                max_dist = dist
                max_m = pm
                max_b = pb
                max_p = p

    return max_dist, max_p, (max_m, max_b)

def iterate_y(p1, p2, m, b, xs, ys, radius):
    """
    similar to iterate_x, except it iterates over max len line in direction of y axis
    """
    p1, p2 = lower_to_higher(p1, p2, 2)

    max_dist = 0
    max_m = 0
    max_b = 0
    max_p = ()
    for i in range(p1[1], p2[1]+1):
        y = i
        x = (y-b)/m

        pm, pb = get_perpendicular_line(x, y, m)

        res = []
        for xx, yy in zip(xs, ys):
            tmp_xx = (yy-pb)/pm
            if np.abs(xx - tmp_xx) < radius:
                res.append([xx, yy])

        #print(res)
        if len(res) >= 2:
            dist, p = filter_res(res)
            check = check_validity(p, res)
            # if ((dist >= max_dist) and check == False):
            #     print("PROBLEM")
            if (dist >= max_dist and check):
                max_dist = dist
                max_m = pm
                max_b = pb
                max_p = p

    return max_dist, max_p, (max_m, max_b)

def find_max_thickness(p1, p2, xs, ys, radius):
    """
    Function computes maximum thickness line to give max length. Thickness line is perpendicular to max length line and cannot go outside of the object
    :param p1: (int, int): first point of max length line
    :param p2: int, int): second point of max length line
    :param xs: x border coords
    :param ys: y border coords
    :param radius: sensitivity of algorithm
    :return: returns length of maximum thickness line, defining points and line coeficients
    """
    # y = a*x + b
    m, b = line_between_points(p1[0], p1[1], p2[0], p2[1])
    if m > 1 or m < -1:
        #print("x")
        max_dist, max_p, (max_m, max_b) = iterate_x(p1, p2, m, b, xs, ys, radius)
    else:
        #print("y")
        max_dist, max_p, (max_m, max_b) = iterate_y(p1, p2, m, b, xs, ys, radius)

    return max_dist, max_p, (max_m, max_b)

# def edge_px(im, i, j):
#     try:
#         up = im[i+1][j]
#     except:
#         up = False
#     try:
#         down = im[i-1][j]
#     except:
#         down = False
#     try:
#         left = im[i][j+1]
#     except:
#         left = False
#     try:
#         right = im[i][j-1]
#     except:
#         right = False
#
#     return not(up and down and left and right)

def get_AABB(xs, ys):
    """
    Computes AABB axis alligned bounding box
    :param xs: x coords
    :param ys: y coords
    :return: (float, float): width and height
    """
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    return (max_x - min_x), (max_y - min_y)


def print_to_csv(values, filename):
    with open(filename, 'w+') as file:
        id = 1
        file.write("Part #,Width,Height,Max Length,Thickness\n")
        for item in values:
            line = ""
            line += str(id)
            for i in range(0, len(item)):
                line += "," + str(item[i])
            file.write(line + "\n")
            id+=1


class Statistics:
    @staticmethod
    def evaluate_contours(output_filepath, contours):
        data = contours
        values = []
        lines = []
        start_all = time.clock()
        for object in data:
            start = time.clock()
            xs = []
            ys = []
            for x, y in object:
                xs.append(x)
                ys.append(y)

            A, B = get_AABB(xs, ys)

            r1 = find_max_length_base2(xs, ys)
            x1, y1, x2, y2 = r1

            max_len_dist = get_distance(x1, y1, x2, y2)

            # start = time.clock()
            radius = 0.7
            r2, p, p2 = find_max_thickness([x1, y1], [x2, y2], xs, ys, radius)
            x1, y1, x2, y2, = p
            max_thick_dist = get_distance(x1, y1, x2, y2)

            print("-----------")

            lines.append((r1, p))
            values.append([A, B, max_len_dist, max_thick_dist])
            print([A, B, max_len_dist, max_thick_dist])
            print("Evaluation Time:", time.clock() - start)
        print("Total Evaluation Time: ", time.clock() - start_all)

        print_to_csv(values, output_filepath)

        return lines
