import sys
import time
import numpy as np

from edge_detector import FloodEdgeDetector
from stats import Statistics
from utilities import ImageUtilities


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    total_start = time.clock()
    #contours = np.load("contours.npy")
    contours = FloodEdgeDetector.find_contours(input_file)
    print("Contours Time", time.clock() - total_start)
    lines = Statistics.evaluate_contours(output_file, contours)
    print("Overall Time", time.clock() - total_start)
    ImageUtilities.show_contours_in_image(input_file, contours, lines)

main()