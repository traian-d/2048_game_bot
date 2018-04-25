import os
import math
import numpy as np
import cv2
from scipy import spatial as sp
import data_paths as dp


class Quadrangle:
    """
    This class provides a few behaviors and measurements around a contour that represents a quadrangle.
    In particular it can determine if a contour (approximately) represents a rectangle.
    """
    APPROX_FACTOR = 0.1

    def __init__(self, contour):
        # Reshape contour in top-left, top-right, bottom-right,
        # and bottom-left order
        pts = contour.reshape(4, 2)
        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis=1)
        self.top_left = pts[np.argmin(s)]
        self.bottom_right = pts[np.argmax(s)]
        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis=1)
        self.top_right = pts[np.argmin(diff)]
        self.bottom_left = pts[np.argmax(diff)]

        # Sides and diagonals of the rectangle
        self.top = sp.distance.euclidean(self.top_left, self.top_right)
        self.bottom = sp.distance.euclidean(self.bottom_left, self.bottom_right)
        self.left = sp.distance.euclidean(self.top_left, self.bottom_left)
        self.right = sp.distance.euclidean(self.top_right, self.bottom_right)
        self.princ = sp.distance.euclidean(self.top_left, self.bottom_right)
        self.sec = sp.distance.euclidean(self.top_right, self.bottom_left)

    def is_approx_rectangle(self):
        """

        :return: Boolean saying whether the quadrangle is approximately a rectangle
        """
        return \
            (1 - self.APPROX_FACTOR) * self.top <= self.bottom <= (1 + self.APPROX_FACTOR) * self.top and\
            (1 - self.APPROX_FACTOR) * self.left <= self.right <= (1 + self.APPROX_FACTOR) * self.left and\
            (1 - self.APPROX_FACTOR) * self.sec <= self.princ <= (1 + self.APPROX_FACTOR) * self.sec

    def array_form(self):
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = self.top_left
        rect[2] = self.bottom_right
        rect[1] = self.top_right
        rect[3] = self.bottom_left
        return rect


def determine_grid_contour(image):
    """
    This method will return the contour of a quadrangle corresponding to the 2048 game grid.
    The assumption is that the grid is the largest quadrangle in the input image.
    :param image: An image as outputted for example by cv2.imread
    :return: None or a contour corresponding to the largest quadrangle in the image
    """
    cont = identify_quadrangles(image, True)
    if not cont:
        return None
    return cont[0]


def identify_quadrangles(image, find_first):
    """
    :param image: An image as outputted for example by cv2.imread
    :param find_first: Boolean noting whether only the first quadrangle by size should be returned. find_first = False
        will return all identified quadrangles.
    :return: The quadrangle(s) identified in the image, in descending size order.
    """
    edges = extract_edges(image)
    cont = find_contours(edges, 16)
    return keep_quadrangles(cont, find_first)


def extract_edges(image):
    """
    The method converts the image to grayscale, blurs it, and find edges in the image.
    :param image: An image as outputted for example by cv2.imread
    :return: The edges found in the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 13, 13)
    return cv2.Canny(gray, 25, 14)


def find_contours(edges, contours_to_keep):
    """
    This method identifies contours in a two-color image with already detected edges.
    The image will return the detected contours in the order of descending size.
    :param edges: A two-color image with already detected edges
    :param contours_to_keep: Number of contours to return
    :return:
    """
    find_contours_out = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # For OpenCV 2 contours are the first argument, for OpenCV 3 they are the second one
    contours = (find_contours_out[0] if cv2_version() == 2 else find_contours_out[1])
    return sorted(contours, key=cv2.contourArea, reverse=True)[:contours_to_keep]


def cv2_version():
    return int(cv2.__version__[0])


def keep_quadrangles(contours, find_first):
    """
    For an array of contours this method outputs an array with those contours that are (approximately) quadrangles.
    If find_first is True the method will simply return the first contour that is a quadrangle.
    :param contours: An array of contours, expected in descending order of size.
    :param find_first: Boolean noting whether only the first quadrangle by size should be returned. find_first = False
        will return all identified quadrangles.
    :return:
    """
    output_contours = []
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            output_contours.append(approx)
            if find_first:
                break
    return output_contours


def contains_game_grid(image):
    """
    A method that determines if an input image contains the 2048 game grid.
    This is done by first identifying a potential contour of the entire grid.
    If such a contour is found, the image is warped to bring it to a rectangular shape.
    Within this rectangular shape we search for other quadrangles.
    If all of the quadrangles found are rectangular, we assume that they correspond to the cells of the 2048 grid.
    :param image:
    :return: Either a contour corresponding to the game grid, or None.
    """
    cont = determine_grid_contour(image)
    if cont is None:
        return None
    warped_image = warp_image(cont, image)
    quadrangles = identify_quadrangles(warped_image, False)
    if not quadrangles:
        return None
    for rectangle in quadrangles:
        rect = Quadrangle(rectangle)
        if not rect.is_approx_rectangle():
            return None
    return cont


def compute_transformation_components(image):
    """
    For an image this method computes a series of values that can be used later on in further transformations.
    :param image: An image.
    :return:
    """
    contour = determine_grid_contour(image)
    warped_image = warp_image(contour, image)
    transformation_matrix, max_width, max_height = get_warping_parameters(contour)
    ch, cw, grid_points = get_grid_coords_in_warped_image(warped_image)
    return warped_image, grid_points, transformation_matrix, cw, ch


def get_grid_coords_in_warped_image(warped_image):
    height, width, channels = warped_image.shape
    ch, cw = calculate_cell_dimensions(width, height)
    bh, bw = calculate_border_dimensions(width, height)
    grid_points = get_cell_grid_coords(bw, bh, cw, ch)
    return ch, cw, grid_points


def warp_image(contour, image):
    """
    A method that warps an image containing to a 2048 game grid so that the grid becomes rectangular.
    :param contour: The contour of the grid in the image
    :param image: An image containing a 2048 game grid
    :return: A warped version of the image.
    """
    M, max_height, max_width = get_warping_parameters(contour)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def get_warping_parameters(contour):
    """
    :param contour: A (quasi)rectangular contour.
    :return: A transformation matrix needed to transform the contour to a rectangle as well as height and width of
    the transformed contour.
    """
    quad = Quadrangle(contour)
    max_width = int(max(quad.top, quad.bottom))
    max_height = int(max(quad.left, quad.right))
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(quad.array_form(), dst)
    return M, max_height, max_width


def calculate_cell_dimensions(width, height):
    """
    For a given width and height of the game grid, this method calculates the dimensions of a single cell.
    The magic numbers correspond to the proportions of the 2048 grid in the actual game.
    :param width: The width of the grid
    :param height: The height of the grid
    :return: height and width of a single cell from the grid
    """
    return int(math.ceil((106.25 / 500) * height)), int(math.ceil((106.25 / 500) * width))


def calculate_border_dimensions(width, height):
    """
    For a given width and height of the game grid, this method calculates the dimensions of the border surrounding the cell.
    The magic numbers correspond to the proportions of the 2048 border in the actual game.
    :param width: The width of the grid
    :param height: The height of the grid
    :return: height and width of the grid border
    """
    return int(math.ceil((15.0 / 500) * height)), int(math.ceil((15.0 / 500) * width))


def get_cell_grid_coords(bw, bh, cw, ch):
    """
    Assuming that the game grid starts from the position [0, 0] in the top left corner,
    this method will return the coordinates of the top left corner of each cell in the grid.
    :param bw: Border width
    :param bh: Border height
    :param cw: Cell width
    :param ch: Cell height
    :return: An array of coordinates for each cell in the game grid
    """
    l1x = bw + cw + bw
    l2x = bw + 2 * cw + 2 * bw
    l3x = bw + 3 * cw + 3 * bw

    l1y = bh + ch + bh
    l2y = bh + 2 * ch + 2 * bh
    l3y = bh + 3 * ch + 3 * bh
    # point coordinates are of the form [y, x] and they start at [0, 0] in the top left corner
    return [
         [bw, bw],  [bw, l1x],  [bw, l2x],  [bw, l3x],
        [l1y, bw], [l1y, l1x], [l1y, l2x], [l1y, l3x],
        [l2y, bw], [l2y, l1x], [l2y, l2x], [l2y, l3x],
        [l3y, bw], [l3y, l1x], [l3y, l2x], [l3y, l3x],
    ]


def crop_cell_from_grid(grid_image, top_left, cell_width, cell_height):
    """
    This method is meant to extract a single cell from the 2048 game grid. It does this by returning a sub-image located
    at certain coordinates, with a given width and height.
    :param grid_image: An image containting exclusively the 2048 game grid.
    :param top_left: The top left corner coordinates of the cell to be extracted.
    :param cell_width: The width of the cell to be exracted
    :param cell_height: The height of the cell to be exracted
    :return: A sub-image corresponding to a singe game cell
    """
    return grid_image[top_left[0]: top_left[0] + cell_height,
           top_left[1]: top_left[1] + cell_width]


if __name__ == "__main__":
    counter = 0
    for filename in os.listdir(dp.RAW_IMAGES_PATH):
        img = cv2.imread(dp.RAW_IMAGES_PATH + filename)
        if img is None:
            continue
        contour = identify_quadrangles(img, True)
        warped_contour = warp_image(contour[0], img)
        height, width, channels = warped_contour.shape
        ch, cw = calculate_cell_dimensions(width, height)
        bh, bw = calculate_border_dimensions(width, height)
        grid_points = get_cell_grid_coords(bw, bh, cw, ch)
        for point in grid_points:
            cv2.imwrite(dp.CELLS_PATH + 'im' + str(counter) + '.jpg', crop_cell_from_grid(warped_contour, point, cw, ch))
            counter += 1
        print(filename)
