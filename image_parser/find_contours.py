import math
import numpy as np
import cv2
from scipy import spatial as sp


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
    This method will return the contour of a rectangle corresponding to the 2048 game grid.
    The assumption is that the grid is the largest rectangle in the input image.
    :param image: An image as outputted for example by cv2.imread
    :return: None or a contour corresponding to the largest rectangle in the image
    """
    cont = identify_rectangles(image, True)
    if not cont:
        return None
    return cont[0]


def identify_rectangles(image, find_first):
    """
    :param image: An image as outputted for example by cv2.imread
    :param find_first: Boolean noting whether only the first rectangle by size should be returned. find_first = False
        will return all identified rectangles.
    :return: The rectangle(s) identified in the image, in descending size order.
    """
    edges = extract_edges(image)
    cont = find_contours(edges, 16)
    return keep_rectangles(cont, find_first)


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


def keep_rectangles(contours, find_first):
    """
    For an array of contours this method outputs an array with those contours that are (approximately) rectangles.
    If find_first is True the method will simply return the first contour that is a rectangle.
    :param contours: An array of contours, expected in descending order of size.
    :param find_first: Boolean noting whether only the first rectangle by size should be returned. find_first = False
        will return all identified rectangles.
    :return:
    """
    output_contours = []
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # We assume four point contours are rectangles
        if len(approx) == 4:
            output_contours.append(approx)
            if find_first:
                break
    return output_contours


def warp_image(contour, image):
    M, maxHeight, maxWidth = get_warping_parameters(contour)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def get_warping_parameters(contour):
    rect = Quadrangle(contour)
    # (tl, tr, br, bl) = rect.array_form()
    # widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # # ...and now for the height of our new image
    # heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    # heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    # maxWidth = max(int(widthA), int(widthB))
    # maxHeight = max(int(heightA), int(heightB))
    maxWidth = max(rect.top, rect.bottom)
    maxHeight = max(rect.left, rect.right)
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect.array_form(), dst)
    return M, maxHeight, maxWidth


def calculate_cell_dimensions(width, height):
    """
    For a given width and height of the game grid, this method calculates the dimensions of a single cell.
    The magic numbers correspond to the proportions of the 2048 grid in the actual game.
    :param width: The width of the grid
    :param height: The height of the grid
    :return: height and width of a single cell from the grid
    """
    return math.ceil((106.25 / 500) * height), math.ceil((106.25 / 500) * width)


def calculate_border_dimensions(width, height):
    """
    For a given width and height of the game grid, this method calculates the dimensions of the border surrounding the cell.
    The magic numbers correspond to the proportions of the 2048 border in the actual game.
    :param width: The width of the grid
    :param height: The height of the grid
    :return: height and width of the grid border
    """
    return math.ceil((15.0 / 500) * height), math.ceil((15.0 / 500) * width)


def get_cell_grid_coords(bw, bh, cw, ch):
    """
    Assuming that the game grid starts from the position [0, 0] in the top left corner,
    this method will return the coordinates of the top left corner of each cell in the grid
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

    :param grid_image:
    :param top_left:
    :param cell_width:
    :param cell_height:
    :return:
    """
    return grid_image[top_left[0]: top_left[0] + cell_height,
           top_left[1]: top_left[1] + cell_width]


def get_cell_boundaries(cell_length, bottom_left_coords):
    return [
            bottom_left_coords,  # bottom left point
            [bottom_left_coords[0] + cell_length, bottom_left_coords[1]],  # bottom right
            [bottom_left_coords[0] + cell_length, bottom_left_coords[1] + cell_length],  # top right
            [bottom_left_coords[0], bottom_left_coords[1] + cell_length]  # top left
    ]


def get_grid_coords_in_warped_image(warped_image):
    height, width, channels = warped_image.shape
    ch, cw = calculate_cell_dimensions(width, height)
    bh, bw = calculate_border_dimensions(width, height)
    grid_points = get_cell_grid_coords(bw, bh, cw, ch)
    return ch, cw, grid_points


def contains_game_grid(image):
    cont = determine_grid_contour(image)
    if cont is None:
        return None
    # return True
    warped_image = warp_image(cont, image)
    rectangles = identify_rectangles(warped_image, False)
    if not rectangles:
        return None
    for rectangle in rectangles:
        rect = Quadrangle(rectangle)
        if not rect.is_approx_rectangle():
            return None
    return cont


def compute_transformation_components(image):
    contour = determine_grid_contour(image)
    warped_image = warp_image(contour, image)
    transformation_matrix, max_width, max_height = get_warping_parameters(contour)
    ch, cw, grid_points = get_grid_coords_in_warped_image(warped_image)
    return warped_image, grid_points, transformation_matrix, cw, ch


if __name__ == "__main__":
    counter = 0
    for image_path in ['035/image' + str(i) + '.jpg' for i in range(380, 410)]:
        img = cv2.imread(image_path)
        if img is None:
            continue
        contour = identify_rectangles(img, True)
        warped_contour = warp_image(contour[0], img)
        height, width, channels = warped_contour.shape
        ch, cw = calculate_cell_dimensions(width, height)
        bh, bw = calculate_border_dimensions(width, height)
        grid_points = get_cell_grid_coords(bw, bh, cw, ch)
        for point in grid_points:
            cv2.imwrite('single_cells/im' + str(counter) + '.jpg', crop_cell_from_grid(warped_contour, point, cw, ch))
            counter += 1
        print(image_path)

