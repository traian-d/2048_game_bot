import math
import numpy as np
import cv2
from scipy import spatial as sp


class Quadrangle:
    APPROX_FACTOR = 0.1

    def __init__(self, contour):
        # our output rectangle in top-left, top-right, bottom-right,
        # and bottom-left order
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
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

    def is_rectangle(self):
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


def cv2_version():
    return int(cv2.__version__[0])


def extract_edges(image):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 13, 13)
    return cv2.Canny(gray, 25, 14)


def find_contours(edges, contours_to_keep):
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    # For OpenCV 2 contours are the first argument, for OpenCV 3 they are the second one
    find_contours_out = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = (find_contours_out[0] if cv2_version() == 2 else find_contours_out[1])
    return sorted(cnts, key=cv2.contourArea, reverse=True)[:contours_to_keep]


def find_rectangles(conts, find_first):
    output_contours = []
    # loop over our contours
    for c in conts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # we assume four point contours are rectangles
        if len(approx) == 4:
            output_contours.append(approx)
            if find_first:
                break
    return output_contours


def identify_rectangles(image, find_first):
    edges = extract_edges(image)
    cont = find_contours(edges, 16)
    return find_rectangles(cont, find_first)


def determine_grid_contour(image):
    cont = identify_rectangles(image, True)
    if not cont:
        return None
    return cont[0]


def warp_image(contour, image):
    M, maxHeight, maxWidth = get_warping_parameters(contour)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def get_warping_parameters(contour):
    # now that we have our screen contour, we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    rect = Quadrangle(contour)
    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect.array_form()
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
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
    return math.ceil((106.25 / 500) * height), math.ceil((106.25 / 500) * width)


def calculate_border_dimensions(width, height):
    return math.ceil((15.0 / 500) * height), math.ceil((15.0 / 500) * width)


def get_cell_grid_coords(bw, bh, cw, ch):
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
        if not rect.is_rectangle():
            return None
    return cont
    # ch, cw, grid_points = get_grid_coords_in_warped_img(warped_image)




if __name__ == "__main__":
    print(cv2.__version__[0])
    print(sp.distance.euclidean([2, 2], [1, 1]))
    print(contains_game_grid())
    # counter = 0
    # for image_path in ['035/image' + str(i) + '.jpg' for i in range(380, 410)]:
    #     img = cv2.imread(image_path)
    #     if img is None:
    #         continue
    #     contour = identify_rectangles(img, True)
    #     warped_contour = warp_image(contour[0], img)
    #     # draw_contours(warped_contour, identify_rectangles(warped_contour, False))
    #     height, width, channels = warped_contour.shape
    #     # cv2.waitKey(0)
    #     ch, cw = calculate_cell_dimensions(width, height)
    #     bh, bw = calculate_border_dimensions(width, height)
    #     grid_points = get_cell_grid_coords(bw, bh, cw, ch)
    #     # cv2.imshow(image_path, warped_contour)
    #     # cv2.waitKey(0)
    #
    #     for point in grid_points:
    #         # cv2.imshow('',crop_cell_from_grid(warped_contour, point, cw, ch))
    #         cv2.imwrite('single_cells/im' + str(counter) + '.jpg', crop_cell_from_grid(warped_contour, point, cw, ch))
    #         counter += 1
    #         # cv2.circle(warped_contour,(int(point[1]), int(point[0])),2,(255,255,255),-11)
    #         # cv2.waitKey(0)
    #     print(image_path)
