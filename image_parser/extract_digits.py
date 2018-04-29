import os
import numpy as np
import cv2
import data_paths as dp


def extract_digit_contours(contours, image_shape):

    # [Next, Previous, First_Child, Parent]
    actual_contours = contours[1]
    hierarchy = contours[2]
    top_level_contour = -1
    digit_contour_rects = []
    for i in range(len(actual_contours)):
        # compute the bounding box of the contour
        contour = actual_contours[i]
        cont_hierarchy = hierarchy[0][i]
        x, y, w, h = cv2.boundingRect(contour)
        if is_digit_bounding_rect(w, h, image_shape):
            top_level_contour = i
            continue
        if is_child_of(cont_hierarchy, top_level_contour):
            digit_contour_rects.append((x, y, w, h))
    return digit_contour_rects


def is_digit_bounding_rect(width, height, image_shape):
    img_h = image_shape[0]
    img_w = image_shape[1]
    return (0.9 * img_h <= height <= 1.1 * img_h) and (0.9 * img_w <= width <= 1.1 * img_w)


def is_child_of(contour, parent_id):
    return contour[3] == parent_id


def crop_cell_from_grid(grid_image, top_left, cell_width, cell_height):

    return grid_image[top_left[0]: top_left[0] + cell_height,
           top_left[1]: top_left[1] + cell_width]


def crop_digits(row_regions, col_regions, img):

    digits = []
    for row_reg in row_regions:
        for col_reg in col_regions:
            x = row_reg[0]
            y = col_reg[0]
            h = col_reg[1] - col_reg[0]
            w = row_reg[1] - row_reg[0]
            digits.append(crop_cell_from_grid(img, (y, x), w, h))
    return digits


def trim_image_sides(image, up, down, left, right):

    height, width = image.shape
    up_trim = int(up * height)
    down_trim = int(down * height)
    left_trim = int(left * width)
    right_trim = int(right * width)
    return image[up_trim:(height - down_trim), left_trim:(width - right_trim)]


def make_digit_boxes(thresh_image):

    row_density = np.sum(thresh_image/255, axis=0)
    col_density = np.sum(thresh_image/255, axis=1)
    return get_digit_regions(row_density, 2), get_digit_regions(col_density, 2)


def get_digit_regions(density, box_adj):

    regions = []
    start = 0
    in_region = False
    for i in range(len(density)):
        if density[i] < 3:
            in_region = False
            if region_is_long_enough(start, i):
                regions.append([max(0, start - box_adj), i + box_adj])
            start = i
            continue
        if not in_region:
            in_region = True
            start = i
    return regions


def region_is_long_enough(start, stop):

    return stop - start > 6


def make_minority_white(image):

    h, w = image.shape
    white_count = np.sum(np.sum(image / 255, axis=0))
    total_count = h*w
    if white_count > total_count - white_count:
        return 255 * abs(image - 255)
    return image


def crop_single_image(img):
    """
    Expects thresholded image
    :param img:
    :return:
    """
    row_regions, col_regions = make_digit_boxes(img)
    return crop_digits(row_regions, col_regions, img)


def threshold_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1.1, -1.1, -1.1], [-1, 9, -1], [-1, -1, -1]])
    blur = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = trim_image_sides(thresh, 0.1, 0.1, 0.0, 0.05)
    thresh = make_minority_white(thresh)
    return thresh


def extract_all_cropped_digits():
    counter = 0
    for filename in os.listdir(dp.CELLS_PATH):
        img = cv2.imread(dp.CELLS_PATH + filename)
        thresh = threshold_image(img)
        digits = crop_single_image(thresh)
        for digit in digits:
            cv2.imwrite(dp.SINGLE_DIGITS_PATH + "digit" + str(counter) + ".jpg", digit)
            print(counter)
            counter += 1


if __name__ == "__main__":
    os.system("rm -rf " + dp.SINGLE_DIGITS_PATH + "*")
    extract_all_cropped_digits()

