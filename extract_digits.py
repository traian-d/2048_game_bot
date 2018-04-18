import os
import numpy as np
import cv2


def is_digit_bounding_rect(width, height, image_shape):
    '''

    :param width:
    :param height:
    :param image_shape:
    :return:
    '''
    img_h = image_shape[0]
    img_w = image_shape[1]
    return (0.9 * img_h <= height <= 1.1 * img_h) and (0.9 * img_w <= width <= 1.1 * img_w)


def is_child_of(contour, parent_id):
    '''

    :param contour:
    :param parent_id:
    :return:
    '''
    return contour[3] == parent_id


def extract_digit_contours(contours, image_shape):
    '''

    :param contours:
    :param image_shape:
    :return:
    '''
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


#filter top level contour with box the size of the image
# and child contours that are not directly nested in the top level
def crop_cell_from_grid(grid_image, top_left, cell_width, cell_height):
    '''

    :param grid_image:
    :param top_left:
    :param cell_width:
    :param cell_height:
    :return:
    '''
    return grid_image[top_left[0]: top_left[0] + cell_height,
           top_left[1]: top_left[1] + cell_width]


def crop_digits(row_regions, col_regions, img):
    '''

    :param row_regions:
    :param col_regions:
    :param img:
    :return:
    '''
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
    '''

    :param image:
    :param up:
    :param down:
    :param left:
    :param right:
    :return:
    '''
    height, width = image.shape
    up_trim = int(up * height)
    down_trim = int(down * height)
    left_trim = int(left * width)
    right_trim = int(right * width)
    return image[up_trim:(height - down_trim), left_trim:(width - right_trim)]


def make_digit_boxes(thresh_image):
    '''

    :param thresh_image:
    :return:
    '''
    row_density = np.sum(thresh_image/255, axis=0)
    col_density = np.sum(thresh_image/255, axis=1)
    return get_digit_regions(row_density, 2), get_digit_regions(col_density, 2)


def get_digit_regions(density, box_adj):
    '''

    :param density:
    :param box_adj:
    :return:
    '''
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
    '''

    :param start:
    :param stop:
    :return:
    '''
    return stop - start > 6


def extract_all_cropped_digits(single_cell_dir, cropped_dir, digit_dirs):
    '''

    :param single_cell_dir:
    :param cropped_dir:
    :param digit_dirs:
    :return:
    '''
    for digit_dir in digit_dirs:
        counter = 0
        for filename in os.listdir(single_cell_dir + digit_dir):
            input_file_path = make_input_file_path(digit_dir, filename, single_cell_dir)
            print(input_file_path)
            img = cv2.imread(input_file_path)
            thresh = threshold_image(img)
            digits = crop_single_image(thresh)
            for digit in digits:
                write_digit_to_file(make_output_file_path(cropped_dir, digit_dir, counter), digit)
                counter += 1


def make_minority_white(image):
    '''

    :param image:
    :return:
    '''
    h, w = image.shape
    white_count = np.sum(np.sum(image / 255, axis=0))
    total_count = h*w
    if white_count > total_count - white_count:
        return 255 * abs(image - 255)
    return image


def crop_single_image(img):
    '''
    Expects thresholded image
    :param img:
    :return:
    '''
    row_regions, col_regions = make_digit_boxes(img)
    return crop_digits(row_regions, col_regions, img)


def threshold_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1.1, -1.1, -1.1], [-1, 9, -1], [-1, -1, -1]])
    blur = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow("", thresh)
    # cv2.waitKey(0)
    thresh = trim_image_sides(thresh, 0.1, 0.1, 0.0, 0.05)
    thresh = make_minority_white(thresh)
    return thresh


def write_digit_to_file(file_path, digit):
    '''

    :param file_path:
    :param digit:
    :return:
    '''
    cv2.imwrite(file_path, digit)
    print(file_path)


def make_input_file_path(digit_dir, filename, single_cell_dir):
    '''

    :param digit_dir:
    :param filename:
    :param single_cell_dir:
    :return:
    '''
    return single_cell_dir + digit_dir + filename


def make_output_file_path(cropped_dir, digit_dir, image_id):
    '''
    
    :param cropped_dir:
    :param digit_dir:
    :param image_id:
    :return:
    '''
    return cropped_dir + digit_dir + '035_im' + str(image_id) + '.jpg'


if __name__ == "__main__":
    #First clear all contents from the cropped digit directories
    for digit_dir in digit_dirs:
        os.system("rm " + "cropped/" + digit_dir + "*")

    # digit_dirs = [str(2**i) + '/' for i in range(1, 12)]
    # digit_dirs = ['32/','256/','512/','1024/','2048/']
    digit_dirs = ['035/']

    # extract_all_cropped_digits('single_cells/', 'cropped/', digit_dirs)
    # extract_all_cropped_digits('single_cells/', 'cropped/', digit_dirs)
    extract_all_cropped_digits('single_cells/', 'cropped/', digit_dirs)

    # crop_single_image(cv2.imread('single_cells/1024_max/1024/im188.jpg'))
