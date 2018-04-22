import cv2
import numpy as np
from sklearn.externals import joblib

import digit_recognition as dr
import extract_digits as ed
import find_contours as fc

trained_mlp = joblib.load('MLP_class_20neurons_sgd.pkl')


def get_grid_points_in_original_image(pts_in_warped, original_transform):
    grid_warped_back = cv2.perspectiveTransform(np.array([np.array(pts_in_warped, dtype="float32")]), np.linalg.inv(original_transform))
    return grid_warped_back


def predict_single_cell_content(cell, trained_model):
    thresh = ed.threshold_image(cell)
    digits = ed.crop_single_image(thresh)
    prediction = 0
    for digit in digits:
        if not digit.any():
            continue
        std_digit = dr.standardize_image_shape(digit)
        std_digit = std_digit.reshape(1, -1) / 255.
        prediction *= 10
        prediction += trained_model.predict(std_digit)
    if not is_power_of_two(prediction):
        return 0
    return prediction


def is_power_of_two(number):
    if number == 0:
        return False
    if number == 2:
        return True
    if number % 2 == 0:
        return is_power_of_two(number / 2)
    return False


def predict_all_cell_contents(grid_image, grid_points, cell_width, cell_height):
    predictions = []
    for point in grid_points:
        cell = fc.crop_cell_from_grid(grid_image, point, cell_width, cell_height)
        prediction = predict_single_cell_content(cell, trained_mlp)
        if prediction == 0:
            predictions.extend([0])
            continue
        predictions.extend(prediction)
    return predictions


def add_points_to_image(image, point_grid, values):
    for i in range(len(values)):
        cv2.putText(image, str(values[4 * (i % 4) + (i / 4)]),
                    (int(point_grid[0, i, 0] * 1.08), int(point_grid[0, i, 1] * 1.08 + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 255), lineType=2)
    return image


def make_empty_image_duplicate(image):
    (height, width, channels) = image.shape
    return np.zeros((height, width, 3), np.uint8)


def predict(image):
    warped_image, grid_points, transformation_matrix, cw, ch = fc.compute_transformation_components(image)
    predictions = predict_all_cell_contents(warped_image, grid_points, cw, ch)
    original_points = get_grid_points_in_original_image(grid_points, transformation_matrix)
    return original_points, predictions


if __name__ == "__main__":
    img = cv2.imread('test_photos/image114.jpg')
    orig_positions, preds = predict(img)
    print(preds)
    cv2.imshow("",ed.threshold_image(img))
    cv2.waitKey(0)
    empty_img = make_empty_image_duplicate(img)
    add_points_to_image(img, orig_positions, preds)
    cv2.imshow("", img)
    cv2.waitKey(0)
    print(fc.contains_game_grid(img))
