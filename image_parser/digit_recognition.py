from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

IMAGE_HEIGHT = 26
IMAGE_WIDTH = 22


def standardize_image_shape(image):
    return cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))


def read_and_standardize_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        image = cv2.imread(folder_path + filename)
        std_image = standardize_image_shape(image)
        gray = cv2.cvtColor(std_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        images.append(thresh.flatten() / 255.)
    return images


if __name__ == "__main__":
    X_train = []
    y_train = []

    X_test = []
    y_test = []
    for i in range(10):
        X_loc = read_and_standardize_images('/home/data/PycharmProjects/2048 data/cropped/single_digits/' + str(i) + '/')
        y_loc = np.ones((len(X_loc),), dtype=np.int) * i
        X_tr, X_te, y_tr, y_te = train_test_split(X_loc, y_loc, test_size=0.3, random_state=42)
        X_train.extend(X_tr)
        X_test.extend(X_te)
        y_train.extend(y_tr)
        y_test.extend(y_te)

    X_test = np.array(X_test)
    X_train = np.array(X_train)
    y_test = np.array(y_test)
    y_train = np.array(y_train)

    mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=15, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-5, random_state=1,
                        learning_rate_init=.1)

    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))

    # mlp = joblib.load('MLP_class_20neurons_sgd.pkl')

    joblib.dump(mlp, 'MLP_class_20neurons_sgd.pkl')

    fig, axes = plt.subplots(4, 4)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(IMAGE_WIDTH, IMAGE_HEIGHT), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()
