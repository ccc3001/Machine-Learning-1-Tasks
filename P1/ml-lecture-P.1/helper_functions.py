import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import random
import math
from plot_functions import visualize_scalar_product_2D, get_angle
import random
from PIL import Image, ImageOps


def generate_random_vector(dim=2, use_numpy=False):
    """
    Creates a random vector with homogeneously sampled values in R^N.
    :param dim: the dimension of the vector
    :param seed: a random seed
    :return: A random vector of length dim in the interval [0,1) or [0,1]
    """
    rnd_vec = [random.random()] * dim # TODO: Overwrite this with your implementation.
    if not use_numpy:
        # <START Your code here>
        rnd_vec = []
        for i in range(dim):
            e = random.random()
            rnd_vec.append(e)
        # <END Your code here>
    else:
        # <START Your code here>
        rnd_vec = np.random.rand(dim)
        # <END Your code here>

    return rnd_vec


def flatten_matrix(x1):
    """
    Transforms a n_1 x n_2 x ... x n_m matrix into a vector of length n_1 * n_2 * ... * n_m
    :param x1: The matrix
    :return: The flattened vector
    """
    x1 = np.resize(x1, (x1.size))
    return x1


# Helper function to define the inner product.
def inner_prod(x1, x2, use_numpy=False):
    """
    Computes the inner product of two vectors
    :param x1: first vector
    :param x2: second vector
    :return: the inner product
    """
    assert len(x1) == len(x2), "Error, cannot compute the inner product because the vector lengths are not equal"
    p = 0 # TODO: Overwrite this value with your implementation
    if not use_numpy:
        # <START Your code here>
        for i in range(len(x1)):
            p = p + x1[i-1] * x2[i-1]
        # <END Your code here>
    else:
        # <START Your code here>
        p = np.inner(x1,x2)
        # <END Your code here>

    return p


# Helper function to determine the magnitude of a vector
def mag(x, use_numpy=False):
    """
    Computes the magnitude of a vector
    :param x: the vector
    :return: the magnitude of the vector
    """


     # TODO: Overwrite this value with your implementation
    if use_numpy is False:
        # <START Your code here>
        m = 0
        for i in range(len(x)):
            m = x[i - 1] * x[i - 1] + m
        m = math.sqrt(m)
        # <END Your code here>
    else:
        # <START Your code here>
        m = np.linalg.norm(x)
        # <END Your code here>
    return m


# Helper function to determine the radius of a set of points
def get_radius(D):
    """
    Computes the radius of a point cloud as the distance to the point that is farthest from the centre
    :param D: A list of vectors
    :return: The radius
    """
    max_r = 0
    # <START Your code here>
    print("Not yet implemented")
    # <END Your code here>
    return max_r


def vector_rotate2D(x, deg):
    """
    Computes the rotation of a vector
    :param x: the 2D vector to rotate as a list with 2 elements
    :param deg: the angle in degrees (not rad!) for the CCW rotation
    :return: the rotated 2D vector as a list with 2 elements (the x and y coordinate)
    """

    theta = np.deg2rad(deg)
    x_rotated = [0, 0]
    x_rotated[0] = x[0] * np.cos(theta) - x[1] * np.sin(theta)
    x_rotated[1] = x[0] * np.sin(theta) + x[1] * np.cos(theta)
    return x_rotated


def gen_lin_sep_dataset(n_samples):
    """
    Generates a dataset of 2D points that are linearly separable in two classes.
    :param n_samples:  Number of points
    :return: A tuple where the first element is the points in one class and the second is the points in the other class.
    """
    separable = False
    while not separable:
        D, c_idx = datasets.make_classification(n_samples=n_samples,  n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
        red = D[c_idx == 0]
        blue = D[c_idx == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])

    C = D[c_idx == 1]
    NotC = D[c_idx == 0]
    return C, NotC


# Normalize a 2d vector
def normalize_2d(vec):
    """
    Normalize a 2d vector
    :param vec: the vector to normalize
    :return: the normalized vector
    """
    normalized_vec = [x / mag(vec) for x in vec]
    return normalized_vec


def project_2D(x1, x2):
    """
    Given two vectors, compute the projection of one vector onto another.
    :param x1: The vector to project onto the other vector
    :param x2: The vector on which to project
    :return: the projected vector
    """
    # Normalize the vector to project onto
    normalized_x2 = normalize_2d(x2)

    # Get the magnitude of the projected vector by computing its inner product with the normalized vector to project onto.
    proj_mag = inner_prod(x1, normalized_x2)

    # Compute the projection
    proj = [x * proj_mag for x in normalized_x2]
    return proj


def plot_inner_product(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)

    inner_prod_magnitude = inner_prod(x1,x2)
    visualize_scalar_product_2D(x1, x2, inner_prod_magnitude, vector_rotate2D)


def load_image(filename="cat.jpg", scale_to_size=None, grayscale=True):
    image = Image.open(filename)
    if scale_to_size is not None:
        image = image.resize(scale_to_size)
    if grayscale:
        image = ImageOps.grayscale(image)
    data = np.asarray(image, dtype=np.float)
    return data


def show_image(numpy_image, scale_to_width=200):
    image = Image.fromarray(numpy_image)
    scale_factor = scale_to_width / image.size[0]
    image = ImageOps.scale(image, scale_factor, resample=Image.BOX)
    image.show()


def compute_angle(x1, x2):
    """
    Computes the angle (in degrees) between two vectors.
    :param x1: the first vector
    :param x2: the second vector
    :return: the angle in degrees
    """
    ang = 45 # TODO: overwrite this value with your implementation
    # <START Your code here>
    ang = inner_prod(x1,x2,True)/(mag(x1,True)*mag(x2,True))
    # <END Your code here>
    return ang



x=generate_random_vector(10,False)
y=generate_random_vector(10,False)
print(mag(x,True),":", mag(x,False))
print( inner_prod(x,y,True),":",inner_prod(x,y,False))