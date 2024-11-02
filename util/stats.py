import numpy as np
import math
from copy import deepcopy

def inverse_gaussian_norm(x):
	return 1 - math.exp(-x**2)

def modified_tanh(x):
    if x >= 0:
        return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    else:
        return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)) + 1

def linear_norm(x, x_arr: list):
	""" AKA min max scaling
	"""
	min_scale = -1
	max_scale = 1
	max_x = max(x_arr)
	min_x = min(x_arr)
	norm_x = min_scale + (x - min_x)*(max_scale - min_scale) / (max_x - min_x)
	return norm_x

# def log_mean_norm(x, x_arr: list):
#     """ AKA mean normalization
#     """
#     # mean_x = np.log(np.mean(x_arr))
#     min_scale = -1
#     max_scale = 1
#     scale = abs(max(x_arr)) + abs(min(x_arr))
#     max_x = np.log(max(x_arr) + scale)
#     min_x = np.log(min(x_arr) + scale)
#     x_arr = [np.log(i + scale) for i in x_arr]
#     norm_x = min_scale + (np.log(x + scale) - min_x)*(max_scale - min_scale) / (max_x - min_x)
#     return norm_x

def z_score_normalization(x, x_arr: list):
    temp_list = [i for i in x_arr if not math.isnan(i)]
    if len(temp_list) == 0:
        return 0  # atau nilai default lain yang sesuai
    mean_x = np.mean(temp_list)
    std_x = np.std(temp_list)
    if std_x == 0:
        return 0  # atau nilai default lain yang sesuai
    return (x - mean_x) / std_x

def z_score_normalization_list(x_arr: list):
	mean_x = np.mean(x_arr)
	std_x = np.std(x_arr)
	return [(x - mean_x)/std_x for x in x_arr]

def softmax(x: list):
	return np.exp(x)/sum(np.exp(x))

def scale_to_1(x, x_arr: list):
	# Broken
	x_max = max(x_arr)
	x_min = min(x_arr)
	return x / x_max - x_min

def scale(x, x_arr: list):
    x_max = max(x_arr)
    x_min = min(x_arr)
    if x_max == x_min:  # Cek untuk menghindari pembagian dengan nol
        return 0  # Atau bisa juga mengembalikan nilai tetap
    return x / (x_max - x_min)