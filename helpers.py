import math
import numpy as np 
from find_vps import WIDTH, HEIGHT, FOCAL_LENGTH

def calculate_error_score(test: np.array, gt: np.array):
    def get_mse(gt, test):
        test = np.nan_to_num(test)
        return np.mean(np.nanmean((gt - test)**2, axis=0))
    zero_mse = get_mse(gt, np.zeros_like(gt))
    mse = get_mse(gt, test)

    percent_err_vs_all_zeros = 100*np.mean(mse)/np.mean(zero_mse)
    print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')
    return percent_err_vs_all_zeros

def vps_to_poses(arr):
    xs = arr[:, 0]
    ys = arr[:, 1]
    pitches = -np.arctan2(ys - HEIGHT / 2, FOCAL_LENGTH)
    yaws = np.arctan2(xs - WIDTH / 2, FOCAL_LENGTH)

    return np.stack([pitches, yaws], axis=1)

def poses_to_vps(arr):
    pitches, yaws = arr[:, 0], arr[:, 1]
    xs = np.tan(yaws) * FOCAL_LENGTH + WIDTH / 2
    ys = -np.tan(pitches) * FOCAL_LENGTH + HEIGHT / 2

    return np.stack([xs, ys], axis=1)