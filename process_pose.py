import numpy as np
import matplotlib.pyplot as plt
from helpers import vps_to_poses, calculate_error_score
import os

def reject_outliers(data, m = 1.0):
    d = np.abs(data - np.median(data, axis=0))
    mdev = np.median(d, axis=0)
    s = d/mdev
    reject_0 = s[:, 0] >= m
    reject_1 = s[:, 1] >= m
    reject = np.logical_or(reject_0, reject_1)
    data[reject, :] = np.nan
    return data

def average(vps, length=None):
    if length is None: length = len(vps) + 1
    avg = np.nanmean(vps, axis=0)
    return np.stack([avg] * length)

def rolling_average(vps, window=150):
    vps_rolling_avg = np.copy(vps)
    for i in range(0, len(vps)):
        earliest = max(0, i - window)
        latest = min(len(vps), i + window)
        i0 = latest - window*2 if latest == len(vps) else earliest
        i1 = earliest + window*2 if earliest == 0 else latest
        vps_rolling_avg[i] = np.nanmean(vps[i0:i1], axis=0)
    return np.concatenate([vps_rolling_avg, vps_rolling_avg[-1:]], axis=0)

def compute(vps): return vps_to_poses(rolling_average(reject_outliers(vps)))

def evaluate():
    total = 0
    for i in range(5):
        gt = np.loadtxt(f"labeled/{i}.txt")
        test = compute(np.loadtxt(f"vps/{i}.txt"))
        total += calculate_error_score(test, gt)
    print("AVERAGE ERROR SCORE IS", str((total / 5)) + "%")

def write_results():
    os.makedirs('test', exist_ok=True)
    for i in range(0, 10):
        test = compute(np.loadtxt(f"vps/{i}.txt"))
        np.savetxt(f"test/{i}.txt", test)

if __name__ == "__main__":
    evaluate()
    write_results()