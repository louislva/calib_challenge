import cv2
import imageio
import numpy as np
import math
import tqdm
import os
import multiprocessing
from scipy.spatial import KDTree

DEBUG = False

WIDTH, HEIGHT = 1164, 874
CENTER_SIZE = 0.4
CENTER_LEFT = 0.5 - CENTER_SIZE / 2
CENTER_RIGHT = 0.5 + CENTER_SIZE / 2
MIN_ANGLE = math.tau / 12
FOCAL_LENGTH = 910
OUTLIER_RADIUS = 10

def calc_line(p0, p1):
    dy = p1[1] - p0[1]
    dx = p1[0] - p0[0]
    if dx == 0:
        print("WARNING: two points with the same x value; returning slope of None")
        return None, None
    m = dy / dx
    b = p0[1] - p0[0] * m
    return m, b

def calc_vanishing_point(lines):
    if not lines or len(lines) < 2:
        return None
    
    # Calculate intersection points for each pair of lines
    intersection_points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            m1, b1 = lines[i]
            m2, b2 = lines[j]
            if m1 == m2: continue
            angle = math.atan2(abs(m1 - m2), 1 + m1 * m2)
            if angle < MIN_ANGLE: continue
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            intersection_points.append((x, y))
    DEBUG and print("calc_vanishing_point", len(intersection_points))

    # Remove anything outside the center, to optimize
    intersection_points = [p for p in intersection_points if p[0] > CENTER_LEFT * WIDTH and p[0] < CENTER_RIGHT * WIDTH and p[1] > CENTER_LEFT * HEIGHT and p[1] < CENTER_RIGHT * HEIGHT]
    DEBUG and print("optimized to", len(intersection_points))

    # Calculate the "least outlier" point / most popular area
    # best_point = None
    # best_score = 0
    # for point in intersection_points:
    #     # Calculate the score for each point by summing the number of points within a radius of 50 pixels
    #     score = sum(((point[0]-p[0])**2 + (point[1]-p[1])**2) < OUTLIER_RADIUS_SQUARED for p in intersection_points)
    #     if score > best_score:
    #         best_score = score
    #         best_point = point

    if len(intersection_points) < 1: return None

    # Build the KDTree
    tree = KDTree(intersection_points)

    best_point = None
    best_score = 0
    for point in intersection_points:
        # Query the KDTree for points within a radius of 50 pixels
        indices = tree.query_ball_point(point, OUTLIER_RADIUS)
        score = len(indices)
        if score > best_score:
            best_score = score
            best_point = point
    
    DEBUG and print("best_point", best_point)
    return best_point

def process(video_path, vp_out_path, video_out_path):
    cap = cv2.VideoCapture(video_path)
    out = imageio.get_writer(video_out_path, fps=20)

    prev_frame = None
    prev_points = None
    i = 0
    vps = []
    t = tqdm.tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False: break
        
        if prev_points is not None:
            tracked_points, statuses, err = cv2.calcOpticalFlowPyrLK(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                prev_points,
                None,
            )
            tracked_points = tracked_points.reshape((-1,2))
            statuses = statuses.reshape((-1))

            out_frame = frame.copy()
            lines = []
            for status, point, prev_point in zip(statuses, tracked_points, prev_points):
                if (status == 1):
                    # out_frame = cv2.circle(out_frame, (int(prev_point[0]), int(prev_point[1])), 3, (255, 0, 0), -1)
                    # out_frame = cv2.circle(out_frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
                    m, b = calc_line(prev_point, point)
                    if m is not None: 
                        # out_frame = cv2.line(out_frame, (0, int(b)), (WIDTH, int(WIDTH * m + b)), (0, 255, 0), 2)
                        lines.append([m, b])
            
            vp = calc_vanishing_point(lines)

            if vp is not None:
                out_frame = cv2.circle(out_frame, (int(vp[0]), int(vp[1])), 9, (255, 255, 255), -1)
                vps += [vp]
            elif len(vps) > 0:
                # In the cases where we don't get a vanishing point, we just reuse the last one, or make something up. It will be smoothed out in the end anyway
                vps += [vps[-1]]
            else:
                vps += [(WIDTH / 2, HEIGHT / 2)]

            out.append_data(out_frame)


        prev_points = cv2.goodFeaturesToTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), maxCorners=100, qualityLevel=0.01, minDistance=30).reshape((-1,2))
        prev_frame = frame
        
        i+=1
        t.update(1)
    cap.release()
    out.close()

    np.savetxt(vp_out_path, vps)
    return vps

def main():
    os.makedirs('vps', exist_ok=True)
    os.makedirs('viz', exist_ok=True)
    with multiprocessing.Pool(1) as pool:
        # tasks = [(f'labeled/{i}.hevc', f'vps/{i}.txt', f'viz/{i}.mp4') for i in range(5)]
        tasks = [(f'unlabeled/{i}.hevc', f'vps/{i}.txt', f'viz/{i}.mp4') for i in range(9, 10)]
        pool.starmap(process, tasks)

if __name__ == '__main__':
    main()
