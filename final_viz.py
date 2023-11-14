import cv2
import numpy as np
import os
from helpers import poses_to_vps

def plot_points_on_video(txt_path, video_path, output_path):
    # Load points from the .txt file
    points = np.loadtxt(txt_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video's original dimensions and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        # Plot the point on the frame if it exists
        if frame_count < len(points):
            point = poses_to_vps(np.stack([points[frame_count]]))[0].astype(np.int32)
            frame = cv2.circle(frame, (point[0], point[1]), 9, (0, 255, 0), -1)

        # Write the frame to the output video file
        out.write(frame)

        frame_count += 1

    # Release everything when the job is finished
    cap.release()
    out.release()

def main():
    os.makedirs('final_viz', exist_ok=True)
    for i in range(5, 10):
        if i > 4:
            plot_points_on_video(f'test/{i}.txt', f'unlabeled/{i}.hevc', f'final_viz/{i}.mp4')
        else:
            plot_points_on_video(f'test/{i}.txt', f'labeled/{i}.hevc', f'final_viz/{i}.mp4')

if __name__ == '__main__':
    main()