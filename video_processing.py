import cv2
import numpy as np
from image_processing import (
    canny_edge_detector,
    region_of_interest,
    detect_lines,
    draw_lines,
    draw_roi,
)
from constants import (
    HORIZON,
    BOTTOM_TRIM,
    LEFT_MARGIN,
    RIGHT_MARGIN,
    TOP_LEFT_MARGIN,
    TOP_RIGHT_MARGIN,
)
from ui_utils import create_trackbar_window, get_trackbar_values


def process_video(video_path):
    """
    Processes the video for lane detection.

    Parameters:
    - video_path: Path to the video file.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    create_trackbar_window("result")

    while True:  # Loop indefinitely
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                print("Reached the end of the video, restarting.")
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            (
                horizon,
                bottom,
                left_margin,
                right_margin,
                top_left_margin,
                top_right_margin,
            ) = get_trackbar_values("result")
            height, width = frame.shape[:2]

            vertices = np.array(
                [
                    [
                        (
                            width * left_margin / 100.0,
                            height * bottom / 100.0,
                        ),  # Bottom left
                        (
                            width * top_left_margin / 100.0,
                            height * horizon / 100.0,
                        ),  # Top left
                        (
                            width * top_right_margin / 100.0,
                            height * horizon / 100.0,
                        ),  # Top right
                        (
                            width * right_margin / 100.0,
                            height * bottom / 100.0,
                        ),  # Bottom right
                    ]
                ],
                dtype=np.int32,
            )

            canny_image = canny_edge_detector(frame)
            cropped_canny = region_of_interest(canny_image, vertices)
            lines = detect_lines(cropped_canny)
            combo_image = draw_lines(frame, lines)
            combo_image = draw_roi(combo_image, vertices)

            cv2.imshow("result", combo_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(
                    f"Final Horizon: {horizon}, Final Bottom: {bottom}, Left Margin: {left_margin}, Right Margin: {right_margin}, Top Left Margin: {top_left_margin}, Top Right Margin: {top_right_margin}"
                )
                print(
                    "Replace the HORIZON, BOTTOM_TRIM, LEFT_MARGIN, and RIGHT_MARGIN constants with these values."
                )
                video_capture.release()
                cv2.destroyAllWindows()
                return  # Exit the function after printing the final values


if __name__ == "__main__":
    video_paths = [
        "test_videos/test1.mp4",
        "test_videos/test2.mp4",
        "test_videos/test3.mp4",
        "test_videos/dash1.mp4",
    ]
    try:
        process_video(video_paths[0])
    except Exception as e:
        print(f"Error processing video: {e}")
