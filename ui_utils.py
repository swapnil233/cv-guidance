import cv2
from constants import (
    HORIZON,
    BOTTOM_TRIM,
    LEFT_MARGIN,
    RIGHT_MARGIN,
    TOP_LEFT_MARGIN,
    TOP_RIGHT_MARGIN,
)


def create_trackbar_window(window_name):
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Horizon", window_name, HORIZON, 100, lambda x: None)
    cv2.createTrackbar("Bottom", window_name, BOTTOM_TRIM, 100, lambda x: None)
    cv2.createTrackbar("Left Margin", window_name, LEFT_MARGIN, 100, lambda x: None)
    cv2.createTrackbar("Right Margin", window_name, RIGHT_MARGIN, 100, lambda x: None)
    cv2.createTrackbar(
        "Top Left Margin", window_name, TOP_LEFT_MARGIN, 100, lambda x: None
    )
    cv2.createTrackbar(
        "Top Right Margin", window_name, TOP_RIGHT_MARGIN, 100, lambda x: None
    )


def get_trackbar_values(window_name):
    horizon = cv2.getTrackbarPos("Horizon", window_name)
    bottom = cv2.getTrackbarPos("Bottom", window_name)
    left_margin = cv2.getTrackbarPos("Left Margin", window_name)
    right_margin = cv2.getTrackbarPos("Right Margin", window_name)
    top_left_margin = cv2.getTrackbarPos("Top Left Margin", window_name)
    top_right_margin = cv2.getTrackbarPos("Top Right Margin", window_name)
    return horizon, bottom, left_margin, right_margin, top_left_margin, top_right_margin
