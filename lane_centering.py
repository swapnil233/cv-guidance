import cv2
import numpy as np

# Define initial global constants for the ROI
# INSTRUCTIONS: run the script, adjust the trackbars to define the region of interest, and then use the printed values to replace the constants

HORIZON = 43
BOTTOM_TRIM = 100
LEFT_MARGIN = 16
RIGHT_MARGIN = 83
TOP_LEFT_MARGIN = 40
TOP_RIGHT_MARGIN = 50


# UI Utilities
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


# Image Processing
def canny_edge_detector(img):
    """
    Applies Canny edge detection algorithm to the input image.

    Parameters:
    - img: Input image in BGR format.

    Returns:
    - Edge-detected image.
    """
    if img is None:
        raise ValueError("Input image is None.")

    # Convert to grayscale to get to single-channel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_intensity = np.median(gray)
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

    # Apply gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, lower_threshold, upper_threshold)
    return edges


def region_of_interest(edges, vertices):
    """
    Applies a mask to the input edge-detected image to focus on the region of interest based on the vertices calculated from global variables or trackbar values.

    Parameters:
    - edges: Edge-detected image.
    - vertices: Vertices of the polygon to mask the image.

    Returns:
    - Masked edge-detected image focusing on the region of interest.
    """
    mask = np.zeros_like(edges)

    # Assuming that vertices are in the correct format, fill the polygon
    cv2.fillPoly(mask, [vertices], 255)

    # Apply the mask to the edge-detected image
    masked_edges = cv2.bitwise_and(edges, mask)

    return masked_edges


def detect_lines(masked_edges):
    """
    Detects straight lines in the masked, edge-detected image using the Hough Transform algorithm.

    This function converts the edge-detected image into a series of lines, defined by their endpoints.
    Parameters such as the resolution of the accumulator, minimum line length, and maximum gap between
    line segments can be adjusted to optimize detection.

    Parameters:
    - masked_edges: Masked edge-detected image.

    Returns:
    - Lines detected in the image.
    """
    return cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=100
    )


def draw_lines(img, lines, top_y, bottom_y):
    """
    Draws lines on the image.

    Parameters:
    - img: Original image.
    - lines: Lines to draw.

    Returns:
    - Image with lines drawn.
    """
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)  # Calculate slope
                if abs(slope) < 0.5:  # Ignore lines that are almost horizontal
                    continue
                if slope <= 0:  # Left lane
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:  # Right lane
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

    # When calling draw_poly_line, pass the new top_y and bottom_y values
    if left_line_x and left_line_y:
        left_poly = np.polyfit(left_line_y, left_line_x, deg=1)
        draw_poly_line(img, left_poly, top_y, bottom_y, color=(255, 0, 0), thickness=5)

    if right_line_x and right_line_y:
        right_poly = np.polyfit(right_line_y, right_line_x, deg=1)
        draw_poly_line(img, right_poly, top_y, bottom_y, color=(255, 0, 0), thickness=5)

    return img


def draw_poly_line(img, poly, top_y, bottom_y, color=(255, 0, 0), thickness=5):
    """
    Draws a line on the image based on polynomial coefficients, extending from bottom_y to top_y.
    """
    x1 = int(np.polyval(poly, bottom_y))
    x2 = int(np.polyval(poly, top_y))

    cv2.line(img, (x1, bottom_y), (x2, top_y), color, thickness)


def draw_roi(img, vertices, color=(0, 255, 0), thickness=5):
    """
    Draws the region of interest on the image.

    Parameters:
    - img: The original image.
    - vertices: The vertices of the polygon representing the region of interest.
    - color: The color of the polygon's edges. Default is green.
    - thickness: The thickness of the polygon's edges.
    """
    for i in range(vertices.shape[1] - 1):
        cv2.line(
            img, tuple(vertices[0][i]), tuple(vertices[0][i + 1]), color, thickness
        )
    # Draw a line from the last vertex to the first vertex
    cv2.line(img, tuple(vertices[0][-1]), tuple(vertices[0][0]), color, thickness)
    return img


# Video Processing
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
            top_y = int(height * horizon / 100.0)  # Convert from percentage to pixels
            bottom_y = int(height * bottom / 100.0)

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
            combo_image = draw_lines(frame, lines, top_y, bottom_y)
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
        "test_videos/test4.mp4",
        "test_videos/test5.mp4",
        "test_videos/dash1.mp4",
        "test_videos/dash2.mp4",
        "test_videos/dash3.mp4",
    ]
    try:
        process_video(video_paths[5])
    except Exception as e:
        print(f"Error processing video: {e}")
