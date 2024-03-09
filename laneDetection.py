import cv2
import numpy as np


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


def region_of_interest(edges, lines_info=None):
    """
    Applies a mask to the input edge-detected image to focus on the region of interest.

    Parameters:
    - edges: Edge-detected image. Comes after Canny edge detector.

    Returns:
    - Masked edge-detected image focusing on the region of interest.
    """
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # These numbers represent the fraction of the image where you'll typically find the horizon and the bottom of the image.
    horizon_line = height * 0.65
    bottom_trim = (
        height * 0.99
    )  # This can be adjusted to ignore the very bottom of the image that might not contain useful information

    # Define the vertices of the trapezoid
    vertices = np.array(
        [
            [
                (width * 0.1, bottom_trim),  # Bottom left
                (width * 0.4, horizon_line),  # Top left
                (width * 0.6, horizon_line),  # Top right
                (width * 0.9, bottom_trim),  # Bottom right
            ]
        ],
        dtype=np.int32,
    )

    # Fill the defined polygon with white
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges, vertices


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
        masked_edges, 2, np.pi / 180, 100, np.array([]), minLineLength=70, maxLineGap=20
    )


def draw_lines(img, lines):
    """
    Draws lines on the image.

    Parameters:
    - img: Original image.
    - lines: Lines to draw.

    Returns:
    - Image with lines drawn.
    """
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return cv2.addWeighted(img, 0.8, line_image, 1, 1)


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


def create_trackbar_window(window_name):
    cv2.namedWindow(window_name)
    # Create trackbars for adjusting ROI. The values are percentages of the image height.
    cv2.createTrackbar("Horizon", window_name, 65, 100, lambda x: None)
    cv2.createTrackbar("Bottom", window_name, 95, 100, lambda x: None)


def get_trackbar_values(window_name):
    horizon = cv2.getTrackbarPos("Horizon", window_name) / 100.0
    bottom = cv2.getTrackbarPos("Bottom", window_name) / 100.0
    return horizon, bottom


def region_of_interest(edges, vertices):
    """
    Modifies the region of interest based on trackbar positions.
    """
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges


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

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Reached the end of the video or error reading the video frame.")
            break

        horizon, bottom = get_trackbar_values("result")
        height, width = frame.shape[:2]

        # Calculate dynamic ROI based on trackbar values
        vertices = np.array(
            [
                [
                    (width * 0.1, height * bottom),  # Bottom left
                    (width * 0.4, height * horizon),  # Top left
                    (width * 0.6, height * horizon),  # Top right
                    (width * 0.9, height * bottom),  # Bottom right
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
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_paths = ["test1.mp4", "test2.mp4", "test3.mp4", "dash1.mp4"]
    try:
        process_video(video_paths[3])
    except Exception as e:
        print(f"Error processing video: {e}")
