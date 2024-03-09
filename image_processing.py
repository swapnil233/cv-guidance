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
