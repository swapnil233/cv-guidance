import cv2
import numpy as np
import datetime
from pid import PIDController
import datetime
import matplotlib.pyplot as plt

# import RPi.GPIO as GPIO  # using Rpi.GPIO module. This only works on Raspberry Pi
from time import sleep  # import function sleep for delay

# Define initial global constants for the region of interest (ROI)
# INSTRUCTIONS: run the script, adjust the trackbars to define the region of interest,
# and then use the values on the trackbars to update the global constants below:
HORIZON = 47
BOTTOM_TRIM = 99
LEFT_MARGIN = 17
RIGHT_MARGIN = 83
TOP_LEFT_MARGIN = 39
TOP_RIGHT_MARGIN = 51

# PID Controller Constants
KP = 0.1  # Proportional. This is used to correct for the current error.
KI = 0.01  # Integral. This is used to correct for steady-state error.
KD = 0.005  # Derivative. This is used to dampen the oscillations around the setpoint.
INTEGRAL_LIMIT = 100  # Maximum value for the integral term. This is to prevent integral windup, which can cause the controller to overshoot.
DERIVATE_FILTER_TAU = 0.01  # Time constant for the derivative filter. This is used to smooth out the derivative term.
SETPOINT_WEIGHTS = (
    1,
    0.1,
)  # Weights for the setpoint and derivative term. This is used to prioritize the setpoint over the derivative term.


# Input can either be a video on disk, or a live video stream from a USB camera
# Video files are stored in the test_videos directory.
# See the very bottom of the script for the list of available videos.
VIDEO_OR_CAMERA = "video"  # "video" or "camera"

# ROAD WIDTH
ROAD_WIDTH = 3.7  # meters

# Set to True to hide the region of interest overlay
HIDE_ROI = True

# Initialize buffers for storing line coefficients
left_line_buffer = []
right_line_buffer = []
buffer_length = 10  # Determines how many frames to average over

# # Motor control using GPIO
# GPIO.setmode(GPIO.BCM)  # GPIO numbering
# GPIO.setwarnings(False)  # enable warning from GPIO
# AN1 = 12  # set pwm1 pin on MD10-hat
# DIG1 = 26  # set dir1 pin on MD10-Hat
# GPIO.setup(AN1, GPIO.OUT)  # set pin as output
# GPIO.setup(DIG1, GPIO.OUT)  # set pin as output
# sleep(1)  # delay for 1 seconds
# p1 = GPIO.PWM(AN1, 100)  # set pwm for M1


# Initialize the PID controller
pid_controller = PIDController(
    kp=KP,
    ki=KI,
    kd=KD,
    integral_limit=INTEGRAL_LIMIT,
    derivative_filter_tau=DERIVATE_FILTER_TAU,
    setpoint_weights=(SETPOINT_WEIGHTS),
)


# def adjust_motor(control_action):
#     # Assuming control_action > 0 means steer right, < 0 means steer left
#     direction = GPIO.HIGH if control_action > 0 else GPIO.LOW

#     # Convert control_action to speed, e.g., larger deviations => higher speed
#     speed = min(abs(control_action) * 10, 100)  # Scale and limit speed to 100%

#     GPIO.output(DIG1, direction)  # Set direction
#     p1.start(speed)  # Set speed based on PID control action


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
    Applies Canny edge detection algorithm to the input image after converting it to HSL color space.

    Parameters:
    - img: Input image in BGR format.

    Returns:
    - Edge-detected image.
    """
    if img is None:
        raise ValueError("Input image is None.")

    # Convert the image to HSL color space
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Use the Lightness channel.
    lightness = hsl[:, :, 1]

    median_intensity = np.median(lightness)
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

    # Apply gaussian blur to reduce noise
    blur = cv2.GaussianBlur(lightness, (5, 5), 0)
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

    # cv2.imshow("roi", masked_edges)
    # cv2.waitKey(0)

    return masked_edges


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
    detected_lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180, 50, np.array([]), minLineLength=20, maxLineGap=150
    )

    # cv2.imshow("lines", masked_edges)
    # cv2.waitKey(0)

    return detected_lines


previous_time = datetime.datetime.now()


def draw_lines(img, lines, top_y, bottom_y, offset_file, control_actions_file):
    """
    Draws lines on the image.

    Parameters:
    - img: Original image.
    - lines: Lines to draw.

    Returns:
    - Image with lines drawn.
    """
    global left_line_buffer, right_line_buffer, previous_time

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    detected_left_line = False
    detected_right_line = False

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:  # Filter out horizontal lines
                    continue
                if slope <= 0:  # Left lane
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                    detected_left_line = True
                else:  # Right lane
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
                    detected_right_line = True

    left_x_pos = img.shape[1] * 0.25
    right_x_pos = img.shape[1] * 0.75

    if detected_left_line and left_line_x and left_line_y:
        left_poly = np.polyfit(left_line_y, left_line_x, deg=1)
        left_line_buffer.append(left_poly)
        left_x_pos = np.polyval(left_poly, bottom_y)
    elif left_line_buffer:
        left_poly = left_line_buffer[-1]
        left_x_pos = np.polyval(left_poly, bottom_y)

    if detected_right_line and right_line_x and right_line_y:
        right_poly = np.polyfit(right_line_y, right_line_x, deg=1)
        right_line_buffer.append(right_poly)
        right_x_pos = np.polyval(right_poly, bottom_y)
    elif right_line_buffer:
        right_poly = right_line_buffer[-1]
        right_x_pos = np.polyval(right_poly, bottom_y)

    lane_center = (left_x_pos + right_x_pos) / 2  # This is the center of the lane
    car_position = img.shape[1] / 2  # This is the center of the image
    deviation_pixels = (
        car_position - lane_center
    )  # Positive if car is to the right of the lane center, negative otherwise
    meters_per_pixel = ROAD_WIDTH / (right_x_pos - left_x_pos)
    deviation_meters = deviation_pixels * meters_per_pixel

    # Write the deviation to the offset file
    print(f"Deviation: {deviation_meters:.2f} m")
    offset_file.write(f"{deviation_meters}\n")

    deviation_direction = "right" if deviation_meters > 0 else "left"
    abs_deviation_meters = abs(deviation_meters)

    # Within the loop, after calculating deviation_meters
    current_time = datetime.datetime.now()
    dt = (current_time - previous_time).total_seconds()
    previous_time = current_time

    # Assuming deviation_meters is the current value you want to correct with PID
    control_action = pid_controller.update(deviation_meters, dt)
    # adjust_motor(control_action)

    # For now, let's just print the control action to see the output
    print(f"Control Action: {control_action}")
    control_actions_file.write(f"{control_action}\n")

    # Draw the control action arrow on the image at the bottom center
    base_position = (
        img.shape[1] // 2,
        img.shape[0] - 30,
    )
    draw_control_action_arrow(
        img,
        control_action,
        base_position,
        scale=10000,
        color=(255, 0, 0),  # Blue
        thickness=4,
        tip_length=0.2,
    )

    if left_line_buffer:
        left_poly_avg = np.mean(left_line_buffer, axis=0)
        draw_poly_line(
            img,
            left_poly_avg,
            top_y,
            bottom_y,
            color=(255, 0, 0),
            thickness=5,
            alpha=0.5,
        )

    if right_line_buffer:
        right_poly_avg = np.mean(right_line_buffer, axis=0)
        draw_poly_line(
            img,
            right_poly_avg,
            top_y,
            bottom_y,
            color=(0, 0, 255),
            thickness=5,
            alpha=0.5,
        )

    # Draw semi-transparent green overlay
    if detected_left_line and detected_right_line:
        overlay = img.copy()
        pts = np.array(
            [
                [np.polyval(left_poly_avg, bottom_y), bottom_y],
                [np.polyval(left_poly_avg, top_y), top_y],
                [np.polyval(right_poly_avg, top_y), top_y],
                [np.polyval(right_poly_avg, bottom_y), bottom_y],
            ],
            np.int32,
        )
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # Average the position of the left and right lanes if detected
    if detected_left_line and detected_right_line:
        # Calculate average lines using polynomial fit
        left_poly = np.polyfit(left_line_y, left_line_x, deg=1)
        right_poly = np.polyfit(right_line_y, right_line_x, deg=1)

        # Calculate the x positions of each lane at the top and bottom y coordinates
        left_bottom_x = np.polyval(left_poly, bottom_y)
        left_top_x = np.polyval(left_poly, top_y)
        right_bottom_x = np.polyval(right_poly, bottom_y)
        right_top_x = np.polyval(right_poly, top_y)

        # Calculate the midpoints of the top and bottom positions
        bottom_midpoint_x = (left_bottom_x + right_bottom_x) / 2
        top_midpoint_x = (left_top_x + right_top_x) / 2

        # Draw the dynamic center line
        cv2.line(
            img,
            (int(bottom_midpoint_x), bottom_y),
            (int(top_midpoint_x), top_y),
            (0, 255, 255),
            2,
        )
    else:
        # Optionally handle cases where one or neither lane is detected
        pass

    # Update text to indicate direction of deviation
    cv2.putText(
        img,
        f"{abs_deviation_meters:.2f} m {deviation_direction} of center",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return img


# Optional: Calculate and display the lane deviation from the center, etc., as previously described
def draw_poly_line(
    img, poly, top_y, bottom_y, color=(255, 0, 0), thickness=5, alpha=0.5
):
    """
    Draws a semi-transparent line on the image based on polynomial coefficients, extending from bottom_y to top_y.

    Parameters:
    - img: Original image.
    - poly: Polynomial coefficients for the line equation.
    - top_y: Top Y-coordinate for the line.
    - bottom_y: Bottom Y-coordinate for the line.
    - color: Line color.
    - thickness: Line thickness.
    - alpha: Transparency of the line.
    """
    # Create an overlay for drawing lines
    lines_overlay = img.copy()

    x_start = int(np.polyval(poly, bottom_y))
    x_end = int(np.polyval(poly, top_y))

    # Draw the line on the overlay
    cv2.line(lines_overlay, (x_start, bottom_y), (x_end, top_y), color, thickness)

    # Blend the overlay with the original image
    cv2.addWeighted(lines_overlay, alpha, img, 1 - alpha, 0, img)


# Read data from a file
def read_data_from_file(file_path):
    """Reads data from a file and returns a list of floats without any occurrences of 2.0.

    Parameters:
    - file_path: Path to the file.

    Returns:
    - List of floats without any occurrences of 2.0.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        data = [float(line.strip()) for line in lines if float(line.strip()) != 2.0]
    return data


def draw_control_action_arrow(
    img,
    control_action,
    base_position,
    scale=1,
    color=(0, 255, 0),
    thickness=2,
    tip_length=0.5,
):
    """
    Draws an arrow representing the PID control action on the image.

    Parameters:
    - img: The image onto which to draw.
    - control_action: The PID control action value.
    - base_position: A tuple (x, y) representing the base position of the arrow on the image.
    - scale: A scaling factor for the control action to adjust arrow length.
    - color: The color of the arrow (B, G, R).
    - thickness: The thickness of the arrow line.
    - tip_length: The length of the arrow tip in relation to the arrow length.
    """
    end_x = int(base_position[0] + control_action * scale)
    end_y = base_position[1]
    cv2.arrowedLine(
        img, base_position, (end_x, end_y), color, thickness, tipLength=tip_length
    )


# Video Processing
def process_video(video_path):
    """
    Processes the video for lane detection.

    Parameters:
    - video_path: Path to the video file.
    """
    video_capture = cv2.VideoCapture(video_path if VIDEO_OR_CAMERA == "video" else 0)
    if not video_capture.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    # Create a window with trackbars for adjusting the region of interest
    create_trackbar_window("result")

    # Create a file to store the offsets
    unique_filename = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_offsets.txt"
    )
    offset_file = open(unique_filename, "w")

    # Create a file to store the control actions
    control_actions_filename = unique_filename.replace(
        "_offsets.txt", "_control_actions.txt"
    )
    control_actions_file = open(control_actions_filename, "w")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Reached the end of the video.")
            print(
                {
                    "HORIZON": horizon,
                    "BOTTOM_TRIM": bottom,
                    "LEFT_MARGIN": left_margin,
                    "RIGHT_MARGIN": right_margin,
                    "TOP_LEFT_MARGIN": top_left_margin,
                    "TOP_RIGHT_MARGIN": top_right_margin,
                }
            )
            break  # Exit the loop if video end is reached

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
        combo_image = draw_lines(
            frame, lines, top_y, bottom_y, offset_file, control_actions_file
        )

        # Draw the ROI only if HIDE_ROI is False
        if not HIDE_ROI:
            combo_image = draw_roi(combo_image, vertices)

        cv2.imshow("result", combo_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            print("File saved as:", unique_filename)

            print(
                f"Final Horizon: {horizon}, Final Bottom: {bottom}, Left Margin: {left_margin}, Right Margin: {right_margin}, Top Left Margin: {top_left_margin}, Top Right Margin: {top_right_margin}"
            )

            break

    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    offset_file.close()
    control_actions_file.close()

    # Read the deviation and control actions data
    deviation_data = read_data_from_file(unique_filename)
    control_actions_data = read_data_from_file(control_actions_filename)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Deviation from Center (m)", color="tab:red")
    ax1.plot(deviation_data, color="tab:red", label="Deviation")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Control Action", color="tab:blue")
    ax2.plot(control_actions_data, color="tab:blue", label="Control Action")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Deviation and Control Action Over Time")
    plt.show()

    # Plotting the deviation from center
    plt.figure(figsize=(10, 5))
    plt.plot(deviation_data, label="Deviation from Center")
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Frames")
    plt.ylabel("Deviation (meters)")
    plt.title("Deviation from Lane Center Over Time")
    plt.legend()
    plt.show()

    # Plotting the control actions
    plt.figure(figsize=(10, 5))
    plt.plot(control_actions_data, label="Control Action")
    plt.xlabel("Frames")
    plt.ylabel("Control Action Value")
    plt.title("PID Control Actions Over Time")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    video_paths = [
        "test_videos/test1.mp4",  # 0
        "test_videos/test2.mp4",  # 1
        "test_videos/test3.mp4",  # 2
        "test_videos/test4.mp4",  # 3
        "test_videos/test5.mp4",  # 4
        "test_videos/dash1.mp4",  # 5
        "test_videos/dash2.mp4",  # 6
        "test_videos/dash3.mp4",  # 7
        "test_videos/test6.mp4",  # 8
    ]
    try:
        process_video(video_paths[1])
    except Exception as e:
        print(f"Error processing video: {e}")
