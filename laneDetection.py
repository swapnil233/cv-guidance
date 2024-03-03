import cv2
import numpy as np


def canny(img):
    # used to see if there is an image. If not it will exit
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Convert from colored images to gray. Open cb uses bgr rather than rgb
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5

    # Blurring to reduce noise level (any image less than 5x5 will be erased)
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    # Done on gray image
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(canny):
    # Obtains height of canny array
    height = canny.shape[0]
    width = canny.shape[1]

    # Want to remove everything excep for the road
    mask = np.zeros_like(canny)
    triangle = np.array(
        [
            [
                (200, height),
                (800, 350),
                (1200, height),
            ]
        ],
        np.int32,
    )

    # Masking out all except for the triangle
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def houghLines(cropped_canny):
    return cv2.HoughLinesP(
        cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5
    )


def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:  # Check if the line is not None
                for x1, y1, x2, y2 in line:
                    # Ensure x1, y1, x2, y2 are integers
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image


def make_points(image, line):
    try:
        slope, intercept = line
        y1 = int(image.shape[0])
        y2 = int(y1 * 3.0 / 5)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]
    except Exception as e:
        print(f"Error in make_points: {e}")
        return None  # Return None if there's an error


# Goes through each line and tries to identify which line is important and which is not
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(image, left_fit_average)
    else:
        left_line = None
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
    else:
        right_line = None
    averaged_lines = [line for line in [left_line, right_line] if line is not None]
    return averaged_lines


def draw_center_line(img, left_line, right_line):
    """
    Draws the center line between the left and right lanes.
    """
    if left_line is not None and right_line is not None:
        # Calculate the midpoint at the bottom of the image (where the car is)
        bottom_midpoint = ((left_line[0][0] + right_line[0][0]) // 2, img.shape[0])

        # Calculate the midpoint further up the road
        top_midpoint = (
            (left_line[0][2] + right_line[0][2]) // 2,
            (left_line[0][3] + right_line[0][3]) // 2,
        )

        # Draw the center line
        cv2.line(img, bottom_midpoint, top_midpoint, (0, 255, 0), 10)

    return img


cap = cv2.VideoCapture("test1.mp4")
while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    # cv2.imshow("cropped_canny",cropped_canny)
    # Showing all the edges
    # cv2.imshow("canny_image", canny_image)
    # Gives an approximate ppresence of lines in an image
    lines = houghLines(cropped_canny)
    # Used to obtain location of the lines
    averaged_lines = average_slope_intercept(frame, lines)

    # Inside the main loop, after calculating averaged_lines
    line_image = display_lines(frame, averaged_lines)
    combo_image = addWeighted(frame, line_image)

    if averaged_lines:
        left_line = None
        right_line = None

        if len(averaged_lines) == 2:
            left_line, right_line = averaged_lines
        elif len(averaged_lines) == 1:
            # Decide based on the slope which line you have (left or right)
            line = averaged_lines[0]
            if np.polyfit((line[0][0], line[0][2]), (line[0][1], line[0][3]), 1)[0] < 0:
                left_line = line
            else:
                right_line = line

        # Now draw the center line if both lines are detected
        if left_line and right_line:
            combo_image = draw_center_line(combo_image, left_line, right_line)

    # Display the lines on the image
    cv2.imshow("result", combo_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
