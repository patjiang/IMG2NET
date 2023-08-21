import sys
import numpy as np
import cv2 as cv
from EndGame.merge import HoughBundler

src = cv.imread('/Users/patrickjiang/PycharmProjects/researchLab/imgs/47603S.png')
cv.imshow("Start", src)

"""
hsva(36.522, 19%, 52%, 1)
hsva(33.333, 40%, 65%, 1)
hsva(96, 7%, 72%, 1)
hsva(38.4, 23%, 57%, 1)
hsva(45, 16%, 60%, 1)
hsva(36, 17%, 53%, 1)
hsva(60, 6%, 62%, 1)
hsva(66.667, 9%, 62%, 1)
hsva(31.765, 17%, 40%, 1)
opencv has h in 0-179, s in 0-255, and v in 0-255
"""
src2 = cv.cvtColor(cv.GaussianBlur(src, (5, 5), 0, 0), cv.COLOR_BGR2GRAY)

src4 = cv.Canny(src2, 50, 100, apertureSize = 3)

dims = src4.shape

src5 = np.zeros(src4.shape, dtype = np.uint8)

# Apply HoughLinesP method to
# to directly obtain line end points
lines_list = []
slope_list = []
center_list = []
lines = cv.HoughLinesP(
    src4,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=60,  # Min number of votes for valid line
    minLineLength=5,  # Min allowed length of line
    maxLineGap=20  # Max allowed gap between line for joining them
)

bundler = HoughBundler(min_distance=12,min_angle=3)
lines = bundler.process_lines(lines)

# Iterate over points
#lenline = sqrt((a.x^2) + (a.y^2))
#dims(0) is ymax and dims(1) is xmax
#we want to extend both sides of line until y == 0 or y == dims(0)
#y = mx + b; we have two points
# solve for slopes
# then recalculate the lines until y = 0 and y = dims(0)
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    # min and max yvals
    ymin = 1
    ymax = dims[0] - 1
    xmin = 1
    xmax = dims[1] - 1

    # calculate the slope
    if x2 == x1:
        xmax = x2
        xmin = x1
        m = sys.maxsize *2 + 1

    else:
        m = (y2 - y1) / (x2 - x1)
    # calculate the intercept
    b = y2 - m * x2
    # now we have y = mx + b
    # calculate new x value at y = 0 and y = dim(0)
    if m != 0:
        xmax = round((ymax - b) / m)
        xmin = round(-b / m)

    else:
        xmax = dims[1] - 1
        xmin = 0
        ymin = y1
        ymax = y2

    cv.line(src5, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Maintain a simples lookup list for points and slopes
    lines_list.append([(xmin, ymin), (xmax, ymax)])
    slope_list.append(m)



numlines = len(lines_list)

cv.imshow('d', src5)
cv.imshow('detectedLines.png', src)

#############################################

def roi_mask(img, vert):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        chan_cnt = img.shape[2]
        mask_color = (255,) * chan_cnt
    else:
        mask_color = 255
    cv.fillPoly(mask, vert, mask_color)

    return cv.bitwise_and(img, mask)


def matrix_trans(narray):
    # --- ensure image is of the type float ---
    img = narray.astype(np.float32)
    # --- the following holds the square root of the sum of squares of the image dimensions ---
    # --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
    value = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    polar_image = cv.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value, cv.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    src5 = cv.resize(polar_image, (180, 180), 0, 0, interpolation=cv.INTER_NEAREST)
    src6 = cv.bitwise_not(src5)
    src7 = np.array(np.where(src6 != 255))
    return src7


cv.waitKey(0)
cv.destroyAllWindows()



cv.waitKey(0)
cv.destroyAllWindows()
