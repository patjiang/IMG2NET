import sys

import numpy as np
import cv2 as cv
from EndGame.merge import HoughBundler

src = cv.imread('/Users/patrickjiang/PycharmProjects/researchLab/imgs/47603W.png')
cv.imshow("Start", src)

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

    # if both x1x2 or y1y2 is within 10% of
    # the midpoint in the image; add to second list

    xperc = xmax/100
    yperc = ymax/100

    xdiff = [abs(x1-(50*xperc)), abs(x2-(50*xperc))]
    ydiff = [abs(y1-(50*yperc)), abs(y2-(50*yperc))]
    print(6*xperc)

    xtrue = False
    ytrue = False
    thresh = 10

    if(xdiff[0] <= thresh * xperc) and (xdiff[1] <= thresh * xperc):
        xtrue = True

    if (ydiff[0] <= thresh * yperc) and (ydiff[1] <= thresh * yperc):
        ytrue = True

    if xtrue or ytrue:
        print(xdiff)
        print(ydiff)
        # added list for "center" lines
        cv.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)
        center_list.append([(x1, y1), (x2, y2)])
        print(center_list)

# how to only highlight "true" midline?
# wait; use the whole picture


cv.imshow('centered lines', src)

cv.waitKey(0)
cv.destroyAllWindows()
