# this code merely polarized a given image while still in raster form,
#the next steps are to vectorize and fourier transform each image

import numpy as np
import cv2 as cv



src = cv.imread('/Users/patrickjiang/PycharmProjects/researchLab/intersection.png')


def roi_mask(img, vert):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        chan_cnt = img.shape[2]
        mask_color = (255,) * chan_cnt
    else:
        mask_color = 255
    cv.fillPoly(mask, vert, mask_color)

    return cv.bitwise_and(img, mask)


src2 = cv.cvtColor(cv.GaussianBlur(src, (5, 5), 0, 0), cv.COLOR_BGR2GRAY)

src3 = cv.resize(src2, (1000, 1000), 0, 0, interpolation=cv.INTER_NEAREST)
cv.imshow('resized', src3)
v1 = [450, 0]
v2 = [0, 0]
v3 = [0, 450]
v4 = [450, 450]
vert1 = np.array([v1, v2, v3, v4])
roi_image1 = cv.fillPoly(src3, [vert1], 0)

# Fill color
v5 = [450, 575]
v6 = [0, 575]
v7 = [0, 1000]
v8 = [450, 1000]
vert2 = np.array([v5, v6, v7, v8])
roi_image2 = cv.fillPoly(src3, [vert2], 0)

v9 = [1000, 0]
v10 = [575, 0]
v11 = [575, 450]
v12 = [1000, 450]
vert3 = np.array([v9, v10, v11, v12])
roi_image3 = cv.fillPoly(src3, [vert3], 0)

v13 = [1000, 550]
v14 = [600, 550]
v15 = [600, 1000]
v16 = [1000, 1000]
vert4 = np.array([v13, v14, v15, v16])
roi_image = cv.fillPoly(src3, [vert4], 0)

src4 = cv.Canny(src3, 50,100)

#--- ensure image is of the type float ---
cv.imshow('scrubbed', src4)
img = src4.astype(np.float32)

# --- the following holds the square root of the sum of squares of the image dimensions ---
# --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
value = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))

polar_image = cv.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value, cv.WARP_FILL_OUTLIERS)

polar_image = polar_image.astype(np.uint8)
polar_image = cv.rotate(polar_image, cv.ROTATE_90_CLOCKWISE)
cv.imshow("Polar Image", polar_image)
