import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.fft import fftfreq
from scipy.fft import fft

src = cv.imread('/Users/patrickjiang/PycharmProjects/researchLab/imgs/47603.jpg')
cv.imshow("Start", src)



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


src2 = cv.cvtColor(cv.GaussianBlur(src, (5, 5), 0, 0), cv.COLOR_BGR2GRAY)

src3 = cv.resize(src2, (1000, 1000), 0, 0, interpolation=cv.INTER_NEAREST)
cv.imshow("Gaussian Blurred", src3)

v1 = [375, 0]
v2 = [0, 0]
v3 = [0, 375]
v4 = [375, 375]
vert1 = np.array([v1, v2, v3, v4])
roi_image1 = cv.fillPoly(src3, [vert1], 0)

# Fill color
v5 = [375, 625]
v6 = [0, 625]
v7 = [0, 1000]
v8 = [375, 1000]
vert2 = np.array([v5, v6, v7, v8])
roi_image2 = cv.fillPoly(src3, [vert2], 0)

v9 = [1000, 0]
v10 = [625, 0]
v11 = [625, 375]
v12 = [1000, 375]
vert3 = np.array([v9, v10, v11, v12])
roi_image3 = cv.fillPoly(src3, [vert3], 0)

v13 = [1000, 625]
v14 = [625, 625]
v15 = [625, 1000]
v16 = [1000, 1000]
vert4 = np.array([v13, v14, v15, v16])
roi_image = cv.fillPoly(src3, [vert4], 0)

src4 = cv.Canny(src3, 50, 100)
cv.imshow("Roi mask", src4)

src5 = cv.resize(src4, (100, 100), 0, 0, interpolation=cv.INTER_NEAREST)
cv.imshow("",src5)

src7 = matrix_trans(src5)

x = [x * 2 for x in src7[0, :]]
y = [y * 2 for y in src7[1, :]]

plt.rcParams["figure.figsize"] = [15, 7]
plt.rcParams["figure.autolayout"] = True
theta = x[:]
r = [(2 * z / max(y)) - 1 for z in y]
plt.xlim(0, 360)
plt.ylim(-1, 1)
plt.grid()
plt.plot(x, r, marker=".", markersize=1, markeredgecolor="red", markerfacecolor="green")
plt.show()

t = np.linspace(0, 360, len(theta))

f = fftfreq(len(t), np.diff(t)[0])
FFT = fft(r)

plt.plot(f[:len(theta) // 2], np.abs(FFT[:len(theta) // 2]))
plt.xlabel('$f_n$ [$s^{-1}$]', fontsize=20)
plt.ylabel('|$\hat{x}_n$|', fontsize=20)
plt.show()
plt.savefig('4_cross2.png')

cv.waitKey(0)
cv.destroyAllWindows()
