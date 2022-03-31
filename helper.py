import PIL
import imageio.core
import numpy
import numpy as np
import colour
import math
import csv
from PIL import Image
from multiprocessing import Pool
import numpy as np

is_install_required_package = True
is_debug = False
T_START = 2000  # 1500 # K
T_FINISH = 12500  # 25000 # K


def mean_CCT_for_image(image):
    # display(image)

    with Pool(processes=5) as p:
        pixels = list(image.getdata())
        print(pixels[0])
        results = p.map(RGB_to_CCT, pixels)

    return get_metriks(results)


def get_metriks(arr, Ts=T_START, Tf=T_FINISH):
    d = {}
    x = np.array(arr)
    # print("arr:", type(arr))  # <class 'list'>
    # print("x:", type(x))      # <class 'numpy.ndarray'>

    # print("\na. > Ts")
    xx = x[x > Ts]
    d["mean"] = np.mean(xx)
    d["count"] = len(xx)

    # Sifirlar
    # print("\nb. == 0")
    zz = x[x == 0]
    d["e0"] = len(zz)

    # Birler
    # print("\ne. == 1")
    bb = x[x == 1]
    d["e1"] = len(bb)

    # Negatifler
    # print("\ne. < 0")
    nn = x[x < 0]
    d["negative"] = len(nn)

    return d


# Referans: colour-develop\colour\plotting\temperature.py#90-100
def uv_to_xy(uv):
    """
    Converts given *uv* chromaticity coordinates to xy/*ij* chromaticity
    coordinates.
    """

    return colour.models.UCS_uv_to_xy(uv)


# Referans: colour-develop\colour\plotting\temperature.py#123-125
def temperature_to_isotemperature_line(T):
    """
      T = 2500 # Kelvin

      return (x0, y0, x1, y1)
    """
    D_uv = 0.025

    x0, y0 = uv_to_xy(colour.temperature.CCT_to_uv(np.array([T, -D_uv]), 'Robertson 1968'))
    x1, y1 = uv_to_xy(colour.temperature.CCT_to_uv(np.array([T, D_uv]), 'Robertson 1968'))

    return ([x0, y0], [x1, y1])


def is_between_Ts_Tf(xy, Ts=T_START, Tf=T_FINISH):
    """
      print("True ?", is_between_Ts_Tf([0.375, 0.375]))
      print("True ?", is_between_Ts_Tf([0.335, 0.25]))
      print("True ?", is_between_Ts_Tf([0.4, 0.65]))
      print("False ?", is_between_Ts_Tf([0.1, 0.2]))
      print("False ?", is_between_Ts_Tf([0.6, 0.3]))
    """
    p = xy
    b1 = is_below_line_at_T(Ts, p)
    b2 = is_below_line_at_T(Tf, p)
    b = b1 and not (b2)

    # print(b1, b2, b)

    return b


def is_below_line_at_T(T, p):
    p1, p2 = temperature_to_isotemperature_line(T)

    x1, y1 = p1
    x2, y2 = p2
    xA, yA = p

    v1 = (x2 - x1, y2 - y1)  # Vector 1
    v2 = (x2 - xA, y2 - yA)  # Vector 1
    xp = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product

    return xp <= 0


def xy_to_CCT_with_andres99(xy, is_K_greater_50000=False):
    x, y = xy[0], xy[1]

    # 50.000 - 8x10^5 K
    if is_K_greater_50000:
        xe = 0.3356
        ye = 0.1691

        A0 = 36284.48953
        A1 = 0.00228
        A2 = 5.4535e-36
        A3 = 0
        t1 = 0.07861
        t2 = 0.01543
        t3 = 1
    # 3.000 - 50.000 K
    else:
        xe = 0.3366
        ye = 0.1735

        A0 = -949.86315
        A1 = 6253.80338
        A2 = 28.70599
        A3 = 0.00004
        t1 = 0.92159
        t2 = 0.20039
        t3 = 0.07125

    e = math.exp

    n = (x - xe) / (y - ye)
    CCT = A0 + A1 * e(-n / t1) + A2 * e(-n / t2) + A3 * e(-n / t3)

    return CCT


"""
  RGB = np.array([255.0, 255.0, 255.0])
"""


def RGB_to_CCT(RGB, method="McCamy 1992", Ts=T_START, Tf=T_FINISH):
    # print(type(RGB[0]))
    if type(RGB) is tuple:
        RGB = np.array(RGB)

    # Conversion to tristimulus values.
    XYZ = colour.sRGB_to_XYZ(RGB / 255)

    if XYZ[0] == 0 and XYZ[1] == 0 and XYZ[2] == 0:
        return 6502  # K

    # Conversion to chromaticity coordinates.
    xy = colour.XYZ_to_xy(XYZ)

    if method == "McCamy 1992":
        if is_debug:
            print("\txy:", xy)

    # a. xy erimde degilse
    if not (is_between_Ts_Tf(xy, Ts, Tf)):
        return 1

    if method == "andres99_1":
        CCT = xy_to_CCT_with_andres99(xy)
    elif method == "andres99_2":
        CCT = xy_to_CCT_with_andres99(xy, is_K_greater_50000=True)
    elif method == "Robertson 1968":
        uv = colour.UCS_to_uv(colour.XYZ_to_UCS(colour.xy_to_XYZ(xy)))
        CCT, d_uv = colour.uv_to_CCT(uv, method='Robertson 1968')
    else:
        # Conversion to correlated colour temperature in K.
        # https://github.com/colour-science/colour#correlated-colour-temperature-computation-methods-colour-temperature
        CCT = colour.xy_to_CCT(xy, method)

    if is_debug:
        print("\t T: %.1f Kelvin (method: %s)" % (CCT, method))

    if CCT < 0:
        CCT = 0
    # elif CCT < Ts or CCT > Tf:
    #  CCT = -CCT
    elif CCT < Ts:
        CCT = Ts
    elif CCT > Tf:
        CCT = Tf

    return CCT

#Piksel methodu
def get_RGB_For_Imagge(image):
    pixels = list(image.getdata())
    return pixels
#Fotoğraf Açılıyor
im = Image.open(r"C:\Users\erenk\Desktop/22222.jpg")
im.show()
#Piksel fazlalığı giderilip iş gücü azaltılıyor..
size = 50, 50
im.thumbnail(size, Image.ANTIALIAS)
#RGB değeri alınıyor
RGB = get_RGB_For_Imagge(im)
#RGB(tuple) değeri np.array dönüştürülüyor.
rgb = np.array(RGB)
#Döngü içerisinde her bir piksel için cct değeri hespalanıyor
i = 1
while i < len(rgb):
    cct = RGB_to_CCT(rgb[i])
    i += 1
    print(cct)
