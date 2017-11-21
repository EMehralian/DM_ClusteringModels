import math
import numpy as np


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(center, a, b):
    v1 = np.array(a)-np.array(center)
    v2 = np.array(b)-np.array(center)
    if length(v1) and length(v2):
        return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def toDegrees(radians):
    if radians:
        return 360 * radians / (2 * math.pi)


