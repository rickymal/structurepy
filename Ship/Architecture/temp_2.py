import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt, pi, degrees

def area(pts):
    'Area of cross-section.'

    if pts[0] != pts[-1]:
        pts = pts + pts[:1]
        x = [ c[0] for c in pts ]
        y = [ c[1] for c in pts ]
        s = 0
        for i in range(len(pts) - 1):
            s += x[i]*y[i+1] - x[i+1]*y[i]
        return s/2


def centroid(pts):
    'Location of centroid.'
    
    if pts[0] != pts[-1]:
    pts = pts + pts[:1]
    x = [ c[0] for c in pts ]
    y = [ c[1] for c in pts ]
    sx = sy = 0
    a = area(pts)
    for i in range(len(pts) - 1):
        sx += (x[i] + x[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
        sy += (y[i] + y[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
    return sx/(6*a), sy/(6*a)


def inertia(pts):
    'Moments and product of inertia about centroid.'
    
    if pts[0] != pts[-1]:
    pts = pts + pts[:1]
    x = [ c[0] for c in pts ]
    y = [ c[1] for c in pts ]
    sxx = syy = sxy = 0
    a = area(pts)
    cx, cy = centroid(pts)
    for i in range(len(pts) - 1):
        sxx += (y[i]**2 + y[i]*y[i+1] + y[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
        syy += (x[i]**2 + x[i]*x[i+1] + x[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
        sxy += (x[i]*y[i+1] + 2*x[i]*y[i] + 2*x[i+1]*y[i+1] + x[i+1]*y[i])*(x[i]*y[i+1] - x[i+1]*y[i])
    return sxx/12 - a*cy**2, syy/12 - a*cx**2, sxy/24 - a*cx*cy