from __future__ import division
import pypylon.pylon as py
import matplotlib.pyplot as plt
import numpy as np
import cv2

# This is an example script to show a machine vision application using color detection.
# The camera takes images which are converted into HSV color space. 
# By adjusting the HSV color filter objects of certain color can be extracted from the image.
# This is a standard method in machine vision applications.
# This version contains an additional example how the extracted objects can be counted.

cam = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
cam.Open()
cam.PixelFormat.Value = 'BGR8'

def nothing(*arg):
    pass


# Initial HSV GUI slider values to load on program start.
icol = (36, 202, 59, 71, 255, 255)  # Green
# icol = (18, 0, 196, 36, 255, 255)  # Yellow
# icol = (89, 0, 0, 125, 255, 255)  # Blue
# icol = (0, 100, 80, 10, 255, 255)   # Red
cv2.namedWindow('colorTest')
# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)

cam.StartGrabbing()
cam.ExposureTime.Value = 20000
cam.BalanceWhiteAuto.Value = 'Once'

while True:
    # Get HSV values from the GUI sliders.
    lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
    lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
    lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
    highHue = cv2.getTrackbarPos('highHue', 'colorTest')
    highSat = cv2.getTrackbarPos('highSat', 'colorTest')
    highVal = cv2.getTrackbarPos('highVal', 'colorTest')

    with cam.RetrieveResult(1000) as img:
        frame = img.Array

    # # Show the original image.
    # cv2.imshow('frame', frame)

    # Blur methods available, comment or uncomment to try different blur methods.
    frameBGR = cv2.GaussianBlur(frame, (7, 7), 0)
    # frameBGR = cv2.medianBlur(frameBGR, 7)
    # frameBGR = cv2.bilateralFilter(frameBGR, 15 ,75, 75)
    """kernal = np.ones((15, 15), np.float32)/255
    frameBGR = cv2.filter2D(frameBGR, -1, kernal)"""

    # Show blurred image.
    # cv2.imshow('blurred', frameBGR)

    # HSV (Hue, Saturation, Value).
    # Convert the frame to HSV colour model.
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

    # HSV values to define a colour range.
    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    # Show the first mask
    # cv2.imshow('mask-plain', mask)

    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    # Show morphological transformation mask
    # cv2.imshow('mask', mask)

    # Put mask over top of the original image.
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show final output image
    cv2.imshow('colorTest', result)

    count_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(count_image, 50,255,1)
    ret, contours, h = cv2.findContours(thresh,1,2)
    number_of_objects = np.size(contours)
    print('number of objects is: '+str(number_of_objects))
    for cnt in contours:
        cv2.drawContours(result,[cnt], 0, (0,0,255),1)
    cv2.imshow('colorTest', result)
    cv2.imshow('contours', thresh)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()






