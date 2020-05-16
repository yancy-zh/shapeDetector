# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
# import argparse
# import imutils
import cv2
import matplotlib.pyplot as pyplot
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the input image")
# args = vars(ap.parse_args())

input_path = r"C:\work\shape-detection\\"
fname="shapes_and_colors.png"
# image = cv2.imread(args["image"])
# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(input_path+fname, -1)

# resized = imutils.resize(image, width=300)
#image size 600*561
resized= cv2.resize(image, None, fx=1, fy=1,
            interpolation=cv2.INTER_CUBIC)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
fig=pyplot.figure()
w_img=image.shape[0]
h_img=image.shape[1]
pyplot.rcParams["figure.figsize"]=[w_img*2, h_img*2]
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
ax1=fig.add_subplot(221)	 #plot graph and histogram
ax1.imshow(gray, cmap="gray")
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ax2=fig.add_subplot(222)
ax2.imshow(blurred, cmap="gray")
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
ax3=fig.add_subplot(223)
ax3.imshow(thresh, cmap="gray")
# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = cnts[0]
# cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
# ax4=fig.add_subplot(224)
# ax4.imshow(image)
# pyplot.show()

# cv2.imshow("Image", image)
# cv2.waitKey(0)

sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    cv2.putText(image, "curr", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    ax4=fig.add_subplot(224)
    ax4.imshow(image)
    pyplot.show()
    shape = sd.detect(c)
#     print "current contour center is: {} {}".format(cX, cY)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)