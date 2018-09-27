#!/usr/bin/env python

import rospy
import cv2
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
import time

def image_publisher():

	pub = rospy.Publisher("center_camera/image_color/compressed", Image, queue_size=100)
	rospy.init_node('image_publisher', anonymous=True)
	bridge = CvBridge()

	# if len(sys.argv) < 2:
	# 	print "You must give an argument to open a video stream (camera device/video file/url of a stream)"
	# 	exit(0)

	# resource = sys.argv[1]
	resource = "/media/morty/SSD2/Shrijit/catkin_ws/src/driving-imitation/challenge.mp4"
    # If we are given just a number, interpret it as a video device
	if len(resource) < 3:
		resource_name = "/dev/video" + resource
		resource = int(resource)
		vidfile = False
	else:
		resource_name = resource
		vidfile = True
	print "Trying to open resource: " + resource_name
	cap = cv2.VideoCapture(resource)
	if not cap.isOpened():
		print "Error opening resource: " + str(resource)
		print "Maybe opencv VideoCapture can't open it"
		exit(0)

	print "Correctly opened resource, starting to show feed."
	rval, frame = cap.read()
	while rval:
		frame = cv2.resize(frame, (640, 480))
		disp_frame = cv2.resize(frame, (640, 480))
		cv2.imshow("Stream: " + resource_name, disp_frame)
		rval, frame = cap.read()
		if frame is None:
			cap = cv2.VideoCapture(resource)
			rval, frame = cap.read()
		# ROS image stuff
		if vidfile and frame is not None:
			frame = np.uint8(frame)
		image_message = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

		pub.publish(image_message)

		key = cv2.waitKey(100)

		if key & 0xFF == ord('q'):
			return
	cv2.destroyWindow("preview")


if __name__ == '__main__':
	try:
		image_publisher()
	except rospy.ROSInterruptException:
		pass
