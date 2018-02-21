import threading
import numpy as np
import rospy
# from dbw_mkz_msgs.msg import SteeringCmd
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge,CvBridgeError
from skimage.exposure import rescale_intensity
import os
import pandas as pd


data_dir = "./data/"
X_train_mean = np.load(os.path.join(data_dir, "X_train_mean.npy"))


class SteeringNode(object):
	def __init__(self, get_model_callback, model_callback):
		rospy.init_node('steering_node')
		self.model, self.graph = get_model_callback()
		self.get_model = get_model_callback
		self.predict = model_callback
		self.bridge = CvBridge()
		self.prev_img = None
		self.img = np.zeros((1, 192, 256, 4), dtype=np.float32)
		self.series = []
		self.wheel = cv2.imread('steering_wheel_image.png')
		self.steering = 0.
		self.smoothed_angle = 0.
		self.rows = self.wheel.shape[0]
		self.cols = self.wheel.shape[1]
		self.image_lock = threading.RLock()
		self.image_sub = rospy.Subscriber('zed/left/image_rect_color', Image, self.update_image)
		# self.pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
		rospy.Timer(rospy.Duration(.02), self.get_steering)

	def update_image(self, img):
		try:
			img = self.bridge.imgmsg_to_cv2(img,"bgr8")
		except CvBridgeError, e:
			print(e)
		else:
			img = cv2.resize(img, (256,192))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = img.astype('float32')
			if self.prev_img is None:
				self.prev_img = img
				return
			diff = img - self.prev_img
			diff = rescale_intensity(diff, in_range=(-255, 255), out_range=(0, 255))
			self.series.append(diff)
			self.prev_img = img

			if len(self.series) != 4:
				return

			if self.image_lock.acquire(True):
				for i in range(4,0,-1):
					self.img[0,:,:,4-i] = self.series[i-1]
				
				if self.model is None:
					self.model = self.get_model()
				
				self.series.pop(0)
				self.img -= X_train_mean
				self.img /= 255.0
				
				with self.graph.as_default():
					self.steering = self.predict(self.model, self.img)
				
				self.image_lock.release()

	def get_steering(self, event):
		if self.img is None:
			return

		# message = SteeringCmd()
		# message.enable = True
		# message.ignore = False
		# message.steering_wheel_angle_cmd = self.steering
		# self.pub.publish(message)

		angle = self.steering * 180 / np.pi
		if angle - self.smoothed_angle == 0:
			return
		self.smoothed_angle += 0.2 * pow(abs((angle - self.smoothed_angle)), 2.0 / 3.0) * (angle - self.smoothed_angle) / abs(angle - self.smoothed_angle)
		M = cv2.getRotationMatrix2D((self.cols/2,self.rows/2),angle,1)
		dst = cv2.warpAffine(self.wheel,M,(self.cols,self.rows))
		cv2.imshow("Pedicted", dst)
		if cv2.waitKey(10) == ord('q'):
			quit()