#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
import tensorflow as tf
from layers import conv2d, dense, flatten
import imutils
import model

test_images = []

class DrivingImitationNode():
    def __init__(self):
        self._cv_bridge = CvBridge()

        self._saver = tf.train.Saver()
        self._session = tf.InteractiveSession()

        init_op = tf.global_variables_initializer()

        self._session.run(init_op)

        self._saver.restore(self._session, tf.train.latest_checkpoint("/media/morty/SSD21/Shrijit/catkin_ws/src/driving-imitation/scripts/models"))

        self._sub = rospy.Subscriber("center_camera/image_color/compressed", CompressedImage, self.callback, queue_size=100000)

        self.steering_wheel  = cv2.imread("/media/morty/SSD21/Shrijit/catkin_ws/src/driving-imitation/scripts/steering_wheel.png")
        self.data = np.load("/media/morty/SSD21/Shrijit/catkin_ws/src/driving-imitation/scripts/processed/steering_angles.npy")
        self.smoothed_angle = 0.0


    def callback(self, image_msg):

        image = cv2.resize(self._cv_bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding="mono8"), (256, 192))
        cv2.imshow("Input Feed", self._cv_bridge.compressed_imgmsg_to_cv2(image_msg))
        cv2.waitKey(32)
        if (len(test_images) == 0):
            test_images.append(image)

        else:
            image2 = test_images.pop()

            if len(test_images) == 4:
                test_images.pop(0)

            image_diff = image2 - image
            image_diff = rescale_intensity(image_diff, in_range=(-255, 255), out_range=(0, 255))

            test_images.append(image_diff)
            test_images.append(image)


            if len(test_images) == 5:
                image_combined = np.array(test_images[:4])
                image_combined = np.transpose(image_combined, [1, 2, 0])
                cv2.imshow("Processed Image", image_combined[0])
                cv2.waitKey(1)
                predict_num = model.predict(self._session, image_combined, restore=False)

                angle = predict_num[0] * 180 / np.pi
                # angle = self.smoothen_angle(angle)
                # if (angle == None):
                #   return

                self.display_steering(angle)

    def process_image():
        pass

    def smoothen_angle(self, angle):
        if angle - self.smoothed_angle == 0:
            return None
        self.smoothed_angle += 0.1 * pow(abs((angle - self.smoothed_angle)), 2.0 / 3.0) * (angle - self.smoothed_angle) / abs(angle - self.smoothed_angle)
        return self.smoothed_angle

    def display_steering(self, angle):
        rows, cols, depth = self.steering_wheel.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), angle ,1)
        rotated = cv2.warpAffine(self.steering_wheel,M,(cols,rows))

        cv2.imshow("Steering Wheel", rotated)
        cv2.waitKey(32)

    def main(self):
        rospy.init_node('steering_predictor', anonymous=False)

        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('steering_predictor')
    tensor = DrivingImitationNode()
tensor.main()
