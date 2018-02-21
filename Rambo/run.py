import rospy
from steering_node import SteeringNode
from model import Rambo
import cv2
import argparse
import numpy as np
from keras.models import model_from_json
import tensorflow as tf

def process(model, img):
    steering = model.predict(img)
    print steering
    return steering


def get_model(model_file):
    with open('./models/rambo.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    print model.summary()
    model.load_weights(model_file)
    graph = tf.get_default_graph()
    return model, graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Runner')
    parser.add_argument('model', type=str, help='Path to model weights')
    args = parser.parse_args()
    node = SteeringNode(lambda: get_model(args.model), process)
    rospy.spin()