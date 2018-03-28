#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhangping
# Create at : 2018.1.10
# Description: manage image input for planner.

from threading import Lock
from functools import partial
import copy
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import pdb

def image_callback(img_input, img_output, mutex):
    """Callback function for image topic.
    Args:
        img_input: topic will write image message to this argument.
        img_output: this function will write image to here.
        mutex: mutex for @img_output.
    Return: .
    """
    mutex.acquire()
    img_output.header = img_input.header
    img_output.height = img_input.height
    img_output.width = img_input.width
    img_output.encoding = img_input.encoding
    img_output.is_bigendian = img_input.is_bigendian
    img_output.step = img_input.step
    img_output.data = img_input.data
    mutex.release()

class ImageInput(object):
    """Class for image input."""
    def __init__(self, img_topic):
        self.mutex_ = Lock()
        self.bridge_ = CvBridge()
        self.img_ = Image()
        self.img_out_ = None
        callback = partial(image_callback, img_output=self.img_, mutex=self.mutex_)
        rospy.Subscriber(img_topic, Image, callback)


    def imgmsg_to_cv2_(self, imgmsg):
        """Transform message 'sensor_msgs/Image' to opencv image
        object(numpy.array).
        """
        try:
            cv_image = self.bridge_.imgmsg_to_cv2(imgmsg, "bgr8")
        except CvBridgeError as event:
            print event

        return cv_image


    def get(self):
        """Get a image.
        Args:
        Return: return a image, its a numpy.array(or image object for cv2). [height, width, 1].
        """
        self.mutex_.acquire()
        self.img_out_ = copy.copy(self.img_)
        self.mutex_.release()
        if self.img_out_ is not None:
            self.img_out_ = self.imgmsg_to_cv2_(self.img_out_)
            self.img_out_ = cv2.cvtColor(self.img_out_, cv2.COLOR_BGR2GRAY)
            return self.img_out_
        else:
            raise Exception("ImageInputError", "self.img_out_ is None")
