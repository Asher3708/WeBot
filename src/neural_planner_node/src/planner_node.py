#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhangping
# Create at : 2018.1.8

import rospy
from neural_planner_node.srv import ActionOrientation, ActionOrientationResponse
from planner_neural import PlannerNeural
from images_input import ImageInput
from feature_extracter import feature_extracter
import pdb


PLANNER = None
IMG_INPUT = None
DIMS = [40, 40]

def planner_handler(req):
    """planner_handler"""
    global PLANNER
    global IMG_INPUT
    global DIMS
    # Request image
    try:
        img = IMG_INPUT.get()
        # Transform to grid feature
        feature = feature_extracter(img, DIMS)
        # Do work
        rows = feature.shape[0]
        cols = feature.shape[1]
        feature = feature.reshape(1, rows, cols, 2)
        angle = PLANNER.run(feature)
    except Exception as arg:
        print "Exception: ", arg
        angle = 0

    return ActionOrientationResponse(angle)


def planner_server():
    """planner_server"""
    global PLANNER
    global IMG_INPUT
    rospy.init_node('neural_planner_node')
    PLANNER = PlannerNeural()
    IMG_INPUT = ImageInput('/mybot/camera1/image_raw')

    server = rospy.Service('planner_service', ActionOrientation, planner_handler)
    print "Neural planner server online..."
    rospy.spin()


if __name__ == "__main__":
    planner_server()
