#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhangping
# Create at : 2018.1.8
import os
import pdb
import tensorflow as tf
import numpy as np
import cnn_model
from planner_base import PlannerBase
from action_status import ActionStatus

DIMS = [100, 128, 1]  # Input feature dimensions, [feature_rows, feature_cols, depth].

class PlannerNeural(PlannerBase):
    """This is neural planner."""
    def __init__(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = os.path.split(os.path.realpath(__file__))[0]+'/checkpoint/'
        # Build model
        self.input_ = tf.placeholder(tf.float32, shape=[None, DIMS[0], DIMS[1], DIMS[2]]) # [batch,heigh,width,depth]
        self.keep_prob_ = tf.placeholder(tf.float32, shape=[])
        self.logits_ = cnn_model.inference_by_action(self.input_, self.keep_prob_)
        # Init model and load train model
        self.sess_ = tf.InteractiveSession()
        self.sess_.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint_file = checkpoint_dir + ckpt.model_checkpoint_path
            print 'Try to load training model %s.'%(checkpoint_file)
            saver.restore(self.sess_, checkpoint_file)
        else:
            warn = 'Failed to find train model at %s.'%(checkpoint_dir)
            raise Exception("InvalidCNNModel", warn)
        # Other
        self.as_ = ActionStatus()
        print "Neural Planner online..."


    def action_to_angle(self, action):
        """Transform Action to the angle corresponded."""
        #pdb.set_trace()
        print "action: ",action
        return self.as_.action_to_angle(action)


    def run(self, features):
        """Run neural planner, it will predict next action orientation
        baseed on input image.
        Args:
            features: input feature, [1, rows, cols , 2].
        Return:
            angle for acion, range from 0 to 359. The front is 0, clockwise.
        """
        action = self.sess_.run(self.logits_, feed_dict={self.input_:features, self.keep_prob_:1.0})
        angle = self.action_to_angle(action[0])
        return angle
