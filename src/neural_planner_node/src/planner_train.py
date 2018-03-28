#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhangping
# Create at : 2018.1.25

import tensorflow as tf
from images_dataset import read_dataset_and_transform
import cnn_model
import pdb

FLAGS = tf.app.flags.FLAGS

"""Config"""
class CheckpointFlags:
    """This is used to save checkpoint config."""
    def __init__(self, checkpoint_dir, checkpoint_file, save_stride):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint_steps = save_stride

class TrainFlags:
    """This is used to save train config."""
    def __init__(self, train_dir, train_file, train_label):
        self.train_dir = train_dir
        self.train_file = train_file
        self.train_label = train_label

CFLAGS = CheckpointFlags('checkpoint/', 'model.ckpt', 100)
TFLAGS = TrainFlags('train_data/', 'train-dataset-images.gz'\
  , 'train-dataset-labels.gz')

GRID_FEATURE_SHAPE = [100, 128, 1] # grid feature shape, [feature_rows, feature_cols, depth]
GRID_SHAPE = [3, 3] # grid shape, image will be divided into 3x3 region.

"""Main body"""
def load_grid_feature_dataset():
    """Load train data set and transform to grid feature.
    Args:
      Return: return dataset.
    """
    global TFLAGS
    global GRID_SHAPE
    global GRID_FEATURE_SHAPE
    #dataset = read_dataset_and_transform(TFLAGS.train_dir, TFLAGS.train_file\
    #    , TFLAGS.train_label, GRID_SHAPE, GRID_FEATURE_SHAPE, one_hot=False)
    dataset = None
    return dataset


def train(dataset):
    """Train this network model.
    Args:
        datasets: contain two elements: a tensor for feature,shape is [batch,height,width,channels]
                , and a tensor for label, shape is [batch, label_num].
    Return:
    """
    global GRID_FEATURE_SHAPE
    global FLAGS

    with tf.Graph().as_default():
        global_step = tf.Variable(2, trainable=False)
        """Build network."""
        x = tf.placeholder(tf.float32, shape=[None, GRID_FEATURE_SHAPE[0]\
            , GRID_FEATURE_SHAPE[1], GRID_FEATURE_SHAPE[2]]) # [batch,heigh,width,depth]
        y_ = tf.placeholder(tf.float32, shape=[None])   # [batch]
        keep_prob = tf.placeholder(tf.float32, shape=[])
        logits = cnn_model.inference(x, keep_prob)
        """get variable"""
        # For Debug

        """build loss function and optimizer."""
        loss = cnn_model.loss(logits, y_)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        """build accuracy function"""
        # for column
        #pdb.set_trace()
        prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y_, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        """train and save new model."""
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            """init saver and load old model."""
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(CFLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print 'Find old model %s, and try to restore it.'%(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            """training"""
            print 'Begin trainning....'
            for i in xrange(global_step.eval(), 10000):
                batch = dataset.next_batch(FLAGS.batch_size)
                #tmp = sess.run(logits,feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
                #pdb.set_trace()
                train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
                # show train accuracy
                if (i+1)%10 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:batch[0]\
                                    , y_:batch[1], keep_prob:1.0})
                    print "step %d, trainning accuracy %g"%(i, train_accuracy)
                # save checkpoint
                if (i+1)%CFLAGS.checkpoint_steps == 0:
                    sess.run(tf.assign(global_step, i+1))
                    saver.save(sess, CFLAGS.checkpoint_dir + CFLAGS.checkpoint_file \
                                , global_step=global_step)

            print 'End trainning...'
            # Test
            # ...


def main(argv=None):  # pylint: disable=unused-argument
    """Load dataset."""
    dataset = load_grid_feature_dataset()
    if not tf.gfile.Exists(CFLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(CFLAGS.checkpoint_dir)

    """train and save"""
    train(dataset)


if __name__ == '__main__':
    tf.app.run()
