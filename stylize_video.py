# Copyright (c) 2016-2017 Shafeen Tejani. Released under GPLv3.

import os

import numpy as np
from os.path import exists
from sys import stdout

import utils
from argparse import ArgumentParser
import tensorflow as tf
import transform

import cv2

NETWORK_PATH='networks'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', type=str,
                        dest='content', help='content video path',
                        metavar='CONTENT', required=True)

    parser.add_argument('--network-path', type=str,
                        dest='network_path',
                        help='path to network (default %(default)s)',
                        metavar='NETWORK_PATH', default=NETWORK_PATH)

    parser.add_argument('--output-path', type=str,
                        dest='output_path',
                        help='path for output',
                        metavar='OUTPUT_PATH', required=True)

    return parser

def check_opts(opts):
    assert exists(opts.content), "content not found!"
    assert exists(opts.network_path), "network not found!"


def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    if not os.path.isdir(options.network_path):
        parser.error("Network %s does not exist." % options.network_path)

    sess = tf.Session()
    img_placeholder, network = None, None

    wrt = None
    i = 0

    content_images = utils.load_video(options.content)
    for content_image in content_images:
        reshaped_content_height = (content_image.shape[0] - content_image.shape[0] % 4)
        reshaped_content_width = (content_image.shape[1] - content_image.shape[1] % 4)
        reshaped_content_image = content_image[:reshaped_content_height,
                                               :reshaped_content_width, :]
        reshaped_content_image = np.ndarray.reshape(
            reshaped_content_image, (1,) + reshaped_content_image.shape)

        if img_placeholder is None:
            img_placeholder = tf.placeholder(
                tf.float32,
                shape=reshaped_content_image.shape,
                name='img_placeholder')
            network = transform.net(img_placeholder)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(options.network_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")

        prediction = sess.run(
            network, feed_dict={img_placeholder:reshaped_content_image})[0]
        i += 1

        if wrt is None:
            wrt = cv2.VideoWriter(options.output_path,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  24.0,
                                  (reshaped_content_width,
                                   reshaped_content_height))
        wrt.write(np.clip(prediction, 0, 255).astype(np.uint8))
        print i

    wrt.release()
    sess.close()


if __name__ == '__main__':
    main()
