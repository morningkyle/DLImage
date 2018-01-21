# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import argparse

import numpy as np
import tensorflow as tf


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def print_result(results, labels):
    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]
    for i in top_k:
        print(labels[i], results[i])
    print('\n')


def initialize_args():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed",
                        default=os.path.join(this_dir, "data/tf_files/models/cropped_panda.jpg"))
    parser.add_argument("--graph", help="graph/model to be executed",
                        default=os.path.join(this_dir, "data/tf_files/retrained_graph.pb"))
    parser.add_argument("--labels", help="name of file containing labels",
                        default=os.path.join(this_dir, "data/tf_files/retrained_labels.txt"))
    parser.add_argument("--input_height", type=int, help="input height",
                        default=299)
    parser.add_argument("--input_width", type=int, help="input width",
                        default=299)
    parser.add_argument("--input_mean", type=int, help="input mean",
                        default=0)
    parser.add_argument("--input_std", type=int, help="input std",
                        default=255)
    parser.add_argument("--input_layer", help="name of input layer",
                        default="Mul")
    parser.add_argument("--output_layer", help="name of output layer",
                        default="final_result")
    args = parser.parse_args()

    return args


def get_image_list(image_path):
    if os.path.isfile(image_path):
        image_files = [image_path]
    elif os.path.isdir(image_path):
        image_files = []
        files = os.listdir(image_path)
        for f in files:
            f = os.path.join(image_path, f)
            if os.path.isfile(f):
                image_files.append(f)
    else:
        image_files = []
    return image_files


if __name__ == "__main__":
    args = initialize_args()

    graph = load_graph(args.graph)
    input_operation = graph.get_operation_by_name("import/" + args.input_layer)
    output_operation = graph.get_operation_by_name("import/" + args.output_layer)

    images = get_image_list(args.image)
    labels = load_labels(args.labels)
    with tf.Session(graph=graph) as sess:
        total = len(images)
        for i, image in enumerate(images):
            print("{0}, {1}/{2}".format(image, i, total))
            start = time.time()
            t = read_tensor_from_image_file(image)
            print('Read image data (1-image): {:.3f}s'.format(time.time() - start))

            start = time.time()
            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
            print('Evaluation time (1-image): {:.3f}s'.format(time.time() - start))
            print_result(results, labels)
