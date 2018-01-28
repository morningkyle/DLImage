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
import time
import argparse
import tensorflow as tf

from recorder import Recorder


def load_graph(model_file):
    start = time.time()
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    print(model_file)
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    print('Load graph: {:.3f}s'.format(time.time() - start))
    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    start = time.time()
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
    print('Read image data: {0:.3f}s, file: {1}'.format(time.time() - start, file_name))
    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def initialize_args():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image or image directory to be processed",
                        default=os.path.join(this_dir, "data/tf_files/panda/cropped_panda.jpg"))
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


def get_image_label(image_path):
    image_path = os.path.normpath(image_path)
    if os.path.isdir(image_path):
        return os.path.basename(image_path)
    elif os.path.isfile(image_path):
        return os.path.basename(os.path.dirname(image_path))
    else:
        return ''


def is_image_file(path):
    if not os.path.isfile(path):
        return False
    if path.endswith('.jpg') or path.endswith('.bmp') or path.endswith('.png'):
        return True
    return False


def get_image_files(path):
    """Return immediate files under @path"""
    if os.path.isfile(path):
        image_files = [path]
    elif os.path.isdir(path):
        image_files = []
        files = os.listdir(path)
        for f in files:
            p = os.path.join(path, f)
            if is_image_file(p):
                image_files.append(p)
    else:
        image_files = []
    return image_files


def classify(graph, args):
    input_operation = graph.get_operation_by_name("import/" + args.input_layer)
    output_operation = graph.get_operation_by_name("import/" + args.output_layer)

    images = get_image_files(args.image)
    labels = load_labels(args.labels)
    recorder = Recorder(labels)
    with tf.Session(graph=graph) as sess:
        total = len(images)
        for i, image in enumerate(images):
            t = read_tensor_from_image_file(image)
            start = time.time()
            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
            print('Evaluation time: {0:.3f}s, {1}/{2}'.format(time.time() - start,
                  i+1, total))
            recorder.add_result(results, image)

    dst = os.path.join('data/', get_image_label(args.image) + '.csv')
    recorder.save(dst)
    return recorder.df


if __name__ == "__main__":
    args = initialize_args()
    graph = load_graph(args.graph)
    df = classify(graph, args)

