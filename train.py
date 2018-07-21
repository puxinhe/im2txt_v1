# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import configuration
import show_attend_tell_model

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "./data/flickr8k/tfrecord-data/train-?????-of-00012",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("vgg_checkpoint_file", "./data/vgg_19.ckpt",
                       "Path to a pretrained vgg_19 model.")
tf.flags.DEFINE_string("train_dir", "./data/flickr8k/output",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_vgg", False,
                        "Whether to train vgg submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 15, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.train_dir, "--train_dir is required"

  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = FLAGS.input_file_pattern
  model_config.vgg_checkpoint_file = FLAGS.vgg_checkpoint_file

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model = show_attend_tell_model.ShowAttendTellModel(
        model_config, mode="train", train_vgg=FLAGS.train_vgg)
    model.build()


    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=model_config.max_checkpoints_to_keep)

  # Run training.
  tf.contrib.slim.learning.train(
      model.opt_op,
      train_dir,
      log_every_n_steps=FLAGS.log_every_n_steps,
      graph=g,
      global_step=model.global_step,
      number_of_steps=FLAGS.number_of_steps,
      init_fn=model.init_fn,
      saver=saver)


if __name__ == "__main__":
  tf.app.run()
