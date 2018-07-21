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

"""Image-to-text model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    self.input_file_pattern = None

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard = 2500
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1

    # Name of the SequenceExample context feature containing image data.
    self.image_feature_name = "image/data"
    # Name of the SequenceExample feature list containing integer captions.
    self.caption_feature_name = "image/caption_ids"

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 3000

    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 4

    # Batch size.
    self.batch_size = 32

    # File containing an VGG_19 checkpoint to initialize the variables
    # of the VGG_19 model. Must be provided when starting training for the
    # first time.
    self.vgg_checkpoint_file = None

    # Dimensions of vgg_19 input images.
    self.image_height = 224
    self.image_width = 224

    # Scale used to initialize model variables.
    self.initializer_scale = 0.08

    # The number of areas L = 14*14 = 196, dimensions D = 512
    self.context_shape = [196, 512]

    # Context dimensionality
    self.context_size = 512

    # LSTM input and output dimensionality, respectively.
    self.embedding_size = 512
    self.dim_embedding = 512
    self.num_lstm_units = 512

    # LSTM hidden layer dimensionlity
    self.hidden_size = 512

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    self.lstm_dropout_keep_prob = 0.7

    # If < 1.0, the dropout keep probability applied to commen layers
    self.dropout_keep_prob = 0.5

    # the number of time step which is equal to caption length-1 (16)
    self.n_time_step = 24

    # the generated caption length
    self.max_len = 20


    self.max_caption_length = 20
    self.num_initalize_layers = 2    # 1 or 2
    self.dim_initalize_layer = 512
    self.num_decode_layers = 2       # 1 or 2
    self.dim_decode_layer = 1024
        
    # attention mechanism
    self.attention_mechanism = "fc2"       # "fc1", "fc2", "rnn" or "bias"
    self.dim_attend_layer = 512     # for "fc1" and "fc2" only
    self.dim_rnn_att_state = 256    # for rnn only

    # about the weight initialization and regularization
    self.fc_kernel_initializer_scale = 0.08
    self.fc_kernel_regularizer_scale = 1e-4
    self.fc_activity_regularizer_scale = 0.0
    self.conv_kernel_regularizer_scale = 1e-4
    self.conv_activity_regularizer_scale = 0.0
    self.fc_drop_rate = 0.5
    self.lstm_drop_rate = 0.3
    self.attention_loss_factor = 0.01

    # about the optimization
    self.log_every_n_steps = 10
    self.optimizer = "Adam"   # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
    self.initial_learning_rate = 0.0001
    self.num_epochs = 100
#    self.num_steps_per_decay = 100000
    self.momentum = 0.0
    self.use_nesterov = True
    self.decay = 0.9
    self.centered = True
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.epsilon = 1e-6

    # about the saver
    self.save_period = 1000
    self.save_dir = './'
    self.eval_dir = './'
    # about the vocabulary
    self.vocabulary_file = './vocabulary.csv'
    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocabulary_size = 3000

    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 30000

    # Optimizer for training the model.
    #self.optimizer = "Adam"   # 'Adam', 'RMSProp', 'Momentum' or 'SGD'

    # Learning rate for the initial phase of training.
    #self.initial_learning_rate = 0.0001
    self.learning_rate_decay_factor = 0.5
    self.num_epochs_per_decay = 8.0

    # Learning rate when fine tuning the Inception v3 parameters.
    self.train_vgg_learning_rate = 0.0005
    self.train_vgg = False

    # If not None, clip gradients to this value.
    self.clip_gradients = 5.0

    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5
