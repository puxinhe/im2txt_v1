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

"""Image-to-text implementation based on Show, Attend and Tell.

"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov,
Richard S. Zemel, Yoshua Bengio
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from utils.nn import NN
from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops


class ShowAttendTellModel(object):
  """Image-to-text implementation based on Show, Attend and Tell.

  "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
  Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov,
  Richard S. Zemel, Yoshua Bengio
  """

  def __init__(self, config, mode, dropout=True, train_vgg=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_vgg: Whether the VGG submodel variables are trainable.
    """

    self.nn = NN(config)
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.is_train = True if mode=="train" else False

    self.train_vgg = train_vgg

    # whether use dropout or not
    self.dropout = dropout

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show Attend Tell" paper we initialize all variables with a
    # truncated_normal initializer.
    self.initializer = tf.contrib.layers.xavier_initializer()
    self.const_initializer = tf.constant_initializer(0.0)
    self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # An int32 recoder maximum padded_length
    self.caption_length = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the batch loss for the trainer to optimize.
    self.batch_loss = None
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # Collection of variables from the vgg submodel.
    self.vgg_variables = []

    # Context encode [batch_size, ]
    self.context = None
    self.context_encode = None

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

    # [batch_size, max_len]
    self.sampled_word_list = None

    # [vocab_size, embedding_size]
    self.embedding_map = None

    # [batch_size, caption_length, context_size]
    self.alphas = None

    # [batch_size, capiton_length]
    self.betas = None


  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def _batch_norm(self, x, mode='train'):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=0.95,
          center=True,
          scale=True,
          is_training=(mode == 'train'),
          updates_collections=None,
          scope='batch_norm'
      )


  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)

      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity,
                                           n_time_step=self.config.n_time_step))

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask


  def build_context_encode(self):
    """Builds the image model subgraph and generates context_encode.

    Inputs:
      self.images

    Outputs:
      self.context_encode
    """
    vgg_output = image_embedding.vgg_19_extract(
        self.images,
        trainable=self.train_vgg,
        is_training=self.is_training())
    self.vgg_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_19")
    self.num_ctx = self.config.context_shape[0]
    self.dim_ctx = self.config.context_shape[1]
    context = tf.reshape(vgg_output, [-1, self.num_ctx, self.dim_ctx])

    # Batch normalize feature vector
    if self.mode == "train":
      context = self._batch_norm(context, 'train')
    else:
      context = self._batch_norm(context, 'test')

    self.conv_feats = context

    # Save the context_shape in the graph.
    tf.constant(self.config.context_size, name="context_encode_size")

#    self.context_encode = context_encode

  def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        # Setup the placeholders
        if self.is_train:
            contexts = self.conv_feats
            sentences = self.input_seqs
            masks = tf.cast(self.input_mask, tf.float32)
        else:
            if self.mode =='eval':
                contexts = self.conv_feats
            else:
                contexts = tf.placeholder(
                    dtype = tf.float32,
                    shape = [config.batch_size, self.num_ctx, self.dim_ctx],
                    name = 'contexts')
            last_memory = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units],
                name = 'last_memory')
            last_output = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units],
                name = 'last_output')
            last_word = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size],
                name = 'last_word')

        # Setup the word embedding
        with tf.variable_scope("word_embedding", tf.device("/cpu:0")):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(contexts, axis = 1)
            initial_memory, initial_output = self.initialize(context_mean)
            initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        if self.is_train:
            alphas = []
            cross_entropies = []
            predictions_correct = []
            num_steps = self.config.max_caption_length
            last_output = initial_output
            last_memory = initial_memory
            last_word = sentences[:, 0]
        else:
            num_steps = 1
        last_state = last_memory, last_output

        # Generate the words one by one
        for idx in range(1,num_steps+1):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(contexts, last_output)
                context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
                                        axis = 1)
                if self.is_train:
                    tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
                                         [1, self.num_ctx])
                    masked_alpha = alpha * tiled_masks
                    alphas.append(tf.reshape(masked_alpha, [-1]))

            # Embed the last word
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = tf.concat([context, word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             context,
                                             word_embed],
                                             axis = 1)
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

            # Compute the loss for this step, if necessary
            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = sentences[:, idx],
                    logits = logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                last_output = output
                last_memory = memory
                last_state = state
                last_word = sentences[:, idx]

            tf.get_variable_scope().reuse_variables()


        # Compute the final loss, if necessary
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)

            alphas = tf.stack(alphas, axis = 1)
            alphas = tf.reshape(alphas, [config.batch_size, self.num_ctx, -1])
            attentions = tf.reduce_sum(alphas, axis = 2)
            diffs = tf.ones_like(attentions) - attentions
            attention_loss = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs) \
                             / (config.batch_size * self.num_ctx)

            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + attention_loss + reg_loss

            predictions_correct = tf.stack(predictions_correct, axis = 1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)

        self.contexts = contexts
        if self.is_train:
            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.attention_loss = attention_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
            self.attentions = attentions
        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.last_memory = last_memory
            self.last_output = last_output
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.probs = probs
            self.alpha = alpha

        print("RNN built.")

  def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config
        context_mean = self.nn.dropout(context_mean)
        if config.num_initalize_layers == 1:
            # use 1 fc layer to initialize
            memory = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a')
            output = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b')
        else:
            # use 2 fc layers to initialize
            temp1 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_a1')
            temp1 = self.nn.dropout(temp1)
            memory = self.nn.dense(temp1,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a2')

            temp2 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_b1')
            temp2 = self.nn.dropout(temp2)
            output = self.nn.dense(temp2,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b2')
        return memory, output

  def fc1_attend(self, contexts, output):
        """use 1 fully connected layer to attend.

        Args:
        contexts: image feature of shape [batchsize 100 2048] after reshape, 
                  become [batchsize*100 2048].
        output: LSTM last generated hidden state.

        Returns:
        Attention weights alpha, has shape [batchsize 100].
        """
        print("fc1 attend")
        logits1 = self.nn.dense(contexts,
                                units = 1,
                                activation = None,
                                use_bias = False,
                                name = 'fc_a')
        logits1 = tf.reshape(logits1, [-1, self.num_ctx])
        logits2 = self.nn.dense(output,
                                units = self.num_ctx,
                                activation = None,
                                use_bias = False,
                                name = 'fc_b')
        logits = logits1 + logits2
        alpha = tf.nn.softmax(logits)
        return alpha

  def fc2_attend(self, contexts, output):
        """use 2 fully connected layer to attend.

        Args:
        contexts: image feature of shape [batchsize 100 2048] after reshape, 
                  become [batchsize*100 2048].
        output: LSTM last generated hidden state.

        Returns:
        Attention weights alpha, has shape [batchsize 100].
        """
        print("fc2 attend")
        temp1 = self.nn.dense(contexts,
                              units = self.config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'fc_1a')
        temp2 = self.nn.dense(output,
                              units = self.config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'fc_1b')
        temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
        temp2 = tf.reshape(temp2, [-1, self.config.dim_attend_layer])
        temp = temp1 + temp2
        temp = self.nn.dropout(temp)
        logits = self.nn.dense(temp,
                               units = 1,
                               activation = None,
                               use_bias = False,
                               name = 'fc_2')
        logits = tf.reshape(logits, [-1, self.num_ctx])
        alpha = tf.nn.softmax(logits)
        return alpha

  def bias_attend(self, contexts, output):
        """Use 1 fully connected layer to attend. Add bias when calculate softmax so
        that LSTM is not necessarily turn to image feature generating each
        word.

        Args:
        contexts: image feature of shape [batchsize 100 2048] after reshape, 
                  become [batchsize*100 2048].
        output: LSTM last generated hidden state.

        Returns:
        Attention weights alpha, has shape [batchsize 100].
        """
        print("bias attend")
        logits1 = self.nn.dense(contexts,
                                units = 1,
                                activation = None,
                                use_bias = False,
                                name = 'fc_a')
        logits1 = tf.reshape(logits1, [-1, self.num_ctx])
        logits2 = self.nn.dense(output,
                                units = self.num_ctx,
                                activation = None,
                                use_bias = False,
                                name = 'fc_b')
        logits = logits1 + logits2
        attend_bias = tf.get_variable("attend_bias",[self.config.batch_size,1],
                                    initializer=tf.constant_initializer(0.0))
        bias_logits = tf.concat([logits,attend_bias],axis=1,name='attend_bias_logits')
        bias_alpha = tf.nn.softmax(bias_logits)
        alpha = tf.slice(bias_alpha,[0,0],[self.config.batch_size,self.num_ctx])
        return alpha

  def bias2_attend(self, contexts, output):
        """use 2 fully connected layer to attend.

        Args:
        contexts: image feature of shape [batchsize 100 2048] after reshape, 
                  become [batchsize*100 2048].
        output: LSTM last generated hidden state.

        Returns:
        Attention weights alpha, has shape [batchsize 100].
        """
        print("bias2 attend")
        temp1 = self.nn.dense(contexts,
                              units = self.config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'fc_1a')
        temp2 = self.nn.dense(output,
                              units = self.config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'fc_1b')
        temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
        temp2 = tf.reshape(temp2, [-1, self.config.dim_attend_layer])
        temp = temp1 + temp2
        temp = self.nn.dropout(temp)
        logits = self.nn.dense(temp,
                               units = 1,
                               activation = None,
                               use_bias = False,
                               name = 'fc_2')
        logits = tf.reshape(logits, [-1, self.num_ctx])
        
        attend_bias = tf.get_variable("attend_bias",[self.config.batch_size,1],
                                    initializer=tf.constant_initializer(0.0))
        bias_logits = tf.concat([logits,attend_bias],axis=1,name='attend_bias_logits')
        bias_alpha = tf.nn.softmax(bias_logits)
        alpha = tf.slice(bias_alpha,[0,0],[self.config.batch_size,self.num_ctx])
        return alpha

  def bias_fc1_attend(self, contexts, output):
        """Use 1 fully connected layer to calculate bias. 

        Args:
        contexts: image feature of shape [batchsize 100 2048] after reshape, 
                  become [batchsize*100 2048].
        output: LSTM last generated hidden state.

        Returns:
        Attention weights alpha, has shape [batchsize 100].
        """
        print("bias_fc1 attend")
        logits1 = self.nn.dense(contexts,
                                units = 1,
                                activation = None,
                                use_bias = False,
                                name = 'fc_a')
        logits1 = tf.reshape(logits1, [-1, self.num_ctx])
        logits2 = self.nn.dense(output,
                                units = self.num_ctx,
                                activation = None,
                                use_bias = False,
                                name = 'fc_b')
        logits = logits1 + logits2
        attend_bias = self.nn.dense(output,
                                units = 1,
                                activation = None,
                                use_bias = False,
                                name = 'attend_bias')
        bias_logits = tf.concat([logits,attend_bias],axis=1,name='attend_bias_logits')
        bias_alpha = tf.nn.softmax(bias_logits)
        alpha = tf.slice(bias_alpha,[0,0],[self.config.batch_size,self.num_ctx])
        return alpha
        
  def bias_fc2_attend(self, contexts, output):
        """use 2 fully connected layer to calculate bias.

        Args:
        contexts: image feature of shape [batchsize 100 2048] after reshape, 
                  become [batchsize*100 2048].
        output: LSTM last generated hidden state.

        Returns:
        Attention weights alpha, has shape [batchsize 100].
        """
        print("bias_fc2 attend")
        temp1 = self.nn.dense(contexts,
                              units = self.config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'fc_1a')
        temp2 = self.nn.dense(output,
                              units = self.config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'fc_1b')

        bias_temp1 = tf.reshape(temp1, [-1, self.num_ctx, self.config.dim_attend_layer])
        bias_temp1 = tf.reduce_max(bias_temp1, axis=1)
        attend_bias = bias_temp1 + temp2
        attend_bias = self.nn.dense(attend_bias,
                               units = 1,
                               activation = None,
                               use_bias = False,
                               name = 'attend_bias')

        temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
        temp2 = tf.reshape(temp2, [-1, self.config.dim_attend_layer])
        temp = temp1 + temp2
        temp = self.nn.dropout(temp)
        logits = self.nn.dense(temp,
                               units = 1,
                               activation = None,
                               use_bias = False,
                               name = 'fc_2')
        logits = tf.reshape(logits, [-1, self.num_ctx])
        
        bias_logits = tf.concat([logits,attend_bias],axis=1,name='attend_bias_logits')
        bias_alpha = tf.nn.softmax(bias_logits)
        alpha = tf.slice(bias_alpha,[0,0],[self.config.batch_size,self.num_ctx])
        return alpha

  def rnn_attend(self, contexts, output):
        """Use rnn to calculate attention weights. 

        Args:
        contexts: image feature of shape [batchsize 100 2048] after reshape, 
                  become [batchsize*100 2048].
        output: LSTM last generated hidden state.

        Returns:
        Attention weights alpha, has shape [batchsize 100].
        """
        print("rnn attend")

        if self.rnn_attend_state is None:
            encode_contex = tf.reshape(contexts, [-1, self.num_ctx, self.dim_ctx])
            encode_contex = tf.reduce_max(encode_contex, axis=1)
            self.rnn_attend_state = self.nn.dense(encode_contex,
                              units = self.config.dim_rnn_att_state,
                              activation = tf.tanh,
                              name = 'rnn_att_init_state')  
        # update hidden state
        self.rnn_attend_state = self.nn.dense(
                              tf.concat([output,self.rnn_attend_state],1),
                              units = self.config.dim_rnn_att_state,
                              activation = tf.tanh,
                              use_bias = True,
                              name = 'rnn_att_update')

        # calculate output
        logits = self.nn.dense(self.rnn_attend_state,
                              units = self.num_ctx,
                              activation = None,
                              use_bias = False,
                              name = 'rnn_att_output')  
        alpha = tf.nn.softmax(logits)
        return alpha

  def attend(self, contexts, output):
        """ Attention Mechanism. """
        ATTENTION_MAP = {
            'fc1': self.fc1_attend,
            'fc2': self.fc2_attend,
            'bias': self.bias_attend,
            'bias2': self.bias2_attend,
            'bias_fc1': self.bias_fc1_attend,
            'bias_fc2': self.bias_fc2_attend,
            'rnn': self.rnn_attend,
        }
        reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        output = self.nn.dropout(output)

        att_fn = ATTENTION_MAP[self.config.attention_mechanism]
        return att_fn(reshaped_contexts,output)


  def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

  def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        config = self.config

        learning_rate_decay_fn = None
        if config.train_vgg:
           learning_rate = tf.constant(config.train_vgg_learning_rate)
        else:
          learning_rate = tf.constant(config.initial_learning_rate)
          if config.learning_rate_decay_factor < 1.0:
             num_batches_per_epoch = (config.num_examples_per_epoch /
                                 config.batch_size)
             decay_steps = int(num_batches_per_epoch *
                          config.num_epochs_per_decay)

             def _learning_rate_decay_fn(learning_rate, global_step):
                 return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = decay_steps, #config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
             learning_rate_decay_fn = _learning_rate_decay_fn
          else:
             learning_rate_decay_fn = None
        
        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                print('Adam')
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                print('RMSProp')
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                print('Momentum')
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                print("SGD")
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

  def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        print("build summary")
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("attention_loss", self.attention_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("attentions"):
            self.variable_summary(self.attentions)

        self.summary = tf.summary.merge_all()

  def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


  def setup_vgg_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.vgg_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring vgg variables from checkpoint file %s",
                        self.config.vgg_checkpoint_file)
        saver.restore(sess, self.config.vgg_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    
    self.setup_global_step()
    self.build_inputs()
    self.build_context_encode()
    self.build_rnn()
    if self.is_train:
            self.build_optimizer()
            self.build_summary()
#    self.build_seq_embeddings()
#    self.build_model()
    self.setup_vgg_initializer()
