import tensorflow as tf
import numpy as np
from ops import image_embedding
from inference import *
from utils import vocabulary



FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("vocab_file", "data/flickr8k/word_counts.txt", "Text file containing the vocabulary.")

tf.flags.DEFINE_string("demo_method", "show", "demo model: show or show_attetion_focus")

tf.flags.DEFINE_string("checkpoint_path", "data/vgg_19.ckpt",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("graph_path", "data/train/freeze_model.pb",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("input_files", "data/image/test.jpg",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

def process_image(encoded_image,
                  height=224,
                  width=224,
                  image_format="jpeg"):


    # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
    with tf.name_scope("decode", values=[encoded_image]):
        if image_format == "jpeg":
           image = tf.image.decode_jpeg(encoded_image, channels=3)
        elif image_format == "png":
           image = tf.image.decode_png(encoded_image, channels=3)
        else:
           raise ValueError("Invalid image format: %s" % image_format)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Central crop, assuming resize_height > height, resize_width > width.
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)

    return image


def main(_):

    with tf.variable_scope('placeholder'):    
         image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")

    images = tf.expand_dims(process_image(image_feed), 0)
    vgg_output = image_embedding.vgg_19_extract(
        images,
        trainable=False,
        is_training=False)
    """
    vgg_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_19")
    """
    saver = tf.train.Saver()
    with tf.Session() as sess:
         saver.restore(sess, FLAGS.checkpoint_path)
         vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
         lstm = LSTMDecoder(FLAGS.graph_path,vocab,max_caption_length=20)

         filenames = []
         for file_pattern in FLAGS.input_files.split(","):
             filenames.extend(tf.gfile.Glob(file_pattern))
    
         for filename in filenames:
        
             with tf.gfile.FastGFile(filename, "rb") as f:
                  image_data = f.read()
             context = tf.reshape(vgg_output, [-1, 196, 512])
             context = sess.run(context,feed_dict={image_feed:image_data})
             tf.get_variable_scope().reuse_variables()
             feat = np.squeeze(context)

             caption, attention = lstm.decode(feat)
    
             #image = Image.open(FLAGS.input_files)
             image = load_image_into_numpy_array(filename)
             if FLAGS.demo_method == "show":
                lstm.show_caption(caption,image)
             else:
                lstm.show_attention(caption, attention, image, "./pic.jpg")
      
if __name__ == "__main__":
    tf.app.run()
