### 简介
Image2Caption项目是基于论文 "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"(arXiv:1502.03044) . 输入一个图片后，系统可以输出一句话来描述图片的内容。系统先是通过提取图片在VGG19网络中vgg_19/conv5/conv5_3的特征，然后输入到LSTM网格中解码成相应的描述语句。在解码过程中加入了 soft attention的机制去提高描述语句的输出质量。这个项目使用了Tensorflow学习框架来实现。可以先在保持VGG19网络参数不变的情况下训练LSTM网络，然后再联合训练VGG19和LSTM网络。

### 需要安装和使用到的库文件
* **Tensorflow** ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](https://scipy.org/install.html))
* **OpenCV** ([instructions](https://pypi.python.org/pypi/opencv-python))
* **Natural Language Toolkit (NLTK)** ([instructions](http://www.nltk.org/install.html))
* **Matplotlib** ([instructions](https://scipy.org/install.html))

### 具体使用

* **数据准备**：从http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_image_Description/KCCA.html 下载Flickr8k数据集
* **数据集生成**：运行data/build_flicker8k_data.py程序，把Flickr8k打包成TFRecord格式的数据集
```shell
缺省参数设置：
train_image_dir=“./flicker8k/images/”
val_image_dir=“./flicker8k/images/”
test_image_dir=“./flicker8k/images/”
output_dir="./flicker8k/tfrecord-data/"
train_text_file="./flicker8k/Flickr_8k.trainImages.txt"
val_text_file="./flicker8k/Flickr_8k.devImages.txt"
test_text_file="./flicker8k/Flickr_8k.testImages.txt"
captions_file="./flicker8k/Flickr8k.token.txt"
word_counts_output_file="./flicker8k/word_counts.txt"
train_shards=12
val_shards=2
test_shards=2
num_threads=4

python data/build_flicker8k_data.py 
```


* **训练:**
训练主程序是train.py, 所有与训练有关的主要参数都放在configuration.py，在命令行下运行:
```shell
python train.py \
--input_file_pattern="./data/flickr8k/tfrecord-data/train-?????-of-00012" \
--vgg_checkpoint_file="./data/vgg_19.ckpt" \
--train_dir="./data/flickr8k/output"
--number_of_steps=1000000 \
--train_vgg=False \
--log_every_n_steps=1
```
在训练过程中会输出summary文件，可以通过运行tensorboard，在chrome中查看:
```shell
tensorboard --logdir='./data/flickr8k/output'
```

* **验证:**
用验证数据集去验证:
```shell
python evaluate.py 
--input_file_pattern="./data/flickr8k/tfrecord-data/val-?????-of-00012" \
--checkpoint_dir="./data/flickr8k/output" \
--eval_dir="./data/flickr8k/output" \
--min_global_step=1 \
--num_eval_examples=32 \

```
* **推断:**

  支持从命令行和web页面两种方式，去输入一张图片，让系统进行推断，输出相应的描述。

  从命令行，

  ```shell
  python inference_run.py \
  --vocab_file="data/flickr8k/word_counts.txt"
  --demo_method="show"(或者“show_attetion_focus”)
  --checkpoint_path="data/vgg_19.ckpt"
  --graph_path="data/train/freeze_model.pb"
  --input_files="data/image/test.jpg"（支持多个jpg文件的输入，用“，”分隔）
  ```

  还可以直接执行infer.cmd或infer_att.cmd这两个批处理，前者执行后显示图片和描述，后者会按照attention可视化的方式显示图片和描述，以及每个attention关注的单词。

  

  从web页面：

​       先要安装 Flask库文件，如果安装过Anaconda,就不需要单独再安装了，系统已经包括了。

```
pip install Flask
```
​      然后在命令行执行，
```shell
python web_server.py \
--checkpoint_path="data/vgg_19.ckpt" \
--graph_path="data/train/freeze_model.pb" \ 
--vocab_file="data/flickr8k/word_counts.txt" \
```
 根据屏幕提示，用浏览器打开 http://127.0.0.1:5000



### 



# im2txt_v1
