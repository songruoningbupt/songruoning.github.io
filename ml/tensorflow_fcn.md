### FCN/TensorFlow

*参考 https://blog.csdn.net/MOU_IT/article/details/81073149 ，代码来自 https://github.com/shekkizh/FCN.tensorflow ，自己开始一点点在Jupyter上试*

*数据 The model was applied on the Scene Parsing Challenge dataset provided by MIT http://sceneparsing.csail.mit.edu/ ，数据说明 https://github.com/CSAILVision/sceneparsing#overview-of-scene-parsing-benchmark，数据和模型如果不提前下载，程序里也会下载*

百度云分享一下：
- Training Set/Validation Set链接: https://pan.baidu.com/s/1hDGlYIiCDlbi4VK_37FarQ 密码: gwhe
- Test set链接: https://pan.baidu.com/s/1BZJM9ccrgtNLz0xT43nqUA 密码: x5hi
- VGG网络的权重参数链接: https://pan.baidu.com/s/1kl4CnXc8xcPawQf0WoOZ4Q 密码: du1h



```python
from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "data/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "model/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

# 迭代的最大次数
MAX_ITERATION = int(1e3 + 1)
# 分类的个数
NUM_OF_CLASSESS = 151
# 图片尺寸
IMAGE_SIZE = 224

# vgg_net：根据权重构建VGG网络
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

# 定义Semantic segmentation network，使用VGG结构
def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    # download Model，建议提前下好，这样不会重新下载
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    # batch normalization
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    # layers字段，所有的权重都存在这里面
    weights = np.squeeze(model_data['layers'])

    # image - mean_pixel：每一channel的均值
    processed_image = utils.process_image(image, mean_pixel)

    # 以inference为名的命名空间
    with tf.variable_scope("inference"):
        # 构建VGG网络
        image_net = vgg_net(weights, processed_image)
        # 最后一层
        conv_final_layer = image_net["conv5_3"]
        # 最后添加一层2*2的max pool
        pool5 = utils.max_pool_2x2(conv_final_layer)

        # 再加conv6，conv7，conv8三个卷基层，都用的ReLU
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        # 进行deconv操作，依次获取前面卷积前的图片大小
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

# 主函数
def main(argv=None):
    # placeholder 定义输入，keep_probability隐含层节点保持工作的概率，这是个什么效果
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    # 输入图像，3是指channel？
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    # 标注图像，只有1个channel
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    # 构建训练模型
    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    # loss函数，
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]),name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)
    # 优化器
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    # 会下载Train和Valid数据
    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print("Train Set Size: " + str(len(train_records)))
    print("Valid Set Size: " + str(len(valid_records)))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    # 如果是train模式
    # 迭代MAX_ITERATION次
    #     读取下一批次图像与标注进行训练
    #     每10轮输出一下train_loss，500轮输出一下验证集的loss
    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)

if __name__ == "__main__":
    tf.app.run()
```

运行的结果

```
    setting up vgg initialized conv layers ...
    Setting up summary op...
    Setting up image reader...
    Found pickle file!
    Train Set Size: 20210
    Valid Set Size: 2000
    Setting up dataset reader
    Initializing Batch Dataset Reader...
    {'resize': True, 'resize_size': 224}
    (20210, 224, 224, 3)
    (20210, 224, 224, 1)
    Initializing Batch Dataset Reader...
    {'resize': True, 'resize_size': 224}
    (2000, 224, 224, 3)
    (2000, 224, 224, 1)
    Setting up Saver...
    INFO:tensorflow:Restoring parameters from logs/model.ckpt-47500
    Model restored...
    Step: 0, Train_loss:2.07889
    2018-08-12 21:02:26.980401 ---> Validation_loss: 1.65172
    Step: 10, Train_loss:1.6146
    Step: 20, Train_loss:1.09385
    Step: 30, Train_loss:0.841486
    Step: 40, Train_loss:1.16472
    Step: 50, Train_loss:0.857076
    Step: 60, Train_loss:2.01374
    Step: 70, Train_loss:1.29242
    Step: 80, Train_loss:1.55015
    Step: 90, Train_loss:1.49861
    Step: 100, Train_loss:1.05323
    Step: 110, Train_loss:2.56539
    Step: 120, Train_loss:2.41875
    Step: 130, Train_loss:1.68616
    Step: 140, Train_loss:2.45747
    Step: 150, Train_loss:1.23132
    Step: 160, Train_loss:0.801146
    Step: 170, Train_loss:1.50493
    Step: 180, Train_loss:2.83727
    Step: 190, Train_loss:1.34507
    Step: 200, Train_loss:1.6949
    Step: 210, Train_loss:1.19958
    Step: 220, Train_loss:0.948488
    Step: 230, Train_loss:0.861522
    Step: 240, Train_loss:1.30292
    Step: 250, Train_loss:1.049
    Step: 260, Train_loss:2.16112
    Step: 270, Train_loss:1.69653
    Step: 280, Train_loss:1.14899
    Step: 290, Train_loss:2.219
    Step: 300, Train_loss:0.828941
    Step: 310, Train_loss:1.12873
    Step: 320, Train_loss:1.76599
    Step: 330, Train_loss:1.71529
    Step: 340, Train_loss:1.4581
    Step: 350, Train_loss:0.65748
    Step: 360, Train_loss:1.42326
    Step: 370, Train_loss:1.18391
    Step: 380, Train_loss:0.826132
    Step: 390, Train_loss:1.76418
    Step: 400, Train_loss:1.03422
    Step: 410, Train_loss:1.62564
    Step: 420, Train_loss:2.05213
    Step: 430, Train_loss:0.988207
    Step: 440, Train_loss:1.21138
    Step: 450, Train_loss:1.74969
    Step: 460, Train_loss:1.18206
    Step: 470, Train_loss:2.2452
    Step: 480, Train_loss:1.9146
    Step: 490, Train_loss:1.55511
    Step: 500, Train_loss:0.648539
    2018-08-12 21:04:42.452861 ---> Validation_loss: 1.28776
    Step: 510, Train_loss:1.54278
    Step: 520, Train_loss:2.89428
    Step: 530, Train_loss:0.672546
    Step: 540, Train_loss:3.83635
    Step: 550, Train_loss:0.471446
    Step: 560, Train_loss:1.15738
    Step: 570, Train_loss:1.40668
    Step: 580, Train_loss:1.15955
    Step: 590, Train_loss:1.75457
    Step: 600, Train_loss:1.45749
    Step: 610, Train_loss:1.19461
    Step: 620, Train_loss:0.903743
    Step: 630, Train_loss:0.80773
    Step: 640, Train_loss:1.29595
    Step: 650, Train_loss:1.25178
    Step: 660, Train_loss:1.51566
    Step: 670, Train_loss:0.998369
    Step: 680, Train_loss:1.39247
    Step: 690, Train_loss:2.10086
    Step: 700, Train_loss:1.62453
```


