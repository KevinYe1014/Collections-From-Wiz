```python
# -*- coding:UTF-8 -*-
import os
import sys

# 设置工程路径
curr_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.join(curr_path, "../../../")
sys.path.append(project_path)

# 确定训练使用的网络
net_name = 'inception_resnet_v1'
# net_name = 'mobienet_v1'
# net_name = 'densenet'
# 加载网络方法
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from cvpack import *
from importlib import import_module
from applications.face_age_gender.tensorflow_implementation.tf_age_gender_data_prepare import \
    age_gender_decode_tfrecord_to_data_label, age_label_conver

net = import_module('nets.tensorflow_nets.' + net_name)
logging = logger._getlogger('tf_age_gender_train')


class Model(net.NetAlgo):
    def train_private_args(self, parser):
        # 设定各个训练独有的参数
        private_train = parser.add_argument_group('Private', 'Private train parser')
        private_train.add_argument('--gender-class-num', type=int,
                                   help='gender_class_num')
        private_train.add_argument('--age-class-num', type=int,
                                   help='age_class_num')
        private_train.add_argument('--dropout-keep-prob', type=float,
                                   help='the drop out keep_prob of inception_resnet_v1')
        private_train.add_argument('--bottleneck-layer-flag', type=bool, default=False,
                                   help='bottleneck_layer_flag of inception_resnet_v1')
        private_train.add_argument('--bottleneck-layer-size', type=int, help='num of net size before full_connected')
        private_train.add_argument('--bn-decay', type=float, default=0.995, help='the bn_decay of inception_resnet_v1')
        private_train.add_argument('--bn-epsilon', type=float, default=0.001,
                                   help='the bn-epsilon of inception_resnet_v1')
        private_train.add_argument('--weight-decay', type=float, default=0.0,
                                   help='the weight-decay of inception_resnet_v1')
        private_train.add_argument('--data-train-tfrecords', type=str, help='train data tfrecords path')
        private_train.add_argument('--data-train-tfrecords-shape', help='train data img shape')
        private_train.add_argument('--data-train-tfrecord', type=str, help='train data tfrecord path')
        private_train.add_argument('--data-val-img', type=str, help='val data img path')
        private_train.add_argument('--data-val-tfrecord', type=str, help='val data tfrecord path')
        private_train.add_argument('--log-path', type=str, help='log path')
        return private_train

    def _do_parse_params(self, parameter):
        '''
        将参数转换成内部变量
        '''
        self.network = parameter.network
        self.mode_type = parameter.mode_type
        if self.mode_type == 'train':
            self.phase_train = True
            logging.info('Train Mode')
        else:
            self.phase_train = False
            logging.info('Test Mode')
        self.gender_class_num = parameter.gender_class_num
        self.age_class_num = parameter.age_class_num
        self.img_width = parameter.img_width
        self.img_height = parameter.img_height
        self.img_channel = parameter.img_channel
        self.img_shape = (self.img_height, self.img_width, self.img_channel)
        self.data_train_tfrecords = parameter.data_train_tfrecords
        self.data_train_tfrecords_shape = parameter.data_train_tfrecords_shape
        self.num_epochs = parameter.num_epochs
        self.lr = parameter.lr
        self.batch_size = parameter.batch_size
        self.dropout_keep_prob = parameter.dropout_keep_prob
        self.bn_decay = parameter.bn_decay
        self.bn_epsilon = parameter.bn_epsilon
        self.weight_decay = parameter.weight_decay
        self.bottleneck_layer_flag = parameter.bottleneck_layer_flag
        self.bottleneck_layer_size = parameter.bottleneck_layer_size
        self.fix_params = parameter.fix_params
        self.log_path = parameter.log_path
        self.gpus = parameter.gpus
        self.fine_tune = parameter.fine_tune
        self.fine_tune_model = parameter.fine_tune_model
        self.disp_batches = parameter.disp_batches
        self.snapshot_iters = parameter.snapshot_iters
        self.model_prefix = parameter.model_prefix

        if self.network == 'mobienet_v1':
            self.depth_multiplier = parameter.depth_multiplier

    # 输入
    def _do_inputs(self):
        inputs_placeholder = tf.placeholder(tf.float32, [None, self.img_width, self.img_height,
                                                         self.img_channel], 'inputs_placeholder')
        labels_placeholder = {}
        gender_labels_placeholder = tf.placeholder(tf.int32, [None, self.gender_class_num], 'gender_labels_placeholder')

        age_labels_placeholder = tf.placeholder(tf.int32, [None, self.age_class_num], 'age_labels_placeholder')

        labels_placeholder['gender'] = gender_labels_placeholder
        labels_placeholder['age'] = age_labels_placeholder

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train_placeholder')
        return inputs_placeholder, labels_placeholder, phase_train_placeholder

    # 最后分类层
    def _do_custom_net(self, base_net, end_points):
        gender_logits = slim.fully_connected(base_net, self.gender_class_num, activation_fn=None,
                                             weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             weights_regularizer=slim.l2_regularizer(1e-5),
                                             scope='gender_logits', reuse=False)

        age_logits = slim.fully_connected(base_net, self.age_class_num, activation_fn=None,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          weights_regularizer=slim.l2_regularizer(1e-5),
                                          scope='age_logits', reuse=False)

        logits = {}
        logits['gender'] = gender_logits
        logits['age'] = age_logits
        end_points['gender_logits'] = gender_logits
        end_points['age_logits'] = age_logits
        gender_softmax = tf.nn.softmax(gender_logits, name='gender_softmax')
        age_softmax = tf.nn.softmax(age_logits, name='age_softmax')
        end_points['gender_softmax'] = gender_softmax
        end_points['age_softmax'] = age_softmax
        return logits, end_points

    # loss 损失函数
    def _do_cost(self, logits, labels):
        logging.info('-------- Cost --------')
        gender_logits = logits['gender']
        gender_labels = labels['gender']
        age_logits = logits['age']
        age_labels = labels['age']
        gender_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=gender_labels,
                                                                       logits=gender_logits)
        gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)

        age_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=age_labels, logits=age_logits)
        age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)

        total_loss = tf.add_n(
            [gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        self.loss['gender_loss'] = gender_cross_entropy_mean
        self.loss['age_loss'] = age_cross_entropy_mean
        self.loss['total_loss'] = total_loss
        tf.summary.scalar('gender_loss', self.loss['gender_loss'])
        tf.summary.scalar('age_loss', self.loss['age_loss'])
        tf.summary.scalar('total_loss', self.loss['total_loss'])

    # 准确率计算
    def _do_evaluate(self, logits, labels):
        logging.info('-------- Evaluate --------')
        gender_logits = logits['gender']
        gender_labels = labels['gender']
        age_logits = logits['age']
        age_labels = labels['age']

        gender_correct_pred = tf.equal(tf.argmax(gender_logits, 1), tf.argmax(gender_labels, 1))
        self.accuracy['gender_accuracy'] = tf.reduce_mean(tf.cast(gender_correct_pred, tf.float32))

        age_correct_pred = tf.equal(tf.argmax(age_logits, 1), tf.argmax(age_labels, 1))
        self.accuracy['age_accuracy'] = tf.reduce_mean(tf.cast(age_correct_pred, tf.float32))

        self.accuracy['total_accuracy'] = (self.accuracy['gender_accuracy'] + self.accuracy['age_accuracy']) / 2

        tf.summary.scalar('gender_accuracy', self.accuracy['gender_accuracy'])
        tf.summary.scalar('age_accuracy', self.accuracy['age_accuracy'])
        tf.summary.scalar('total_accuracy', self.accuracy['total_accuracy'])

    # 优化器
    def _do_optimizer(self):
        logging.info('-------- Optimizer --------')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # 固定参数
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.fix_params is not None:
            # 指定固定多少参数
            self.train_vars = all_vars[:self.fix_params]
        else:
            self.train_vars = all_vars[:]

        decay_lr = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_steps=10000, decay_rate=0.9,
                                              staircase=True)
        tf.summary.scalar("lr", decay_lr)

        op_handel = tf.train.AdamOptimizer(learning_rate=decay_lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
        with tf.control_dependencies(update_ops):
            self.optimizer = op_handel.minimize(self.loss['total_loss'],
                                                global_step=self.global_step,
                                                var_list=self.train_vars)

    # 训练
    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus  # 使用 GPU
        num_classes_dir = {}
        num_classes_dir['gender'] = self.gender_class_num
        num_classes_dir['age'] = self.age_class_num

        train_img_data_batch, train_gender_batch, train_age_batch = age_gender_decode_tfrecord_to_data_label(
            [self.data_train_tfrecords],
            num_classes=num_classes_dir,
            feature_shape=self.img_shape,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            data_augmentation=True)

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        with tf.Session(config=tfconfig) as sess:
            self.summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_path, sess.graph)
            self.saver = tf.train.Saver(max_to_keep=3)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            # 看有没有模型需要重新加载,初始化参数
            if self.fine_tune:
                # 迁移学习
                logging.info('fine_tune restore')
                all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                #for v in all_vars:
                #     print(v.name)
                #print("-----------------------------------")
                restore_vars = all_vars[:]  # -2-4
                sess.run(init_op)
                saver = tf.train.Saver(restore_vars)
                saver.restore(sess, self.fine_tune_model)
            else:
                # num_files, files = self.find_previous(os.path.join(self.log_path, self.model_prefix))
                # if num_files == 0:
                #     logging.info('Init No Models')
                #     sess.run(tf.global_variables_initializer())
                # else:
                #     # 加载最后那个模型
                #     logging.info('Restore %s', files[-1])
                #     # 获取Iter
                #     _, start_step = files[-1].split('iter_')
                #     sess.run(tf.global_variables_initializer())
                #     self.saver.restore(sess, files[-1])
                #     tf.assign(self.global_step, int(start_step))

                ckpt = tf.train.get_checkpoint_state(self.log_path)
                if ckpt and ckpt.model_checkpoint_path:
                    logging.info('Restore %s', ckpt.model_checkpoint_path)
                    sess.run(init_op)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    logging.info('Init No Models')
                    sess.run(init_op)

            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                logging.info('start train...')
                while not coord.should_stop():
                    # 准备数据
                    batch_img_data, batch_gender_label, batch_age_label = sess.run(
                        [train_img_data_batch, train_gender_batch, train_age_batch])
                    print("batch_age_label", batch_age_label[0])
                    # 年龄转换
                    #batch_age_label = age_label_conver(batch_age_label)
                    # 训练
                    _, summary, age_loss, gender_loss, total_loss, acc, gender_acc, age_acc = sess.run(
                        [self.optimizer, self.summary_op, self.loss['age_loss'], self.loss['gender_loss'], self.loss['total_loss'], self.accuracy['total_accuracy'],
                         self.accuracy['gender_accuracy'], self.accuracy['age_accuracy']],
                        feed_dict={self.inputs_placeholder: batch_img_data,
                                   self.labels_placeholder['gender']: batch_gender_label,
                                   self.labels_placeholder['age']: batch_age_label,
                                   self.phase_train_placeholder: self.phase_train})
                    run_step = sess.run(self.global_step)
                    self.writer.add_summary(summary, run_step)
                    if run_step % self.disp_batches == 0:
                        logging.info("global_step " + str(run_step) + ", Age Loss= " + \
                                     "{:.6f}".format(age_loss) + ", Gender Loss= " + \
                                     "{:.6f}".format(gender_loss) + ", Total Loss= " + \
                                     "{:.6f}".format(total_loss) + ", Training Accuracy= " + \
                                     "{:.5f}".format(acc) + ", gen acc= " + "{:.5f}".format(
                            gender_acc) + ", age acc= " + "{:.5f}".format(age_acc))

                        val_acc = sess.run([self.accuracy['total_accuracy']],
                                           feed_dict={self.inputs_placeholder: batch_img_data,
                                                      self.labels_placeholder['gender']: batch_gender_label,
                                                      self.labels_placeholder['age']: batch_age_label,
                                                      self.phase_train_placeholder: False})
                        logging.info("val_acc " + "{:.5f}".format(val_acc[0]))

                        # print("global_step " + str(run_step) + ", Iter " + str(i) + "/" + str(
                        #     idx * self.config.TRAIN.BATCH_SIZE) + ", Minibatch Loss= " + \
                        #       "{:.6f}".format(loss_out) + ", Training Accuracy= " + \
                        #       "{:.5f}".format(acc))

                    # 保存临时模型
                    if run_step % self.snapshot_iters == 0:
                        # 每snapshot个n_epoch保存一下模型
                        self.saver.save(sess, os.path.join(self.log_path, self.model_prefix + '_iter_' + str(run_step)))

            except tf.errors.OutOfRangeError:
                # OutOfRangeError is thrown when epoch limit per
                # tf.train.limit_epochs is reached.
                logging.info('Stopping Training.')
            finally:
                # 最终保存下模型
                self.writer.close()
                self.saver.save(sess, os.path.join(self.log_path, self.model_prefix))
                coord.request_stop()

            coord.join(threads)


def train_age_gender():
    # 设定参数
    parser = argparse.ArgumentParser(description="train",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    common_model = Model(parser)

    # 设定确切的值
    parser.set_defaults(
        # public
        mode_type='train',  # 指定运行模式 train 、predict 、feature
        network=net_name,
        gender_class_num=2,
        age_class_num=101,
        img_width=171,  # 图片宽度
        img_height=171,  # 图片高度
        img_channel=3,  # 图片通道数
        num_epochs=100,  # 训练epoch 个数
        lr=0.005,  # 设置学习率
        bn_decay=0.9,
        batch_size=160,  # 设定batch size
        dropout_keep_prob=0.8,
        bottleneck_layer_flag=True,
        bottleneck_layer_size=128,
        fix_params=None,
        fine_tune=True,
        fine_tune_model=r'../../../models/age_gender_fit_inception_resnet_v1/face_age_gender_inception_resnet_v1_iter_380000',
        # data_train_tfrecords=r'/lianlian/data/face_age_gender/age_gender_face_train.tfrecords',
        # data_train_tfrecords_shape=(171, 171, 3),
        # big
        # data_train_tfrecords=r'../../../image_data/age_gender_face_big_train.tfrecords',
        # data_train_tfrecords_shape=(171, 171, 3),
        # log_path=r'../../../models/age_gender_big_' + net_name,
        # gpus='0',
        # fit
        data_train_tfrecords=r'E:\data\age_gender_face_big_train.tfrecords',
        data_train_tfrecords_shape=(171, 171, 3),
        log_path=r'../../../models/age_gender_fit_' + net_name,
        gpus='1',
        disp_batches=100,  # 显示logging信息
        snapshot_iters=1000,  # 保存临时模型
        model_prefix='face_age_gender_' + net_name
    )
    # 生成参数
    args = parser.parse_args()
    with tf.Graph().as_default():
        common_model.build_graph(args)
        common_model.build_train_op()
        common_model.train()


def pack_pb_model():
    from cvpack.tensorflow_core.tf_pack import pack_db_model
    # model_file = r'C:\Users\chenxin.ZHIQUPLUS\Desktop\t\face_age_gender_mobienet_v1_iter_2000'
    model_file = r'C:\Users\maoying\Desktop\work1\algo_cv\models\face_age_gender_inception_resnet_v1_iter_380000'
    pack_handle = pack_db_model()
    pack_handle.transform_to_db(model_file,
                                [net_name + "/gender_logits/BiasAdd", net_name + "/age_logits/BiasAdd"])


if __name__ == '__main__':
    train_age_gender()
    # pack_pb_model()
```

