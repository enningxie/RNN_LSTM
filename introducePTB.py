# coding=utf-8
import tensorflow as tf
import reader
import os


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# 读取数据并打印长度及前100位数据
DATA_PATH = "./PTB_data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print(len(train_data))
print(train_data[:100])


# 将训练数据组织成batch大小为4、截断长度为5的数据组。并使用队列读取前3个batch。

result = reader.ptb_producer(train_data, 4, 5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        x, y = sess.run(result)
        print("X%d:" % i, x)
        print("Y%d:" % i, y)
    coord.request_stop()
    coord.join(threads)