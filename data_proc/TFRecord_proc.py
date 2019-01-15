#!/usr/bin/python2.6  
# -*- coding: utf-8 -*-
import tensorflow as tf
'''
读写tfrecord文件
'''

class TFRecord(object):
    '''
    写入tfrecord数据
    
    输入：
    data数组，
    label数组
    dir保存路径
    
    输出：
    保存文件
    '''
    def writer(data,label,dir):
        writer = tf.python_io.TFRecordWriter(dir)
        img_raw = data.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))

        writer.write(example.SerializeToString())  #序列化为字符串
        writer.close()

        pass
    '''
    tfrecord读取器
    输入：
    tfrecords_filename 需要读取的文件名list
    is_batch=False 是否是batch输出，测试的时候输出一个就好方便一点，最好记录下单个输出跟多个输出的不同，比如输出数据在维度上的区别
    is_shuffle=False 是否打乱，训练的时候打乱，valid跟test的时候不打乱
    batch_size=32 batch的大小
    z_size=36 z的大小
    y_size=48  y的大小
    x_size=48 x的大小
    '''
    def reader(tfrecords_filename,is_batch=False,is_shuffle=False,batch_size=32,z_size=36,y_size=48,x_size=48):
        filename_queue = tf.train.string_input_producer([tfrecords_filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })

        img = tf.decode_raw(features['img_raw'], tf.int16)
        img = tf.reshape(img, [z_size, y_size, x_size])
        label = tf.cast(features['label'], tf.int64)
        if (is_batch):
            if (is_shuffle):
                img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                num_threads=5,
                                                batch_size=batch_size, 
                                                capacity=3000,
                                                min_after_dequeue=1000)
            else:
                img_batch, label_batch = tf.train.batch([img,label],
                                                batch_size=batch_size,
                                                num_threads=5,
                                                capacity=3000
                                                )
                pass
        else:
            return img,label
        return img_batch, label_batch
    pass




