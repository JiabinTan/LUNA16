#!/usr/bin/python2.6  
# -*- coding: utf-8 -*-
'''
文档为数据读取的模板
'''
import TFRecord_proc.TFRecord as tfrd
sys.path.append('..//')
from Config.Config import Config as conf


if __name__=='__main__':
    #这边是我随便写的，可以使用绝对路径或者相对路径，训练的时候把所有文件位置写进去
    #测试的时候写一个差不多了，能运行就好
    tfrecords_filename = ['1.tfrecord','2.tfrecord']
    #d单个读取，随便测试下batch读取，记录下两者的差别，特别是shape，也就是维度的差别
    #记录下来，避免因为这些小细节导致后面代码大批量的重写
    images,labels=tfrd.reader(tfrecords_filename)

    with tf.Session() as sess: #开始一个会话
        #之前我做过测试，不初始化读取器也是可以运行的，应该是没有tf变量吧
        #这边还是初始化写者吧，当做模板，因为之后训练肯定是有变量的，需要初始化
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        try:
            for i in range(conf.epoch*conf.data_size/conf.batch_size):
                example, l = sess.run([image,label])#在会话中取出image和label
                '''
                这边就可以对图片数据跟label数据进行处理了，比如测试阶段的查看3D图
                训练阶段的feed data操作
                    '''
        except tf.errors.OutOfRangeError:
            print('Done Train...')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)