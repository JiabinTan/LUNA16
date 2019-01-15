#!/usr/bin/python2.6  
# -*- coding: utf-8 -*-
'''
文件为配置文件
'''
class Config(object):
    separate='\\'                   #文件路径的分隔符，windows \\ linux\
    data_dir = 'G:\\luna16\\'       #总数据的目录
    scv_dir='G:\\luna16\\CSVFILES\\'#csv文件的目录
    CT_dir='G:\\luna16\\subset0\\'  #CT图像subset0的目录
    batch_size=10                   #训练时候的batch_size的大小
    tfrecord_path=''                #tfrecord文件路径
    data_size=754975                #V2版本中提供了754，975条训练数据，V1提供了551，065条，这边默认使用V2，毕竟数据比较多
    epoch=2                        #训练的轮数，体现在我们训练的for循环次数为  epoch*data_size/batch_size 
    pass




