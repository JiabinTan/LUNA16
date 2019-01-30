#!/usr/bin/python2.6  
# -*- coding: utf-8 -*-
'''
文件为配置文件
'''
class Config(object):
    '''
    整个类为一个配置类
    '''

    '''
    全局路径
    '''
    data_dir = 'G:\\luna16\\'           #总数据的目录
    scv_dir='G:\\luna16\\CSVFILES\\'    #csv文件的目录
    CT_dir='G:\\luna16\\subset0\\'      #CT图像subset0的目录
    tfrecord_path=''                    #tfrecord文件路径


    '''
    训练参数
    '''
    batch_size=32                       #训练时候的batch_size的大小
    data_size=754975                    #V2版本中提供了754，975条训练数据，V1提供了551，065条，这边默认使用V2，毕竟数据比较多
    epoch=250                           #训练的轮数，体现在我们训练的for循环次数为  epoch*data_size/batch_size 
    save_dir='..\\model_save\\my_model_weights.h5'
    #SWA = False                         #是否使用SWA lr decay schedule
    cos = False                         #是否使用 cosine_decay_restarts ：https://arxiv.org/abs/1608.03983
    '''
    其他
    '''
    separate='\\'                       #文件路径的分隔符，windows \\ linux\
    
                                        #发送人必须是QQ邮箱账号
    username = 'komo.tan@foxmail.com'   #发送人邮箱账号
    password='mvvimofoqhtogege'         #发送人邮箱密码

    email_list=receiver=['1035844563@qq.com',] 
                                        #收件人邮箱，163邮箱之类的都可

    pass




