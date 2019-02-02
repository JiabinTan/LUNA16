import tensorflow as tf
import numpy as np
import sys
sys.path.append('..//')
from Config.Config import Config as conf

'''
data first dim is depth
'''
def random_flip_FB(data,flip_prop=0.5):
    if flip_prop<np.random.uniform(low=0.0, high=1.0, size=None):
        return data
    else:
        if(np.size(data.shape)==3):
            return np.flip(data,0)
            pass
        else:
            return np.flip(data,1)
            pass
        pass

def random_rotate(data,rotate_prop):
    if flip_prop<np.random.uniform(low=0.0, high=1.0):
        return data
    else:
        K=np.random.randint(low=1, high=4)
        return tf.image.rot90(data,k)
def crop_resize(data,fraction_x=0.9,fraction_y=0.9,keep_prop=0.8,batch_size=conf.batch_size,crop_height=48, crop_width=48):
    if flip_prop>np.random.uniform(low=0.0, high=1.0):
        return data
    else:
        for i in batch_size:
            x1=flip_prop<np.random.uniform(low=0.0, high=1-fraction_x)
            x2=x1+fraction_x
            y1=flip_prop<np.random.uniform(low=0.0, high=1-fraction_y)
            y2=x1+fraction_y
            if i==0:
                box=[[y1, x1, y2, x2]]
                pass
            else:
                box+=[[y1, x1, y2, x2]]
                pass


        box_ind=[x for x in range(batch_size)]
        crop_size=[crop_height, crop_width]
        return tf.image.crop_and_resize(
                                image=data,
                                boxes=box,
                                box_ind=box_ind,
                                crop_size=crop_size
                                )

'''
数据增强
data:输入数据

返回增强后的数据
'''

def augmentate(data,is_gen=True):
    '''
    数据增强
    '''
    data=np.array(data)
    data=random_flip_FB(data,0.4)
    '''
    通道转换
    '''
    if(np.size(data.shape)==3):
        data=data.tranpose((1,2,0))
        pass
    else:
        data=data.transpose((0,2,3,1))
        pass
    '''
    上下左右翻转
    '''
    data=tf.image.random_flip_left_right(data)
    data=tf.image.random_flip_up_down(data)
    '''
    rotation 
    '''
    data=random_rotate(data)
    '''
    crop & resize
    '''
    if(is_gen):
        data=tf.image.crop_resize(data,batch_size=1)
        pass
    else:
        data=tf.image.crop_resize(data)
    return data





