
#!/usr/bin/python2.6  
# -*- coding: utf-8 -*-
'''
文档为数据读取的模板
'''
import sys
import tensorflow as tf
from tf.keras.models import Model
from dara_proc.TFRecord_proc import TFRecord as tfrd

from Config.Config import Config as conf
from tensorflow.python.ops import array_ops
import math
from tf.keras.callbacks import EarlyStopping
from extend.email import Email
from tf.keras.callbacks import LearningRateScheduler
from CNN import CNN_3d

from tf.keras.callbacks import ModelCheckpoint

'''
output messages to file
not display on screen
'''




'''
focal loss function
'''
def focal_loss(target_tensor,prediction_tensor,  weights=None, alpha=0.9, gamma=2):
    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)
'''
lr decay schedule
SWA schedule :https://github.com/timgaripov/swa/blob/master/train.py
swa IS abandoned for it's need two model

noisy_linear_cosine_decay : 
'''
def schedule(epoch):

    learning_rate = tf.train.cosine_decay_restarts(
                                                    learning_rate=0.001,
                                                    global_step=epoch,
                                                    first_decay_steps=400,
                                                    t_mul=1.5,
                                                    m_mul=0.8,
                                                    alpha=0.0,
                                                    name=None
                                                )
    return learning_rate


'''
日志记录
'''
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    #def on_batch_end(self, batch, logs={}):
    #    self.losses['batch'].append(logs.get('loss'))
    #    self.accuracy['batch'].append(logs.get('acc'))
    #    self.val_loss['batch'].append(logs.get('val_loss'))
    #    self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def log(self):
        bk=sys.stdout
        log_file = open("./logs/train_message.log", "w")
        sys.stdout = log_file
        iters = range(len(self.losses[loss_type]))
        for i in iters:
            print('====epoch '+str(i)+'=====')
            print('==train loss==')
            print(self.losses['epoch'][i])
            print('==train acc==')
            print(self.accuracy['epoch'][i])
            print('==val loss==')
            print(self.val_loss['epoch'][i])
            print('==val acc==')
            print(self.val_acc['epoch'][i])
        sys.stdout=bk
        
if __name__=='__main__':
    '''
    训练数据
    '''
    train_tfrecords_filename = ['test.tfrecords',] #训练集文件位置
    train_images,train_labels=tfrd.reader(train_tfrecords_filename,is_batch=True,batch_size=conf.batch_size)
    train_labels=tf.one_hot(train_labels,2)
    #train_dataset=tf.data.Dataset.zip((train_images, train_labels)) 
    '''
    测试集数据
    测试下，这两个数据集是不是有问题。我怕后面重新调用这个函数后，可能前面的数据集会出问题
    '''
    eval_tfrecords_filename = ['test.tfrecords',] #测试集文件位置
    eval_images,eval_labels=tfrd.reader(eval_tfrecords_filename,is_batch=True,batch_size=50)
    eval_labels=tf.one_hot(eval_labels,2)
    #eval_dataset=tf.data.Dataset.zip((eval_images, eval_labels))

    '''
    验证集数据
    测试下，这两个数据集是不是有问题。我怕后面重新调用这个函数后，可能前面的数据集会出问题
    '''
    val_tfrecords_filename = ['test.tfrecords',] #测试集文件位置
    val_images,val_labels=tfrd.reader(val_tfrecords_filename,is_batch=True,batch_size=50)
    val_labels=tf.one_hot(val_labels,2)
    #val_dataset=tf.data.Dataset.zip((val_images, val_labels))


    '''
    模型中需要的输入数据
    类似于占位
    '''

    xs = tf.keras.layers.Input(train_images)
    '''
    调用模型
    '''
   # import #这边会倒入你的文件

    #这边调用class
    CNN=CNN_3d(xs,is_training=True,keep_prob=0.9)
    #这边产生model、

    model=CNN.cnn()
    m#odel=mymodel_(xs,eval_labels)#这边需要一个model class，返回的是一个keras model
    if os.path.exists(conf.save_dir):
        model.load_weights(conf.save_dir)


    '''
    finish last part of the net
    '''
    model.compile(loss=focal_loss,
              optimizer=tf.train.AdamOptimizer,
              metrics=['accuracy'],
              target_tensors=train_labels)
    if conf.cos:
        lrate = LearningRateScheduler(schedule)
        pass
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)


    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    '''
    training 
    validation
    if you want learn about parameters , refer to official document . 
    validation is going at the end of every epoch
    now my parameters is 
    100 batched - 1 epoch
    batch_size - 32

    the number of valid batches - 20 because of 50(valid batch_size)*20(valid total batches)=1000(1 tfrecord file)
    '''
    history = LossHistory()
    
    checkpointer = ModelCheckpoint(filepath=conf.ckp_dir,
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='min',
                                    period=1)

    if conf.cos:
        model.fit(epochs=conf.epoch, steps_per_epoch=100,
            validation_data=(val_images,val_labels),validation_steps=20,
            callbacks=[lrate,early_stopping,history,checkpointer])
        pass
    else:
        model.fit(epochs=conf.epoch, steps_per_epoch=100,
            validation_data=(val_images,val_labels),validation_steps=20,
            callbacks=[early_stopping,history,checkpointer])
        pass
    history.log()
    '''
    last save the model
    I need to know what is saved.So, send me the h5 file
    '''
    


    model.save_weights(conf.save_dir)


    '''
    测试性能
    when training done , evaluate on test set(batch_size=50,total batches=20.I assume that we just choose 1 file with 1000 datas)
    '''
    xs = tf.keras.layers.Input(eval_images)
    #这边调用class
    CNN=CNN_3d(xs,is_training=False,keep_prob=0.9)
    #这边产生model、

    model=CNN.cnn()
    m#odel=mymodel_(xs,eval_labels)#这边需要一个model class，返回的是一个keras model
    if os.path.exists(conf.save_dir):
        model.load_weights(conf.save_dir)
        pass
    model.compile(loss=focal_loss,
              metrics=['accuracy'])
    test_loss=model.evaluate(eval_images,eval_labels, steps=20)
    
    bk=sys.stdout
    log_file = open("./logs/test_loss.log", "w")
    sys.stdout = log_file
    print('===test set loss===')
    print(test_loss)
    sys.stdout=bk

    coord.request_stop()
    coord.join(threads)
    '''
    traning done 
    send email to admin
    '''
    mail=Em()
    mail.send()
    del mail



    #with tf.Session() as sess: #开始一个会话
    #    #之前我做过测试，不初始化读取器也是可以运行的，应该是没有tf变量吧
    #    #这边还是初始化写者吧，当做模板，因为之后训练肯定是有变量的，需要初始化
    #    init_op = tf.initialize_all_variables()
    #    sess.run(init_op)
    #    coord=tf.train.Coordinator()
    #    threads= tf.train.start_queue_runners(coord=coord)
    #    try:
    #        #for i in range(math.ceil(conf.epoch*conf.data_size/conf.batch_size)):
    #        for i in range(4):
    #            data, l = sess.run([images,labels])#在会话中取出image和label
    #            model.fit

    #    except tf.errors.OutOfRangeError:
    #        print('Done Train...')
    #    finally:
    #        coord.request_stop()

        #coord.request_stop()
        #coord.join(threads)