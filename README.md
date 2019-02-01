# LUNA16

LUNA16 competition repository
结构说明：<br/>

~~~
/  根目录
├─Config                      配置目录
│  └─Confog.py                配置文件
│
├─data_proc                   数据处理目录
│  ├─reader_disp.py           里面主要是图像读取跟csv数据读取以及显示，主要目的是为了熟悉数据，其中的一些函数也会用于网络的数据处理部分
│  ├─data_read_demo.py        tfrecord文件读取demo
│  ├─TFRecord_proc.py         TFreord文件处理（读写）
│  ├─test.py                  测试一些模块的测试文件，因为模块我这边不能直接运行
│  └─3D_candidates_gen.py	  3D训练数据生成物文件
│
├─extend
│  ├─3Dplot                   画3D图像
│  └─email.py                 邮件发送模块（为了简化配置文件，发送内容之类的已经写死，如果需要我再改）
├─cpk                         checkpointer 文件保存位置
├─log                         训练日志保存位置
├─model_save                  模型保存位置
├─CNN.py                      网络结构文件
└─train.py                    训练驱动文件，类似于训练入口文件
~~~
