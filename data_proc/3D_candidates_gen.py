#!/usr/bin/python2.6  
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('..//')
from Config.Config import Config as conf
import re
import reader_disp as reader
import numpy as np
import TFRecord_proc.TFRecord as tfrd
'''
获取3D训练数据

输入参量：
image：图像数据，整个数据
origin：原点
spacing：缩放因子
candidate：一系列选区，这边应该是个数组
ys:对应的label，也是个数组
z_pad:需要的z方向的大小
y_pad：需要的y方向的大小
x_pad:需要的x方向的大小

输出：

一个z_pad*y_pad*x_pad大小的数据块
'''
def get_3D_candidate(image,origin,spacing,candidates,ys,z_pad=36,y_pad=48,x_pad=48):
    for index,values in enumerate(candidates):
        #提取label
        y=ys[index]
        width,height,length=image.shape[0],image.shape[1],image.shape[2]
        z,y,x=values[0],values[1],values[2]
        z_index_min=max(0,z-z_pad/2)
        z_index_max=min(width-1,z+z_pad/2)

        y_index_min=max(0,y-y_pad/2)
        y_index_max=min(height-1,y+y_pad/2)
        
        x_index_min=max(0,x-x_pad/2)
        x_index_max=min(length-1,x+x_pad/2)

        data=np.zeros((z_pad,y_pad,x_pad),dtype=np.int16)
        #原图中截取候选区
        crop_cube=image[z_index_min:z_index_max,y_index_min:y_index_max,x_index_min:x_index_max]
        #判断是否只是截取了一部分，用0填充
        if(z_index_max-z_index_min<z_pad):
            if(z_index_min==0):
                data[:-(z_index_max-z_index_min),:,:]=crop_cube
            else:
                data[(z_index_max-z_index_min):,:,:]=crop_cube
                pass

        if(y_index_max-y_index_min<y_pad):
            if(y_index_min==0):
                data[:,:-(y_index_max-y_index_min),:]=crop_cube
            else:
                data[:,(y_index_max-y_index_min):,:]=crop_cube
                pass

        if(x_index_max-x_index_min<x_pad):
            if(x_index_min==0):
                data[:-(x_index_max-x_index_min),:,:]=crop_cube
            else:
                data[(x_index_max-x_index_min):,:,:]=crop_cube
                pass

        yield data,y

        
'''
生成候选区数据并保存

输入：输入subset文件夹的目录

输出：保存所有候选区的3d数据

'''
def main(file_dir):
    file_names_l1=[]#当前文件目录，level_1
    file_names_l1=os.listdir(file_dir)
    file_name_l1=[i for i in file_name_l1 if not re.match(r'^subset\d$',i)]
    csv_path=conf.scv_dir+'candidates.csv'
    candidates=reader.read_csv(csv_path)

    save_name='3D_data_'
    count=0
    if(not file_names_l1):
        print('no files')
        return
    for namas in file_names_l1:
        dir_path=file_dir+conf.separate+names   #当前文件目录
        file_name_l2=os.listdir(dir_path)#level_2
        if(not file_names_l2):
            print('no files')
            continue
        file_name_l2=[i for i in file_name_l2 if not re.match(r'.+raw$',i)]
        for i in file_name_l2:
            file_name=dir_path+conf.separate+i  #文件名
            image,origin,spacing=reader.load_image(file_name)
            CT_name=i[:-4]#得到CT实例名字
            cand_list=np.asarray(candidates.loc[candidates['seriesuid']==CT_name])
            cand_list=np.asarray([cand_list[:,3],cand_list[:,2],cand_list[:,1]],dtype=float).T
            class_y=cand_list[:,3]  #得到nodule的label
            #转化为voxel坐标
            voxel_coord=np.rint(reader.coord_convert(cand_list[:,:-1],origin,spacing)).astype(int)
            #得到3D图像数据，数据大小可以制定，默认36*48*48
            crop_data=get_3D_candidate(image,origin,spacing,voxel_coord,class_y)
            data_set=[]
            label_set=[]
            
            for data,label in crop_data:
                if((count%1000==0)):
                    if(count!=0):
                        '''
                        注意读取数据的时候为
                        set=np.load(path)
                        data=set[()][0]
                        label=set[()][1]
                        '''
                        #np.save(save_name+str(int(count/1000)),np.array([data_set,label_set]))

                        dir=save_name+str(int(count/1000))
                        tfrd.writer(data_set,label_set,dir)

                        data_set=np.array([data])
                        label_set=np.array([label])
                        pass

                    else:
                        data_set=np.array([data])
                        label_set=np.array([label])
                        pass

                else:
                    data_set=np.insert(data_set,1,data,axis=0)
                    label_set=np.stack(label_set,1,label,axis=0)
                    pass
                count+=1
                pass

            if((count%1001)==0):
                dir=save_name+str(int(count/1000)+1)
                tfrd.writer(data_set,label_set,dir)
                pass




if __name__=='__main__':
    main('')