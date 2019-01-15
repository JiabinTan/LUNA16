import SimpleITK as sitk
import numpy as np
#!/usr/bin/python2.6  
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import sys


'''python import模块时， 是在sys.path里按顺序查找的。
sys.path是一个列表，里面以字符串的形式存储了许多路径。
使用A.py文件中的函数需要先将他的文件路径放到sys.path中
'''
sys.path.append('..//')

from Config.Config import Config as conf

'''
读取图像相关数据

输入：
图像数据路径

输出：
图像数据、原点、缩放因子
'''
def load_image(filename):
    image=sitk.ReadImage(filename)
    numpy_image=sitk.GetArrayFromImage(image)
    numpy_origin=np.array(list(reversed(image.GetOrigin())))
    numpy_spacing=np.array(list(reversed(image.GetSpacing())))
    return numpy_image,numpy_origin,numpy_spacing
'''
读取候选区数据
输入：候选区文件路径
输出：pd的数据类型，可以直接使用pandas进行相关处理
'''
def read_csv(filename):
    lines=[]
    lines=pd.read_csv(filename)
    return lines
'''
坐标转换
输入：候选区坐标、原点、缩放因子
输出：对应的image数组中的index
'''
def coord_convert(worldcood,origin,spacing):
    stretched_voxel_coord=np.absolute(worldcood-origin)
    voxel_coord=stretched_voxel_coord/spacing
    return voxel_coord
#正规化CT图(范围0-1)
def normalize_planes(ct_image):
    maxHU=400#人体组织正常的HU应该是在这个范围之下
    minHU=-1000#空气的HU

    normalized_image=(ct_image-minHU)/(maxHU-minHU)
    normalized_image[normalized_image>1]=1
    normalized_image[normalized_image<0]=0
    return normalized_image
'''
这边是对2D来说的，把候选区的位置在图片上框出来
输入：图片数据，x，y坐标，框的半径、框的厚度
输出：加入框的图片数据
'''
def draw_box(data,y,x,radius=30,pad=2):
    data[max(0, y - radius):min(data.shape[0], y + radius),\
            max(0, x - radius - pad):max(0, x - radius)] = 3000
    data[max(0, y - radius):min(data.shape[0], y + radius),\
        min(data.shape[1], x + radius):min(data.shape[1], x + radius + pad)] = 3000
    data[max(0, y - radius - pad):max(0, y - radius),\
        max(0, x - radius):min(data.shape[1], x + radius)] = 3000
    data[min(data.shape[0], y + radius):min(data.shape[0], y + radius + pad),\
        max(0, x - radius):min(data.shape[1], x + radius)] = 3000  # 横线

    return data

#image_path=conf.CT_dir+'1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
#1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492 
image_path=conf.CT_dir+'1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd'
csv_path=conf.scv_dir+'candidates.csv'

image,origin,spacing=load_image(image_path)
#为了后面的batch处理，这边需要对origin、和spacing进行维度处理
origin=origin[np.newaxis]
spacing=spacing[np.newaxis]

print("=======image info=====")
print('size:',image.shape)
print('origin:',origin)
print('spacing:',spacing)

candidates=read_csv(csv_path)
#print('====candidates samples====')
#for i in range(conf.batch_size+1):
#    print(candidates[i])
#    pass

start=15647
#9313
#16051

cand=candidates.loc[15645:15654]
cand=np.asarray(cand)
world_coord=np.asarray([cand[:,3],cand[:,2],cand[:,1]],dtype=float).T
print(world_coord)
print(coord_convert(world_coord,origin,spacing))
voxel_coord=np.rint(coord_convert(world_coord,origin,spacing)).astype(int)


for i in range(0,conf.batch_size):
    #patch=image
    plt.clf()
    image_no_cut=np.copy(image[voxel_coord[i][0]])#避免引用传参
    new_image=draw_box(image_no_cut,voxel_coord[i][1],voxel_coord[i][2],radius=10,pad=2)
    plt.title(str(cand[i][4]))
    plt.imshow(new_image,cmap='gray')
    plt.show()

#numpyImage, numpyOrigin, numpySpacing = load_image(image_path)
#print(numpyImage.shape)
#print(numpyOrigin)
#print(numpySpacing)
#cands = read_csv(csv_path)
##这边要注意candidate的数据要跟我的读取的文件对应
#for cand in cands[9315:9317]:
#    worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
#    voxelCoord = np.rint(coord_convert(worldCoord, numpyOrigin, numpySpacing)).astype(int)
#    voxelWidth = 64
#    patch = numpyImage[voxelCoord[0],voxelCoord[1]-32:voxelCoord[1]+32,voxelCoord[2]-32:voxelCoord[2]+32]
#    patch = normalize_planes(patch)
    
#    print(worldCoord)
#    print(voxelCoord)
#    print(patch)
#    plt.imshow(patch, cmap='gray')
#    plt.show()

