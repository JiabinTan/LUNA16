import SimpleITK as sitk
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from PIL import Image

import sys


'''python import模块时， 是在sys.path里按顺序查找的。
sys.path是一个列表，里面以字符串的形式存储了许多路径。
使用A.py文件中的函数需要先将他的文件路径放到sys.path中
'''
sys.path.append('..//')

from Config.Config import Config as conf

def load_image(filename):
    image=sitk.ReadImage(filename)
    numpy_image=sitk.GetArrayFromImage(image)
    numpy_origin=np.array(list(reversed(image.GetOrigin())))
    numpy_spacing=np.array(list(reversed(image.GetSpacing())))
    return numpy_image,numpy_origin,numpy_spacing
def read_csv(filename):
    lines=[]
    with open(filename) as f:
        csvreader=csv.reader(f)
        for line in csvreader:
            lines.append(line)
            pass
        pass
    return lines
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



image_path=conf.CT_dir+'1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
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
cand=candidates[1:3]
cand=np.asarray(cand)
world_coord=np.asarray([cand[:,3],cand[:,2],cand[:,1]],dtype=float).T
voxel_coord=np.rint(coord_convert(world_coord,origin,spacing)).astype(int)


for k in range(9315,(9315+conf.batch_size)):
    i=k-9315
    patch=image[voxel_coord[i][0],voxel_coord[i][1]-60:voxel_coord[i][1]+60,voxel_coord[i][2]-60:voxel_coord[i][2]+60]
    plt.figure(i)
    plt.title('original image')
    plt.subplot(2,1,1)
    plt.imshow(image[voxel_coord[i][0]])
    plt.subplot(2,1,2)
    
    plt.title('shortcut image:'+str(cand[i][4]))
    plt.imshow(patch)
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

