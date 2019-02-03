#!/usr/bin/python2.6  
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('..//')
from Config.Config import Config as conf
import re

from data_proc import reader_disp as reader
import numpy as np
from TFRecord_proc import TFRecord as tfrd
from data_aug import augmentate

from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi

import matplotlib.pyplot as plt




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
def get_3D_candidate(image, origin, spacing, candidates, ys, z_pad=36, y_pad=48, x_pad=48):
    for index, values in enumerate(candidates):
        # 提取label
        label = ys[index]
        width, height, length = image.shape[0], image.shape[1], image.shape[2]
        z, y, x = values[0], values[1], values[2]
        z_index_min = int(max(0, z - z_pad / 2))
        z_index_max = int(min(width - 1, z + z_pad / 2))

        y_index_min = int(max(0, y - y_pad / 2))
        y_index_max = int(min(height - 1, y + y_pad / 2))

        x_index_min = int(max(0, x - x_pad / 2))
        x_index_max = int(min(length - 1, x + x_pad / 2))

        data = np.zeros((z_pad, y_pad, x_pad), dtype=np.int16)
        # 原图中截取候选区
        crop_cube = image[z_index_min:z_index_max, y_index_min:y_index_max, x_index_min:x_index_max]
        # 判断是否只是截取了一部分，用0填充
        z_flag = (z_index_max - z_index_min) == 36  # 判断是否缺少
        y_flag = (y_index_max - y_index_min) == 48
        x_flag = (x_index_max - x_index_min) == 48

        # if z_flag and y_flag and x_flag:
        #     # z ok y ok x ok
        #     data[:, :, :] = crop_cube
        #     yield data, label
        if ((z_index_min == 0) and (not z_flag)):
            if ((y_index_min == 0) and (not y_flag)):
                if ((x_index_min == 0) and (not x_flag)):
                    # z front y front x front
                    data[:-z_pad + (z_index_max - z_index_min), :-y_pad + (y_index_max - y_index_min),
                    :-x_pad + (x_index_max - x_index_min)] = crop_cube
                    pass
                elif ((x_index_max == length - 1) and (not x_flag)):
                    # z front y front x back
                    data[:-z_pad + (z_index_max - z_index_min), :-y_pad + (y_index_max - y_index_min),
                    x_pad - (x_index_max - x_index_min):] = crop_cube
                    pass
                else:
                    # z front y front x ok
                    data[:-z_pad + (z_index_max - z_index_min), :-y_pad + (y_index_max - y_index_min), :] = crop_cube
                    pass
                pass
            elif ((y_index_max == height - 1) and (not y_flag)):
                if ((x_index_min == 0) and (not x_flag)):
                    # z front y back x front
                    data[:-z_pad + (z_index_max - z_index_min), y_pad - (y_index_max - y_index_min):,
                    :-x_pad + (x_index_max - x_index_min)] = crop_cube
                    pass
                elif ((x_index_max == length - 1) and (not x_flag)):
                    # z front y back x back
                    data[:-z_pad + (z_index_max - z_index_min), y_pad - (y_index_max - y_index_min):,
                    x_pad - (x_index_max - x_index_min):] = crop_cube
                    pass
                else:
                    # z front y back x ok
                    data[:-z_pad + (z_index_max - z_index_min), y_pad - (y_index_max - y_index_min):, :] = crop_cube
                    pass
                pass
            else:
                if ((x_index_min == 0) and (not x_flag)):
                    # z front y ok x front
                    data[:-z_pad + (z_index_max - z_index_min), :, :-x_pad + (x_index_max - x_index_min)] = crop_cube
                    pass
                elif ((x_index_max == length - 1) and (not x_flag)):
                    # z front y ok x back
                    data[:-z_pad + (z_index_max - z_index_min), :, x_pad - (x_index_max - x_index_min):] = crop_cube
                    pass
                else:
                    # z front y ok x ok
                    data[:-z_pad + (z_index_max - z_index_min), :, :] = crop_cube
                    pass
                pass
            pass
        elif ((z_index_max == width - 1) and not z_flag):
            if ((y_index_min == 0) and not y_flag):
                if ((x_index_min == 0) and not x_flag):
                    # z back y front x front
                    data[z_pad - (z_index_max - z_index_min):, :-y_pad + (y_index_max - y_index_min),
                    :-x_pad + (x_index_max - x_index_min)] = crop_cube
                    pass
                elif ((x_index_max == length - 1) and not x_flag):
                    # z back y front x back
                    data[z_pad - (z_index_max - z_index_min):, :-y_pad + (y_index_max - y_index_min),
                    x_pad - (x_index_max - x_index_min):] = crop_cube
                    pass
                else:
                    # z back y front x ok
                    data[z_pad - (z_index_max - z_index_min):, :-y_pad + (y_index_max - y_index_min), :] = crop_cube
                    pass
                pass
            elif ((y_index_max == height - 1) and not y_flag):
                if ((x_index_min == 0) and not x_flag):
                    # z back y back x front
                    data[z_pad - (z_index_max - z_index_min):, y_pad - (y_index_max - y_index_min):,
                    :-x_pad + (x_index_max - x_index_min)] = crop_cube
                    pass
                elif ((x_index_max == length - 1) and not x_flag):
                    # z back y back x back
                    data[z_pad - (z_index_max - z_index_min):, y_pad - (y_index_max - y_index_min):,
                    x_pad - (x_index_max - x_index_min):] = crop_cube
                    pass
                else:
                    # z back y back x ok
                    data[z_pad - (z_index_max - z_index_min):, y_pad - (y_index_max - y_index_min):, :] = crop_cube
                    pass
                pass
            else:
                if ((x_index_min == 0) and not x_flag):
                    # z back y ok x front
                    data[z_pad - (z_index_max - z_index_min):, :, :-x_pad + (x_index_max - x_index_min)] = crop_cube
                    pass
                elif ((x_index_max == length - 1) and not x_flag):
                    # z back y ok x back
                    data[z_pad - (z_index_max - z_index_min):, :, x_pad - (x_index_max - x_index_min):] = crop_cube
                    pass
                else:
                    # z back y ok x ok
                    data[z_pad - (z_index_max - z_index_min):, :, :] = crop_cube
                    pass
                pass
            pass
        else:
            if ((y_index_min == 0) and not y_flag):
                if ((x_index_min == 0) and not x_flag):
                    # z ok y front x front
                    data[:, :-y_pad + (y_index_max - y_index_min), :-x_pad + (x_index_max - x_index_min)] = crop_cube
                    pass
                elif ((x_index_max == length - 1) and not x_flag):
                    # z ok y front x back
                    data[:, :-y_pad + (y_index_max - y_index_min), x_pad - (x_index_max - x_index_min):] = crop_cube
                    pass
                else:
                    # z ok y front x ok
                    data[:, :-y_pad + (y_index_max - y_index_min), :] = crop_cube
                    pass
                pass
            elif ((y_index_max == height - 1) and not y_flag):  # z前面+y后面
                if ((x_index_min == 0) and not x_flag):
                    # z ok y back x front
                    data[:, y_pad - (y_index_max - y_index_min):, :-x_pad + (x_index_max - x_index_min)] = crop_cube
                    pass
                elif ((x_index_max == length - 1) and not x_flag):
                    # z ok y back x back
                    data[:, y_pad - (y_index_max - y_index_min):, x_pad - (x_index_max - x_index_min):] = crop_cube
                    pass
                else:
                    # z ok y back x ok
                    data[:, y_pad - (y_index_max - y_index_min):, :] = crop_cube
                    pass
                pass
            else:
                if ((x_index_min == 0) and not x_flag):
                    # z ok y ok x front
                    data[:, :, :-x_pad + (x_index_max - x_index_min)] = crop_cube
                    pass
                elif ((x_index_max == length - 1) and not x_flag):
                    # z ok y ok x back
                    data[:, :, x_pad - (x_index_max - x_index_min):] = crop_cube
                    pass
                else:
                    # z ok y ok x ok
                    data[:, :, :] = crop_cube
                    pass
                pass
            pass

        yield data, label


'''
把CT图像进行处理，只保留肺部图像
输入：
ct_scan:输入的CT图像组，三维
输出：
处理完后的CT图像组，三维
'''
def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

'''
输入：
im:输入的CT图像，二维
plot：是否显示，不需要显示，最后直接3d画出来测试下就好
输出：
处理完后的CT图像，二维
'''
def get_segmented_lungs(im, plot=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    #im[im == -2000] = np.int16(0)
    #Step 1: Convert into a binary image. 
    binary = im < -300
    if plot == True:
        #np.save('im',im)
        plt.imshow(im,cmap='gray')
        plt.show()
        plt.imshow(binary,cmap='gray')
        plt.show()
        
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plt.imshow(cleared,cmap='gray')
        plt.show()
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plt.imshow(label_image,cmap='gray')
        plt.show()

    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plt.imshow(binary,cmap='gray')
        plt.show()
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plt.imshow(binary,cmap='gray')
        plt.show()
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(20)
    binary = binary_closing(binary, selem)
    if plot == True:
        plt.imshow(binary,cmap='gray')
        plt.show()
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plt.imshow(binary,cmap='gray')
        plt.show()
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = np.int16(-1200)
    im=np.clip(im,-1200,600)
    if plot == True:
        plt.imshow(im,cmap='gray')
        plt.show()
        #plt.show()
    return im

        
'''
生成候选区数据并保存

输入：输入subset文件夹的目录

输出：保存所有候选区的3d数据

'''
def main(file_dir):
    file_names_l1=[]#当前文件目录，level_1
    file_names_l1=os.listdir(file_dir)
    # print("sl1:",file_names_l1)
    file_names_l1=[i for i in file_names_l1 if re.match(r'^subset\d$',i)]
    # print("nl1:",file_name_l1)
    csv_path=conf.scv_dir+'candidates.csv'
    candidates=reader.read_csv(csv_path)

    data_set=[]
    label_set=[]
    count=0
    if(not file_names_l1):
        print('no files')
        return
    for names in file_names_l1:
        dir_path=file_dir+conf.separate+names   #当前文件目录
        file_names_l2=os.listdir(dir_path)#level_2
        if(not file_names_l2):
            print('no files')
            continue
        file_names_l2=[i for i in file_names_l2 if not re.match(r'.+raw$',i)]
        for i in file_names_l2:
            file_name=dir_path+conf.separate+i  #文件名
            image,origin,spacing=reader.load_image(file_name)

            image=segment_lung_from_ct_scan(image)  #产生mask提取肺部，删除杂质信息（空气、脊柱等）

            CT_name=i[:-4]#得到CT实例名字

            cand_list_temp=np.asarray(candidates.loc[candidates['seriesuid']==CT_name])
            # print("c1:",cand_list)
            cand_list=np.asarray([cand_list_temp[:,3],cand_list_temp[:,2],cand_list_temp[:,1]],dtype=float).T
            # print("c2:",cand_list)
            # class_y=candidates[:,3]  #得到nodule的label
            class_y = cand_list_temp[:,4]

            #转化为voxel坐标
            voxel_coord=np.rint(reader.coord_convert(cand_list,origin,spacing)).astype(int)
            #得到3D图像数据，数据大小可以制定，默认36*48*48
            crop_data=get_3D_candidate(image,origin,spacing,voxel_coord,class_y)
            
            
            save_name='3D_data_'#保存路径加文件名，如果没有路径，保存到当前文件夹下，这边为了方便直接保存在当前文件夹了
            is_test = True #是否测试
            if not is_test:
                for data,label in crop_data:
                    if label==1:
                        for i in range(100):
                            
                            data=data[np.newaxis,:]
                            data=(data+1200)/1800
                            #转化为float32 可以加速后面的增强速度
                            data=data.astype(np.float32)
                            data=augmentate(data)
                            data=data*1800-1200
                            data=data.astype(np.int16)

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
                                pass
                            else:
                                data_set=np.insert(data_set,1,data,axis=0)
                                label_set=np.insert(label_set,1,label,axis=0)
                                pass
                            count+=1
                            pass
                        pass                        
                    else:
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
                            label_set=np.insert(label_set,1,label,axis=0)
                            pass
                        count+=1
                        pass
                    pass
                pass
            else:
                x=0
                for data,label in crop_data:
                    if (x==0):
                        data_set=np.array([data])
                        label_set=np.array([label])
                    else:
                        data_set=np.insert(data_set,1,data,axis=0)
                        label_set=np.insert(label_set,1,label,axis=0)
                    if(x==2):
                        break   #只测试3个数据
                    x+=1
                    pass
                print(data_set.shape)
                print(label_set.shape)
                np.save('3d_data',data_set)
                np.save('label',label_set)    
                pass
            if is_test:
                break
            pass
        pass
    dir=save_name+str(int(count/1000)+1)
    tfrd.writer(data_set,label_set,dir)
 

if __name__=='__main__':
    # print(os.listdir(sys.path.append('..//')))
    main(conf.data_dir)
