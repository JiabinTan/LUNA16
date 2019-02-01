#1/27 : 2 PM
#import numpy as np
#from skimage.segmentation import clear_border
#from skimage.measure import label,regionprops, perimeter
#from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
#from skimage.filters import roberts, sobel
#from scipy import ndimage as ndi

#import matplotlib.pyplot as plt

#def segment_lung_from_ct_scan(ct_scan):
#    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

#'''
#输入：
#im:输入的CT图像，二维
#plot：是否显示，不需要显示，最后直接3d画出来测试下就好
#输出：
#处理完后的CT图像，二维
#'''
#def get_segmented_lungs(im, plot=False):
    
#    '''
#    This funtion segments the lungs from the given 2D slice.
#    '''
#    #im[im < -1000] = np.int16(-1000)
#    #Step 1: Convert into a binary image. 
#    binary = im < -300
#    plt.imshow(im,cmap='gray')
#    plt.show()
#    if plot == True:
#        #np.save('im',im)
#        plt.imshow(im,cmap=plt.cm.bone)
#        plt.show()
#        plt.imshow(binary,cmap=plt.cm.bone)
#        plt.show()
        
#    '''
#    Step 2: Remove the blobs connected to the border of the image.
#    '''
#    cleared = clear_border(binary)
#    if plot == True:
#        plt.imshow(cleared,cmap='gray')
#        plt.show()
#    '''
#    Step 3: Label the image.
#    '''
#    label_image = label(cleared)
#    if plot == True:
#        plt.imshow(label_image,cmap='gray')
#        plt.show()

#    '''
#    Step 4: Keep the labels with 2 largest areas.
#    '''
#    areas = [r.area for r in regionprops(label_image)]
#    areas.sort()
#    if len(areas) > 2:
#        for region in regionprops(label_image):
#            if region.area < areas[-2]:
#                for coordinates in region.coords:                
#                       label_image[coordinates[0], coordinates[1]] = 0
#    binary = label_image > 0
#    if plot == True:
#        plt.imshow(binary,cmap='gray')
#        plt.show()
#    '''
#    Step 5: Erosion operation with a disk of radius 2. This operation is 
#    seperate the lung nodules attached to the blood vessels.
#    '''
#    selem = disk(2)
#    binary = binary_erosion(binary, selem)
#    if plot == True:
#        plt.imshow(binary,cmap='gray')
#        plt.show()
#    '''
#    Step 6: Closure operation with a disk of radius 10. This operation is 
#    to keep nodules attached to the lung wall.
#    '''
#    selem = disk(20)
#    binary = binary_closing(binary, selem)
#    if plot == True:
#        plt.imshow(binary,cmap='gray')
#        plt.show()
#    '''
#    Step 7: Fill in the small holes inside the binary mask of lungs.
#    '''
#    edges = roberts(binary)
#    binary = ndi.binary_fill_holes(edges)
#    if plot == True:
#        plt.imshow(binary,cmap='gray')
#        plt.show()
#    '''
#    Step 8: Superimpose the binary mask on the input image.
#    '''
#    get_high_vals = binary == 0
#    im[get_high_vals] = np.int16(0)
#    plt.imshow(im,cmap='gray')
#    plt.show()
#    if plot == True:
#        plt.imshow(im,cmap='gray')
#        plt.show()
#        #plt.show()
#    return im
#def largest_label_volume(im, bg=-1):
#    vals, counts = np.unique(im, return_counts=True)

#    counts = counts[vals != bg]
#    vals = vals[vals != bg]

#    if len(counts) > 0:
#        return vals[np.argmax(counts)]
#    else:
#        return None

#def segment_lung_mask(image, fill_lung_structures=True):
    
#    # not actually binary, but 1 and 2. 
#    # 0 is treated as background, which we do not want
#    binary_image = np.array(image > -320, dtype=np.int8)+1
#    plt.imshow(binary_image,cmap='gray')
#    plt.show()
#    labels = measure.label(binary_image)
#    plt.imshow(labels,cmap='gray')
#    plt.show()
#    # Pick the pixel in the very corner to determine which label is air.
#    #   Improvement: Pick multiple background labels from around the patient
#    #   More resistant to "trays" on which the patient lays cutting the air 
#    #   around the person in half
#    background_label = labels[0,0,0]
    
#    #Fill the air around the person
#    binary_image[background_label == labels] = 2
    
    
#    # Method of filling the lung structures (that is superior to something like 
#    # morphological closing)
#    if fill_lung_structures:
#        # For every slice we determine the largest solid structure
#        for i, axial_slice in enumerate(binary_image):
#            axial_slice = axial_slice - 1
#            labeling = measure.label(axial_slice)
#            l_max = largest_label_volume(labeling, bg=0)
            
#            if l_max is not None: #This slice contains some lung
#                binary_image[i][labeling != l_max] = 1

    
#    binary_image -= 1 #Make the image actual binary
#    binary_image = 1-binary_image # Invert it, lungs are now 1
    
#    # Remove other air pockets insided body
#    labels = measure.label(binary_image, background=0)
#    l_max = largest_label_volume(labels, bg=0)
#    if l_max is not None: # There are air pockets
#        binary_image[labels != l_max] = 0
 
#    return binary_image
#scan=np.load('im.npy')
#plt.imshow(scan[60])
#plt.show()
#segment_lung_from_ct_scan(scan[14:])
##segmented_lungs = segment_lung_mask(scan, False)
'''
1/27 : 2 PM
'''
##from TFRecord_proc import TFRecord as tfrd
#import numpy as np
#data_set=np.load('3d_data.npy')
#label_set=np.load('label.npy')
#print(data_set.shape)
#print(label_set.shape)
#print(data_set.dtype)
##dir='test.tfrecords'
##tfrd.writer(data_set,label_set,dir)
'''
1/27 : 10 PM
'''
#def func():
#    for i in range(5):
#        yield i
#for i in func():
#    print(i)
'''
1/2/2019
'''
# -*- coding:utf-8 -*-


