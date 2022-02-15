from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import imgaug as ia
from imgaug import augmenters as iaa

type="train_add"
global_pre_file="dataset/generate_word/"

# 获取轨迹边界
def getboundary(nz):
    boundary_list = []
    boundary_num = 0

    for i, x in enumerate(nz[:-1]):
        # 寻找边界i
        if (nz[i] == boundary_num and nz[i + 1] != boundary_num) or nz[i] != boundary_num and nz[i + 1] == boundary_num:
            boundary_list.append(i + 1)


    return min(boundary_list),max(boundary_list)

# 对轨迹进行定位与裁剪
def locate_crop(rawimg, target=None):
    grayscaleimg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2GRAY)
    grayscaleimg = grayscaleimg - int(np.mean(grayscaleimg))
    grayscaleimg[grayscaleimg < 50] = 0
    # grayscaleimg[grayscaleimg > 150] =255

    # counting non-zero value by row , axis y
    # 可以得到字符高的边界
    row_nz = []
    for row in grayscaleimg.tolist():
        row_nz.append(len(row) - row.count(0))

    # print(row_nz)
    col_nz = []
    for col in grayscaleimg.T.tolist():
        col_nz.append(len(col) - col.count(0))

    uy, ly = getboundary(row_nz)
    # print(upper_y,lower_y,uy,ly)

    col_left, col_right = getboundary(col_nz)
    #print(col_left,col_right)
    final = grayscaleimg[uy:ly, col_left:col_right]
    final[final != 0] = 255
    if (target != None):
        cv2.imwrite(target, final)
    return final


def get_all_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files  # 当前路径下所有非目录子文件

# 对pre_proccess下的所有图片做二值化处理，并根据定位裁剪成正方形
def source_proccess(global_pre_file):
    pre_path=global_pre_file+"_add_pprocess/"
    files_list=get_all_file_name(pre_path)
    #print(files)
    file_path=global_pre_file+"_source/"

    for file in files_list:
        source_path=pre_path+file
        target_path=file_path+file
        locate_crop(source_path,target_path)
        print(file+" is ok")


# 从train_source 文件夹中读取处理过的图像，并进行数据增强,增强倍数为num*epoch
def data_aug(global_pre_file,num):
    source_path=global_pre_file+"_orig/"
    target_path=global_pre_file+"/"

    files_list=get_all_file_name(source_path)
    images_list=[]
    for file in files_list:
        print(file+" is being proccessed")
        # a,b,c,d=locate_crop(source_path+file,None)

        images_list=[cv2.imread(source_path+file) for _ in range((num))]
        seq=iaa.Sequential([
            iaa.Affine(
                # # 缩放变换
                #scale={"x": (0.8, 1.3), "y": (0.8, 1.3)},
                # 旋转
                rotate=(-3, 8),
                cval=255
            )
        ],random_order=True)

        images_aug_list=seq.augment_images(images_list)
        #ia.imshow(ia.draw_grid(images_aug_list, cols=4, rows=2))

        for i in range(0,len(images_aug_list)):
            tmp=images_aug_list[i]
            #print(tmp.shape)
            #tmp=tmp[80:-45,80:-50]
            tmp=locate_crop(tmp)

            plt.imshow(tmp)

            mew_im=Image.fromarray(tmp)
            w,h=mew_im.size
            ratio=max(w/190,h/58)

            new_w=int(w/ratio)
            new_h=int(h/ratio)

            last_w=0
            last_h=0
            last_ratio=0
            w_gap=195-new_w
            h_gap=64-new_h
            if(w_gap>20 or h_gap):
                epoch=1
            else:
                epoch=1

            for j in range(0,epoch):
                left=random.randint(5,195-new_w)
                high = random.randint(2, h_gap)
                if(j==1 or j==4):
                    add_ratio=1
                else:
                    add_ratio=(random.randint(7,9)+random.random())/10

                while(last_h==high and last_w==left and last_ratio==add_ratio):
                    add_ratio = (random.randint(7, 9)+random.random())/ 10
                    left = random.randint(5, 195 - new_w)
                    high = random.randint(2, h_gap)
                #print(add_ratio)
                #print(new_h*add_ratio)
                mew_im = mew_im.resize((int(new_w*add_ratio), int(new_h*add_ratio)))

                width = 200
                height = 64
                result = Image.new("L", (width, height))

                result.paste(mew_im, box=(left, high))
                result.save(target_path+file.split(".png")[0]+"-"+str(i)+str(j)+".png")
                last_w=left
                last_h=high
            cv2.imwrite(target_path+file.split(".png")[0]+"-"+str(i)+".png",tmp)

def source_proccess(type):
    pre_path=global_pre_file+type+"_orig/"
    #pre_path = global_pre_file + type + "_aug50/"
    files_list=get_all_file_name(pre_path)

    file_path=global_pre_file+type+"/"
    for file in files_list:
        print(file)
        source_path=pre_path+file
        target_path=file_path+file
        locate_crop(source_path,target_path)
        print(file+" is ok")



type = "train"
path = "data/train/" + type
# 数据增强
data_aug(path,6)

