import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import functools
import re
def cmp(a, b):
    if(len(a[0])>len(b[0])):
        return 1
    return -1

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(v, bytes):
                # 图片类型为bytes
                txn.put(k.encode(), v)
            else:
                # 标签类型为str, 转为bytes
                txn.put(k.encode(), v.encode())


def createDataset(outputPath, imagePathList, labelList, size=19951162,lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=size) #    env = lmdb.open(outputPath, map_size=109951162)
    cache = {}
    cnt = 1
    for i in range(0,nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def get_all_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

def get_path_label(path):
    file_list=get_all_file_name(path)
    label_list=[]
    for file in file_list:
        # 获取label。
        label_list.append(re.split("-|_",file)[0])
    file_list=[path+file for file in file_list]
    return file_list,label_list


# 生成lmdb格式数据集
if __name__ == '__main__':
    date_type=["source/","target/"]
    data_mode=["test","train","val"]
    type=date_type[0]
    mode=data_mode[1]
    if(mode=="train"):
        pre_file="data/"+type+mode+"/"
        map_size=450000000  # 根据数据集数量大小需要调整
    elif(mode=="val"):
        pre_file = "data/val/"
    elif(mode=="test"):
        pre_file = "data/"+type+mode+"/"
        map_size = 10951162

    # 生成每个图片的path和对应label
    # 图片的文件名应为 label-其它.png ，才能正确获取label
    path_list,label_list=get_path_label(pre_file)

    # 按label长度排序
    label_list, path_list = zip(*sorted(list(zip(label_list, path_list)), key=functools.cmp_to_key(cmp)))
    createDataset("data/"+type+mode+"_lmdbdata",path_list,label_list,size=map_size)
