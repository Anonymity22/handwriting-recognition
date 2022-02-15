import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import numpy as np
import math
import collections
import models.crann as crnn
#import models.res_crnn as resnn
import models.crann as crann
import matplotlib.pyplot as plt
import os

#model_path="./expr/bishe_CRANN.pth"
model_path="./expr/netCRANN_cp.pth"
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
img_root_path="data/demo/"

model=crann.CRNN(32,1,37,256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))
model=model.eval()
converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))

def get_all_file_name(file_dir):
    for root,dirs,files in os.walk(file_dir):
        return files

file_list=get_all_file_name(img_root_path)
print(file_list)
n_correct=0
n_all=0

for file in file_list:
    target=file.split('-')[0]
    n_all = n_all + len(target)
    print("label:",target)
    image = Image.open(img_root_path+file).convert('L')
    image = transformer(image)

    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    preds = model(image,0)[0].log_softmax(2)

    tmp=preds.permute(1,0,2).detach().cpu().numpy()
    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))

    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

    result =utils.decode(tmp)
    if(sim_pred==target):
        n_correct+=1

print("number of words:", len(file_list))
print("accuarcy:", n_correct/len(file_list))


