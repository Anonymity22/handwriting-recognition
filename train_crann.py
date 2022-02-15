from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import time
from torch.autograd import Variable
import numpy as np
import os
import utils
import difflib
import sys
import dataset
from PIL import Image
import matplotlib.pyplot as plt

import models.crann as crann

parser = argparse.ArgumentParser()
parser.add_argument('--trainSourceRoot', help='path to dataset',default="data/source/train_lmdbdata/")
parser.add_argument('--valSourceRoot', help='path to dataset',default="data/source/test_lmdbdata/")
parser.add_argument('--trainTargetRoot', help='path to dataset',default="data/target/train_lmdbdata/")
parser.add_argument('--valTargetRoot', help='path to dataset',default="data/target/test_lmdbdata/")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')  #32
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')  #100
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', default=True,action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#expr/test_netCRNN.pth
#expr/rot_netCRNN.pth
parser.add_argument('--pretrained', default='expr/bishe_CRANN.pth', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=535, help='Interval to be displayed')  #
parser.add_argument('--n_test_disp', type=int, default=5, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=535, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)#manualSeed是啥？
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best= 0.5
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_source_dataset = dataset.lmdbDataset(root=opt.trainSourceRoot)
assert train_source_dataset
if not opt.random_sample:
    source_sampler = dataset.randomSequentialSampler(train_source_dataset, opt.batchSize)
else:
    sampler = None

train_target_dataset = dataset.lmdbDataset(root=opt.trainTargetRoot)
assert train_target_dataset
if not opt.random_sample:
    target_sampler = dataset.randomSequentialSampler(train_target_dataset, opt.batchSize)
else:
    sampler = None

train_source_loader = torch.utils.data.DataLoader(
    train_source_dataset, batch_size=opt.batchSize,
    sampler=source_sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_source_dataset = dataset.lmdbDataset(
    root=opt.valSourceRoot, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

train_target_loader = torch.utils.data.DataLoader(
    train_target_dataset, batch_size=opt.batchSize,
    sampler=target_sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_target_dataset = dataset.lmdbDataset(
    root=opt.valTargetRoot, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

nclass = len(opt.alphabet) + 1  #第二次LSTM得到的输出的类别数
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = torch.nn.CTCLoss()
loss_domain = torch.nn.NLLLoss()

# custom weights initialization called on crann
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crann = crann.CRNN(opt.imgH, nc, nclass, opt.nh)

crann.apply(weights_init)
if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    try:

        crann.load_state_dict(torch.load(opt.pretrained))
    except RuntimeError:
        pretrained_dict = torch.load(opt.pretrained)
        model_dict=crann.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k!='rnn.1.embedding.weight' and k!='rnn.1.embedding.bias'}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        crann.load_state_dict(model_dict)


print(crann)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.LongTensor(opt.batchSize * 5)
length = torch.LongTensor(opt.batchSize)

t_image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
t_text = torch.LongTensor(opt.batchSize * 5)
t_length = torch.LongTensor(opt.batchSize)

if opt.cuda:
    crann.cuda()
    #crann = torch.nn.DataParallel(crann, device_ids=range(opt.ngpu))  #可能存在负载不均衡
    image = image.cuda()
    t_image=t_image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

t_image = Variable(t_image)
t_text = Variable(t_text)
t_length = Variable(t_length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crann.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crann.parameters())
else:
    optimizer = optim.RMSprop(crann.parameters(), lr=opt.lr)

def jud(a,target,dic,num):
    if(len(a)==len(target)):
        for i in range(0,len(a)):
            if(a[i]!=target[i]):
                if((a[i],target[i]) in dic):
                    dic[(a[i],target[i])]+=1
                else:
                    dic[(a[i], target[i])] = 1
            if(target[i] in dic):
                dic[target[i]]+=1
            else:
                dic[target[i]]=1
    else:
        num[0]+=1

def val(net, dataset,criterion,type="t", max_iter=100):
    for p in crann.parameters():
        p.requires_grad = False

    net=net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    pre_search_num = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    dic={}
    num=[0]
    wrong_word = []
    wrong_pred = []
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)


        utils.loadData(text, t)
        utils.loadData(length, l)

        preds,_ = net(image,0)
        preds=preds.log_softmax(2).requires_grad_()
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        # for item in preds:
        #     print(item)
        cost = criterion(preds, text, preds_size, length)
        loss_avg.add(cost)


        st=time.perf_counter()
        _, preds = preds.max(2)
        #preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

        #print("argmax time:\t\t",time.clock()-st)
        sim=0
        #print(final)

        for pred, target in zip(sim_preds, cpu_texts):
            sim+=difflib.SequenceMatcher(None, pred, target.lower()).quick_ratio()
            # if final[n]==target.lower():
            #     pre_search_num+=1
            if pred == target.lower():
                n_correct += 1
            else:
                wrong_word.append(target)
                wrong_pred.append(pred)
                #print(pred,target.lower())
                jud(pred,target.lower(),dic,num)
    if(type=="t"):
        print(dic,num)
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        accuracy = n_correct / 3000
        print(wrong_word)
        print(wrong_pred)
    else:
        accuracy = n_correct / 2071


    print(n_correct)
    pre_search_acc=pre_search_num/float(max_iter * 100)
    sim=sim/float(max_iter * 100)
    print('Test loss: %f, accuray: %f,pre_search_accuray: %f,  similarity: %f' % (loss_avg.val(), accuracy,pre_search_acc,sim))
    return accuracy



def trainBatch(net, criterion, optimizer,alpha):
    source_data = train_source_iter.next()
    target_data=train_target_iter.next()


    # source
    cpu_images, cpu_texts = source_data  # cpu_images N*1*imgH*imgW   1可能是channel
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    optimizer.zero_grad()
    domain_label = torch.zeros(batch_size).long().cuda()
    preds,domain_output = net(image,alpha)
    preds=preds.log_softmax(2).requires_grad_()
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    source_label_loss = criterion(preds, text, preds_size, length)
    source_domain_loss = loss_domain(domain_output, domain_label)

    # target
    t_cpu_images, t_cpu_texts = target_data  # cpu_images N*1*imgH*imgW   1可能是channel
    t_batch_size = t_cpu_images.size(0)
    utils.loadData(t_image, t_cpu_images)
    t_t, t_l = converter.encode(t_cpu_texts)
    utils.loadData(t_text, t_t)
    utils.loadData(t_length, t_l)
    optimizer.zero_grad()
    t_domain_label = torch.ones(t_batch_size).long().cuda()
    t_preds,t_domain_output = net(t_image,alpha)
    t_preds=t_preds.log_softmax(2).requires_grad_()
    t_preds_size = Variable(torch.IntTensor([t_preds.size(0)] * t_batch_size))
    target_label_loss = criterion(t_preds, t_text, t_preds_size, t_length)
    target_domain_loss = loss_domain(t_domain_output, t_domain_label)

    cost=source_label_loss+source_domain_loss+target_domain_loss+target_label_loss


    sys.stdout.write(
        '\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_t_label: %f,err_s_domain: %f, err_t_domain: %f, total_err: %f' \
        % (epoch, i + 1, len_dataloader, source_label_loss.data.cpu().detach().numpy(), target_label_loss.data.cpu().detach().numpy(),
           source_domain_loss.data.cpu().detach().numpy(), target_domain_loss.data.cpu().detach().item(), cost.data.cpu().detach().numpy()))
    sys.stdout.flush()

    cost.backward()
    optimizer.step()
    return cost


for epoch in range(opt.nepoch):
    train_source_iter = iter(train_source_loader)
    train_target_iter=iter(train_target_loader)
    i = 0
    st = time.perf_counter()
    len_dataloader=min(len(train_source_loader),len(train_target_loader))
    opt.displayInterval=len_dataloader
    opt.valInterval=len_dataloader
    while i < len_dataloader:
        for p in crann.parameters():
            p.requires_grad = True
        crann=crann.train()

        p = float(i + epoch * len_dataloader) / opt.nepoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        cost = trainBatch(crann, criterion, optimizer,alpha)
        loss_avg.add(cost)

        i += 1

        if i % opt.displayInterval == 0:
            print()
            print('[%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            print("Start val")
            acc = val(crann,test_source_dataset, criterion,"s")
            acc = val(crann, test_target_dataset, criterion)
            if(acc>best):
                best=acc
                torch.save(
                    crann.state_dict(), '{0}/netCRANN_best.pth'.format(opt.expr_dir))

    print(time.perf_counter() - st)
        # do checkpointing
    if epoch % 5 == 0:
        #pass
        torch.save(
            crann.state_dict(), '{0}/netCRANN_cp.pth'.format(opt.expr_dir))
