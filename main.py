import os
import time
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader

from model.detection.split_combine import SplitComb
from model.detection.layers import nms
import model.detection.res18 as res18
from preprocessing._classes import CTScan

import model.classification.net_classifier as net_classifier
from model.classification.data_classifier import DataBowl3Classifier

from configs import OUTPUT_PATH, RESOURCES_PATH


def preprocess(seriesuids):
    start_time = time.time()

    dirs = os.listdir(f'{RESOURCES_PATH}/')
    for seriesuid in list(set(dirs).intersection(set(seriesuids))):
        print(f'{seriesuid} is loading...')
        ct = CTScan(seriesuid = seriesuid)
        ct.preprocess()
        ct.save_preprocessed_image()
    end_time = time.time()
    print('    elapsed time is %3.2f seconds\n' % (end_time - start_time))


def detecte(seriesuids):
    start_time = time.time()

    config, net, loss, get_pbb = res18.get_model()
    checkpoint = torch.load('./model/detection/res18fd9020.ckpt')
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    net.eval()

    # Split and Combine
    margin = 32
    sidelen = 144
    stride = config['stride']
    split_comber = SplitComb(sidelen, config['max_stride'], stride, margin, config['pad_value'])

    dirs = os.listdir(f'{OUTPUT_PATH}/')
    for seriesuid in list(set(dirs).intersection(set(seriesuids))):
        print(f'{seriesuid} is being detected...')

        # Load
        imgs = np.load(f'{OUTPUT_PATH}/{seriesuid}/{seriesuid}_clean.npy')
        print(f'    INPUT imgs.shape = {imgs.shape}')

        # Pad
        nz, nh, nw = imgs.shape[1:]
        pz = int(np.ceil(float(nz) / stride)) * stride
        ph = int(np.ceil(float(nh) / stride)) * stride
        pw = int(np.ceil(float(nw) / stride)) * stride
        imgs = np.pad(imgs, [[0,0],[0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',constant_values = config['pad_value'])
        print(f'    PADDED imgs.shape = {imgs.shape}')

        # Split
        xx, yy, zz = np.meshgrid(np.linspace(-0.5,0.5, imgs.shape[1]//stride),
                    np.linspace(-0.5,0.5, imgs.shape[2]//stride),
                    np.linspace(-0.5,0.5, imgs.shape[3]//stride), indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
        coord2, nzhw2 = split_comber.split(coord,
                                    side_len = split_comber.side_len//stride,
                                    max_stride = split_comber.max_stride//stride,
                                    margin = split_comber.margin//stride)
        imgs, nzhw = split_comber.split(imgs)
        
        assert np.all(nzhw == nzhw2)
        imgs = (imgs.astype(np.float32)-128)/128
        print(f'    SPLITTED imgs.shape = {imgs.shape}')

        # DETECTION
        data, target, coord, nzhw = torch.from_numpy(imgs), [], torch.from_numpy(coord2), np.array(nzhw)
        
        n_per_run = 4
        splitlist = list(range(0, len(data)+1, n_per_run))
        if splitlist[-1] != len(data):
            splitlist.append(len(data))

        outputlist = []
        for j in range(len(splitlist)-1):
            print(f'        p {j} in splitlist')
            with torch.no_grad():
                input = Variable(data[splitlist[j]:splitlist[j+1]]).cuda()
                inputcoord = Variable(coord[splitlist[j]:splitlist[j+1]]).cuda()
                output = net(input, inputcoord)
                outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        print(f'    Get output.shape = {output.shape}')

        # Get pbb
        thresh = -3 # or -8
        pbb, mask = get_pbb(output, thresh, ismask = True)
        pbb = np.array(pbb[pbb[:,0] > -2])
        pbb = nms(pbb, 0.1)
        np.save(f'{OUTPUT_PATH}/{seriesuid}/{seriesuid}_pbb.npy', pbb)

    end_time = time.time()
    print('    elapsed time is %3.2f seconds\n' % (end_time - start_time))


def classify(seriesuids):
    start_time = time.time()

    net, config = net_classifier.get_model()
    checkpoint = torch.load('./model/classification/classifier.ckpt', encoding= 'unicode_escape')
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    net.eval()
    cudnn.benchmark = True
    net = DataParallel(net)

    dirs = os.listdir(f'{OUTPUT_PATH}/')
    testsplit = list(set(dirs).intersection(set(seriesuids)))
    dataset = DataBowl3Classifier(testsplit, config, phase = 'test')
    data_loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 32,
        pin_memory=True)

    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i, (x, coord, pbb) in enumerate(data_loader):
        print(f'{seriesuids[i]} is being classified...')
        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        crop, out = net(x,coord)
        np.save(f'{OUTPUT_PATH}/{testsplit[i]}/{testsplit[i]}_crop.npy', crop.cpu().numpy())
        # print(pbb.shape, out.shape)
        for idx in range(out.shape[1]):
            pbb[0, idx, 0] = out[0, idx]
        pbb.sort()
        np.save(f'{OUTPUT_PATH}/{testsplit[i]}/{testsplit[i]}_pbb.npy', pbb.cpu().detach().numpy())
    end_time = time.time()
    print('    elapsed time is %3.2f seconds\n' % (end_time - start_time))


if __name__ == '__main__':
	seriesuids = ['chenpeiying1']
	[os.makedirs(f'{OUTPUT_PATH}/{d}', exist_ok=True) for d in seriesuids]
	preprocess(seriesuids)
	detecte(seriesuids)
	classify(seriesuids)




