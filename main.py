import os
import torch

import model.detection.res18 as res18
from preprocessing._classes import CTScan
from configs import OUTPUT_PATH, RESOURCES_PATH


def preprocess(seriesuids):
	dirs = os.listdir(f'{RESOURCES_PATH}/')
	for seriesuid in list(set(dirs).intersection(set(seriesuids))):
		# print(seriesuid)
		CTScan
		ct = CTScan(seriesuid = seriesuid)
		ct.preprocess()
		ct.save_preprocessed_image()


def detecte(seriesuids):
	config, res18, loss, get_pbb = res18.get_model()
	checkpoint = torch.load('./model/detection/res18fd9020.ckpt')
	res18.load_state_dict(checkpoint['state_dict'])


def classify(seriesuids):
	pass


if __name__ == '__main__':
	seriesuids = ['xuyi']
	preprocess(seriesuids)




