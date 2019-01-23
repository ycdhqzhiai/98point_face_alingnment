# coding: utf-8
import sys
import os
sys.path.insert(0, '/home/yc/workplace/deeplearning/face/caffe/python')
import caffe
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

data_dir = '../data/'
new_dir = './98_point/'


train_lab_txt = './train_lab.txt'
test_lab_txt = './test_lab.txt'

train_txt = './train.txt'
test_txt = './test.txt'

input_width = 64
input_height = 64

def cal_padding(bbox, im):
    '''
    计算padding的大小
    :param bbox: 
    :param im: 
    :return: 
    '''
    x1,y1,x2,y2 = bbox
    h,w,c = im.shape
    pad = np.max([-x1, -y1, x2 - w, y2 - h, 0]) + 10
    return int(pad)

def preprocess(im, bbox):
    # 添加padding
    pad = cal_padding(bbox, im)
    im_pad = cv2.copyMakeBorder(im, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    bbox = bbox + pad
    # 尺度变换
    bb_w = bbox[2] - bbox[0]
    scale = bb_w * 1.0 / input_width
    h, w, c = im_pad.shape
    # Important
    bbox = bbox / scale
    bbox[0] = round(bbox[0])
    bbox[1] = round(bbox[1])
    bbox[2:] = bbox[0:2] + [input_width - 1, input_height - 1]
    bbox = bbox.astype(np.int32)
    im_pad = cv2.resize(im_pad, (int(w / scale), int(h / scale)))
    cropImg = im_pad[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    return cropImg,pad,bbox[0:2]+1,scale

def pad_bbox(bbox, pad_ratio):
    '''
    添加padding
    :param bbox: 
    :param pad_ratio: 添加
    :return: 
    '''
    # padding
    pad_w = (bbox[2] - bbox[0]) * pad_ratio
    pad_h = (bbox[3] - bbox[1]) * pad_ratio
    bbox = np.array([bbox[0] - pad_w, bbox[1], bbox[2] + pad_w, bbox[3] + 2 * pad_h])
    return np.array(bbox)

def obtain_bbox(bbox, w_h_ratio):
    '''
    生成特定长宽比的bbox
    :param bbox: 
    :param w_h_ratio: 输出的bbox的宽高比
    :return: 
    '''

    print (bbox)
    bbox = np.array(bbox).astype(np.float32)

    w, h = bbox[2:] - bbox[0:2] + 1
    # 确保高宽比
    if w*1.0/h >= w_h_ratio:
        pad_h = (w/w_h_ratio -h)/2
        pad_w = 0
    elif w/h < w_h_ratio:
        pad_h = 0
        pad_w = (h*w_h_ratio -w)/2
    bbox = bbox[0] - pad_w, bbox[1] - pad_h, bbox[2] + pad_w, bbox[3] + pad_h
    return np.array(bbox)


def gendata(lab_txt ,target_txt, symbol):
	with open(lab_txt, 'w') as target:
		count = 0
		for line in open(target_txt):
			bbox_exp = []
			bbox_roi = []
			landmarks_roi = []
			if line.isspace() : continue
			img_name = line.split()[0]
			landmarks_str = line.split()[5:204]	
			
 			sub_dir = img_name.split('/')[1]			
			sub_path = os.path.join(new_dir, symbol, sub_dir)
			if not os.path.exists(sub_path) : os.makedirs(sub_path)
			savename = sub_path + '/' + img_name.split('/')[2]		
			print savename
	
			landmarks = list(map(float, landmarks_str))	
			img_path = data_dir + img_name
			img = cv2.imread(img_path)

			bbox = []
			min_x = landmarks[0]
			min_y = landmarks[1]
			max_x = 0
			max_y = 0
			
			for key in range(0, len(landmarks), 2):
				if min_x > landmarks[key] : min_x = landmarks[key]
				if min_y > landmarks[key + 1] : min_y = landmarks[key + 1]
				if max_x < landmarks[key] : max_x = landmarks[key]
				if max_y < landmarks[key + 1] : max_y = landmarks[key + 1]

			bbox.append(min_x)
			bbox.append(min_y)
			bbox.append(max_x)
			bbox.append(max_y)

			#cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
			#cv2.imshow('test', img)
			#cv2.waitKey(0)
		
			# preprocess
			bbox = np.array(bbox, dtype=np.float32)
			bbox = obtain_bbox(bbox, input_width * 1.0 / input_height)
			bbox = pad_bbox(bbox, pad_ratio=0.15)
			cropImg, pad, offset, scale = preprocess(img, bbox)
			cv2.imwrite(savename, cropImg)

			target.write(savename)
                        target.write(' ')
	
			for key in range(0, len(landmarks), 2):	
				landmarks_x = (np.array(landmarks[key]) + 1 + pad) / scale - offset[0]
				landmarks_y = (np.array(landmarks[key + 1]) + 1 + pad) / scale - offset[1]
				#cv2.circle(cropImg,(int(landmarks_x),int(landmarks_y)),1,(0,0,255),1)

				# normalize
				landmarks_normal_x = (landmarks_x + 1) / input_width
				landmarks_normal_y = (landmarks_y + 1) / input_height
				#print landmarks_normal_x, landmarks_normal_y
                                target.write(str(landmarks_normal_x))
				target.write(' ')	
				target.write(str(landmarks_normal_y))
				if key == 194:
					target.write('\n')
				else:
					target.write(' ')
			#cv2.imshow('test', cropImg)
			#cv2.waitKey(0)			
		open(target_txt).close	
#gendata(test_txt, test_lab_txt, 'test')
gendata(train_txt, train_lab_txt, 'train')
