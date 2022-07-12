# -*- coding: utf-8 -*-

#ignore future warnings
import warnings
warnings.simplefilter('ignore', FutureWarning)

import numpy as np
from PIL import Image
import json, os, datetime
from tensorflow.keras.models import load_model
from tensorflow_hub import KerasLayer

# get now time
dt_now = datetime.datetime.now()
dirname = '{:0>4}/{:0>2}/'.format(dt_now.year, dt_now.month)
filename = '{:0>2}-{:0>2}-00'.format(dt_now.day, dt_now.hour)


# open model and load weights
model = load_model('/root/opt/model/model.h5', custom_objects={"KerasLayer": KerasLayer})
print('target :', '\033[32m' + dirname + filename + '.png\033[0m')
model.summary()

# import coordinate data
cam_arr = json.load(open('/root/storage/config/inference-new.json', 'r'))['object_list']

# open target image
image = Image.open('/root/storage/storage/Image/testProject/1/' + dirname + filename + '.png')


image_size = 128
label = ['cover', 'none', 'few', 'many']
output = {'data': []}


# device individual times loop
for index, data in enumerate(cam_arr):
    print('\ncamera-{:0>4}'.format(index+1))
    apex, target = [], []
    output_child = {'camera-{:0>4}'.format(index+1): []}

    # import indivisual coordinate from json
    for i in data['camera-{:0>4}'.format(index+1)]:
        for j in ['left-top', 'right-bottom']:
            for k in ['x', 'y']:
                apex.append(i[j][k])

    #crop, resize and convert image to RGB, ndarray
    apex = np.asarray(apex)
    apex = np.reshape(apex, [len(data['camera-{:0>4}'.format(index+1)]), 4])
    for i in range(int(apex.size/4)):
        tmp = image.crop(apex[i])
        tmp = tmp.convert("RGB")
        tmp = tmp.resize((image_size, image_size))
        tmp = np.asarray(tmp)
        target.append(tmp)
    target = np.array(target)
    target = target.astype('float32')
    target = target / 255.0

    # predict and get result
    y_predict = model.predict(target)
    answers = y_predict.argmax(axis=-1)

    # print and write predict result
    for i in range(len(data['camera-{:0>4}'.format(index+1)])):
        if label[answers[i]] == 'none':
            print('\033[41m {:^6}\033[0m'.format(label[answers[i]]), end='')
        if label[answers[i]] == 'few':
            print('\033[43m {:^6}\033[0m'.format(label[answers[i]]), end='')
        if label[answers[i]] == 'many':
            print('\033[42m {:^6}\033[0m'.format(label[answers[i]]), end='')
        if label[answers[i]] == 'cover':
            print('\033[07m {:^6}\033[0m'.format(label[answers[i]]), end='')
        output_child['camera-{:0>4}'.format(index+1)].append({'name': 'camera-{:0>4}-{:0>4}'.format(index+1, i+1), 'status': label[answers[i]]})
    output['data'].append({'camera-{:0>4}'.format(index+1): output_child['camera-{:0>4}'.format(index+1)]})
    print('')

# save predict result
with open('/root/storage/inference/result{}.json'.format(filename), 'w') as f:
    json.dump(output, f, indent=4)
print('The predict result is saved as {}.'.format('result-{}.json'.format(filename)))