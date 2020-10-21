import gluoncv
from mxnet.gluon import data as gdata, nn
import mxnet as mx
from mxnet import nd
import numpy as np
import cv2
import os

category_list = ['bjbmyw', 'dmyw', 'gkxfw', 'nc']

def get_net():
    finetune_net = gluoncv.model_zoo.get_model(name='mobilenetv3_large',
                                               pretrained=False)
    with finetune_net.name_scope():
        finetune_net.output.add(nn.Dense(4))

    finetune_net.load_parameters('./resource/mobilenetv3.params')

    return finetune_net


aug = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
])


def classify(img_array):
    net = get_net()
    img = nd.array(img_array)
    input_data = nd.stack(aug(img))

    output = net(input_data)
    category_id = nd.softmax(output).argmax(axis=1).asscalar()

    return category_list[category_id.astype('int')]
    