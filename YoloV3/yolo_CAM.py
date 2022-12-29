import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

# from YoloV3.nets.yolo3 import YoloBody
from YoloV3.nets.yolo3_CAM import YoloBody 
from YoloV3.utils.utils import (DecodeBox)

from YoloV3 import attention

class YOLO(object):
    _defaults = {
        "model_path"        : r'', #YoloV3\checkpoint\Epoch300-Total_Loss1.9337-Val_Loss1.9373.pth
        "anchors_path"      : r'YoloV3\utils\yolo_anchors.txt',
        "classes_path"      : r'YoloV3\utils\CARLA_classes.txt',
        "model_image_size"  : (608, 608, 1),
        "confidence"        : 0.25,
        "iou"               : 0.5,
        "cuda"              : True,
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self,config, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()        
        self.config=config
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]


    def generate(self):
        self.num_classes = len(self.class_names)
        self.net = YoloBody(self.anchors, self.num_classes)

        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path=self.config.weightfile  
        state_dict = torch.load(model_path, map_location=device)#self.model_path
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], self.num_classes, (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        #Grad-CAM
        self.grad_cam = attention.Attention(self.net, ori_shape=self._defaults["model_image_size"],
                         final_shape=self._defaults["model_image_size"],yolo_decodes=self.yolo_decodes,
                         num_classes=self.num_classes, conf_thres=self._defaults["confidence"], 
                         nms_thres=self._defaults["iou"])
        
    def get_output(self, images):
        outputs = self.net(images)
        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))           
        output = torch.cat(output_list, 1)                                   
        return output
     
                  
  
