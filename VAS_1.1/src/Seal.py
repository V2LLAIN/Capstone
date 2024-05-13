"""
This class is for seal detection using yolov4
"""

from ctypes import *
import math
import random
import os
import numpy as np
import cv2
import glob
import time
from scipy.linalg import inv,norm

import darknet

class Seal:

    def __new__(self):
        if not hasattr(self, 'instance'):
            self.instance = super(Seal, self).__new__(self)

        return self.instance

    def LoadSealNetworks(self, sealD_paths, sealD_thresh, sealR_paths, sealR_thresh):
        """ This function is for loading networs about sealD.

        :input param: paths for network loading, threshold for each network.

        """   
        self.sealD_net, self.sealD_meta, self.sealD_colors = darknet.load_network(
            sealD_paths[0], # cfg
            sealD_paths[2], # meta
            sealD_paths[1], # weight
            batch_size = 1
        )
        self.sealD_width = darknet.network_width(self.sealD_net)
        self.sealD_height = darknet.network_height(self.sealD_net)
        self.sealD_thres = sealD_thresh


        self.sealR_net, self.sealR_meta, self.sealR_colors = darknet.load_network(
            sealR_paths[0], # cfg
            sealR_paths[2], # meta
            sealR_paths[1], # weight
            batch_size = 1
        )
        self.sealR_width = darknet.network_width(self.sealR_net)
        self.sealR_height = darknet.network_height(self.sealR_net)
        self.sealR_thres = sealR_thresh



    def do_sharpening(self, img):
        """ 
        
        (preprocessing) make seal image sharpen

        """
        sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, 2, 9, 2, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, -1, -1, -1, -1]]) / 9.0
        sharpening_3 = np.array([[-1,-1,-1,-1,-1],
                            [-1,2,2,2,-1],
                            [-1,2,8,2,-1],
                            [-1,2,2,2,-1],
                            [-1,-1,-1,-1,-1]]) / 8.0
        
        sharpend_img = cv2.filter2D(img, -1, sharpening_2)

        return sharpend_img



    def seal_detection(self, img_origin):
        """
        detect the seal presense

        Input: back image
        Return: detected image and detected signal
        """

        ### preprocessing ###
        img = self.do_sharpening(img_origin)
        img_h, img_w, _ = img.shape

        detections, _ = darknet.detect_NN(img, self.sealD_net, self.sealD_meta, self.sealD_thres)
        roi_pts = darknet.point_cvt(detections, img_w/self.sealD_width, img_h/self.sealD_height)

        detections_switch = []
        ### [sealR] Trimmed Images ###
        for idx in range(len(detections)):
            temp_img = darknet.im_trim(img, roi_pts[idx])
            temp_img_h, temp_img_w, _ = temp_img.shape

            ### seal checker ###
            trim_detections, _ = darknet.detect_NN(temp_img, self.sealR_net, self.sealR_meta, self.sealR_thres)
            trim_roi_pts = darknet.point_cvt(trim_detections, temp_img_w/self.sealR_width, temp_img_h/self.sealR_height)
            if len(trim_detections) != 0 and trim_detections[0][0] == "True": detections_switch.append(True)
            else: detections_switch.append(False)

        detected_img = darknet.draw_boxes_switches(detections, detections_switch, img, roi_pts)

        for switch in detections_switch:
            if switch == True: return detected_img, True

        return detected_img, False