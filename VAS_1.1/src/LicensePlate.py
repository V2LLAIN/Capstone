"""
This class is for detecting License plate ROI and recognizing License plate id by yolov4 object detection.
As a detailed function, There are function that removing duplicated character detected, sorting detected character in order, etc.
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

from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


class LicensePlate:

    def __new__(self):
        if not hasattr(self, 'instance'):
            self.instance = super(LicensePlate, self).__new__(self)

        return self.instance
    
    def SetStartSignalROI(self, cam8_json_data):
        """ This function is for setting start signal ROI.

        :input param: ROI rect cneter, rect_width, rect_height .

        """
        self.start_rect_center = (cam8_json_data[0], cam8_json_data[1]) #(1725, 300)
        self.rect_lx = cam8_json_data[2] #125
        self.rect_ly = cam8_json_data[3] #125
        self.LP_bottom_border = cam8_json_data[4]
        self.start_signal_truck_dim = cam8_json_data[5]



    
    def LoadLPNetworks(self, lpd_paths, lpd_thres, lpr_paths, lpr_thres) :
        """ This function is for loading networs about LPD and LPID Recognition.

        :input param: paths for network loading, threshold for each network.

        """
        self.LPD_net , self.LPD_meta, self.LPD_colors = darknet.load_network(
            lpd_paths[0], # cfg
            lpd_paths[2], # weight
            lpd_paths[1], # meta
            batch_size=1
        )
        self.LPD_width = darknet.network_width(self.LPD_net)
        self.LPD_height = darknet.network_height(self.LPD_net)
        self.LPD_thres = lpd_thres

        self.LPR_net = darknet.load_net_custom(lpr_paths[0].encode("ascii"), lpr_paths[1].encode("ascii"), 0, 1)  # batch size = 1
        self.LPR_meta = darknet.load_meta(lpr_paths[2].encode("ascii"))
        self.LPR_width = darknet.network_width(self.LPR_net)
        self.LPR_height = darknet.network_height(self.LPR_net)
        self.LPR_thres = lpr_thres

    
    def num_Identification(self, LP_type, trimmed_images, language_flag):
        """ 
        This function is for recognition of LP ID.
        This function replaces the detection result with the correct form of Container ID via LP Rule.

        :input param: LP_type (ex. LP_1 or LP_2), trimmed LP ID images.
        :return: LP ID.

        """
        output_result = None

        LPR_box = darknet.performDetect(trimmed_images, self.LPR_net, self.LPR_meta, self.LPR_thres)
        output_result = self.LP_rule(LP_type, LPR_box, language_flag)
        # print("output_result: ", output_result)

        if output_result is None :
            return None

        if language_flag == 1:
            if len(output_result) == 8 :
                return output_result
            else : return None
        elif language_flag == 2:

            if 6 <= len(output_result) <= 15:
                return output_result
        else: return None
    
    def FrontDetection(self, img, truck, roi_size):
        """ This function is for detecting classes with LPD network from the front camera.

        :input param: front img, front video width, front video height, truck instance.
        :return: flag - if something is detected, return true.
                 detectios - detection result.
                 roi_pts - roi points of the detection result (format : left, top, right, bottom).
                 trimmed_imgs - trimmed side cid area images.
                 LP_type - LP type (ex, LP_1 or LP_2).
        """
        cap_h, cap_w, _ = img.shape
        detections, _ = darknet.detect_NN(img, self.LPD_net, self.LPD_meta, self.LPD_thres)
        if len(detections) ==0 :
            False, None, None, None, None

        roi_pts = darknet.point_cvt(detections, cap_w/self.LPD_width, cap_h/self.LPD_height)

        check_LPD = False
        check_TF = False
        trimmed_imgs = []
        LP_type = None
        for idx in range(len(roi_pts)):
            temp = darknet.im_trim(img, roi_pts[idx])

            if temp is not None and detections[idx][0] == "LP":
                dim = (roi_pts[idx][2] - roi_pts[idx][0]) * (roi_pts[idx][3] - roi_pts[idx][1])

                if dim > roi_size:
                    check_LPD = True

                    trimmed_imgs.append(temp)
                    if detections[idx][0] == "LP":
                        LP_type = "LP"


                    truck.save_img('front', img, roi_pts[idx], 100)

        if not check_LPD:
            for idx in range(len(roi_pts)):
                temp = darknet.im_trim(img, roi_pts[idx])

                if temp is not None and detections[idx][0] == "Truck_front":
                    dim = (roi_pts[idx][2] - roi_pts[idx][0]) * (roi_pts[idx][3] - roi_pts[idx][1])
                    if dim > 160000:
                        check_TF = True
                        truck.save_img('front_not_lp', img, roi_pts[idx], 0)
                        # print("Lp is not detected")
        if not check_LPD and not check_TF:
            if truck.truck_front_flag:
                truck.save_img('front_not_lp', img, [100, 100, 800, 800], 0)
                # print("LP and TF is not detected")


        if len(detections) > 0 and check_LPD == True:
            return True, detections, roi_pts, trimmed_imgs, LP_type
        else :
            return False, None, None, None, None
    
    def CheckMode(self, img, viewer):
        """ This function is for checking whether a truck is coming or not. 

        :input param: front image, front video's width, front video's height.
        :return: if the Truck_front's ROI center is in the start signal ROI, return True.
        """

        cap_h, cap_w, _ = img.shape

        # detect the object
        detections, _ = darknet.detect_NN(img, self.LPD_net, self.LPD_meta, self.LPD_thres)
        roi_pts = darknet.point_cvt(detections, cap_w/self.LPD_width, cap_h/self.LPD_height)

        if detections is not None and viewer is not None:
            viewer.F_img = darknet.draw_boxes(detections, img, (0, 255, 0), roi_pts)
        
        for idx in range(len(detections)):
            if detections[idx][0] == 'Truck_front':
                left, top, right, bottom = roi_pts[idx]
                center_pt = (int(left + (right - left) / 2),  int(top + (bottom - top) / 2))
                dim = (right - left) * (bottom - top)
                # print('dim= ', dim, 'acc=',detections[idx])

                if (( center_pt[0] > (self.start_rect_center[0] - self.rect_lx) and center_pt[0] < (self.start_rect_center[0] + self.rect_lx) ) and
                 ( center_pt[1] > (self.start_rect_center[1] - self.rect_ly) and center_pt[1] < (self.start_rect_center[1] + self.rect_ly)) # inside ROI
                 and (self.start_rect_center[1] + self.rect_ly - self.LP_bottom_border) < bottom # under border
                 and self.start_signal_truck_dim < dim): # dim
                    return True
            elif detections[idx][0] == 'LP':
                left, top, right, bottom = roi_pts[idx]
                center_pt = (int(left + (right - left) / 2), int(top + (bottom - top) / 2))
                dim = (right - left) * (bottom - top)

        return False

    """
    Modified CheckMode for Auto Annotation
    """
    
    def CheckMode_for_AA(self, img, viewer):
        """ This function is for checking whether a truck is coming or not.

        :input param: front image, front video's width, front video's height.
        :return: if the Truck_front's ROI center is in the start signal ROI, return True.
        """

        cap_h, cap_w, _ = img.shape

        # detect the object
        detections, _ = darknet.detect_NN(img, self.LPD_net, self.LPD_meta, self.LPD_thres)
        roi_pts = darknet.point_cvt(detections, cap_w / self.LPD_width, cap_h / self.LPD_height)

        if detections is not None and viewer is not None:
            viewer.F_img = darknet.draw_boxes(detections, img, (0, 255, 0), roi_pts)

        for idx in range(len(detections)):
            if detections[idx][0] == 'Truck_front':
                left, top, right, bottom = roi_pts[idx]
                center_pt = (int(left + (right - left) / 2), int(top + (bottom - top) / 2))
                dim = (right - left) * (bottom - top)

                if ((center_pt[0] > (self.start_rect_center[0] - self.rect_lx) and center_pt[0] < (
                        self.start_rect_center[0] + self.rect_lx)) and
                        (center_pt[1] > (self.start_rect_center[1] - self.rect_ly) and center_pt[1] < (
                                self.start_rect_center[1] + self.rect_ly))  # inside ROI
                        and (self.start_rect_center[1] + self.rect_ly - self.LP_bottom_border) < bottom  # under border
                        and dim > 160000):  # dim
                    return True, detections, roi_pts

        return False, detections, roi_pts


    ### For allowing duplicated letters like ~~~12'34' '34'G12. Without this, it would be ~~~12'34'G12.
    class ClassForAllowingDupleKey(object) :
        def __init__(self, name) :
            self.name = name
        def __str__(self) :
            return self.name
        def __repr__(self) :
            return "'"+self.name+"'"

    
    def LP_rule(self,LP_class, LP_data, language_flag):
        """ This method is for sorting detected character

        1. Remove duplicated character detected.
        2. Sort license plate ID in order by two kinds of license plate
            <license plate in two line>
                -1. Sort with y attribute.
                -2. Select upper 3 characters that are respectively region and two number(ex.경남 99) and Sort with x attribute.
                -3. Sort the remainer with x attribute.

            <license plate in one line>
                -1. Sort with x attribute.

            **Exception handling
                - If the number of detected ID is less than 8

        :input param: LP_class - kind of licens plate whether it has one line or two lines.
                    LP_data - detected ID that has label, confidence, and detected location.
        :return: sorted license plate ID

        """

        ### duplicate reducing ###
        canditate_idxes = []
        remain_idxes = []
        remove_idxes = []
        for idx in range(len(LP_data)):
            max_idx, selected_LP_idx = self.select_LP_data(LP_data[idx], idx, LP_data, 3) # return idx of array of overlapping value

            remain_idxes.append(max_idx)
            if len(selected_LP_idx) > 1:
                canditate_idxes.extend(selected_LP_idx)

        remove_idxes = list(set(canditate_idxes))
        remain_idxes = list(set(remain_idxes))

        for i in remain_idxes :
            if i == -1 :
                remain_idxes.remove(-1)

        for idx in range(len(remain_idxes)):
            remove_idxes.remove(remain_idxes[idx])

        remove_idxes.sort(reverse=True)
        for idx_remove in range(len(remove_idxes)):
            del LP_data[remove_idxes[idx_remove]]

        recog_result = {}

        ruled_LP_data = LP_data



        if language_flag == 1:
            # two lines
            if LP_class == 'LP_1' and len(ruled_LP_data) >= 8:
                # sort with y, and divide into upper 3, and lower 3
                ruled_LP_data.sort(key=lambda ruled_LP_data : ruled_LP_data[2][1])

                up_LP = ruled_LP_data[:3]

                # sort with x coordiante (to read from small x value)
                up_LP.sort(key=lambda up_LP : up_LP[2][0])
                for idx_l in range(len(up_LP)):
                    recog_result.update({self.ClassForAllowingDupleKey(str(up_LP[idx_l][0].decode('utf-8'))) : up_LP[idx_l][1]})

                down_LP = ruled_LP_data[3:]

                # sort with x coordiante (to read from small x value)
                down_LP.sort(key=lambda down_LP : down_LP[2][0])
                for idx_l in range(len(down_LP)):
                    recog_result.update({self.ClassForAllowingDupleKey(str(down_LP[idx_l][0].decode('utf-8'))) : down_LP[idx_l][1]})

            # one line
            elif LP_class == 'LP_2'and len(ruled_LP_data) >= 8:
                # sort with x without dividing into upper and lower.
                ruled_LP_data.sort(key=lambda ruled_LP_data : ruled_LP_data[2][0])
                for idx_l in range(len(ruled_LP_data)):
                    recog_result.update({self.ClassForAllowingDupleKey(str(ruled_LP_data[idx_l][0].decode('utf-8'))) : ruled_LP_data[idx_l][1]})
        elif language_flag == 2:
            # sort with x without dividing into upper and lower.
            ruled_LP_data.sort(key=lambda ruled_LP_data : ruled_LP_data[2][0])
            for idx_l in range(len(ruled_LP_data)):
                recog_result.update({self.ClassForAllowingDupleKey(str(ruled_LP_data[idx_l][0].decode('utf-8'))) : ruled_LP_data[idx_l][1]})


        return recog_result

    
    def select_LP_data(self,cur_LPd, cur_idx, LP_data, dist_thresh):
        """ This function is for eliminating duplicate detection results.

        :input param: cur_LPd - one of the detection results
                  cur_idx - the cur_LPd's index.
                  LP_data - the detection results.
                  dist_thresh - the detection results are removed when they are in the dist_thresh.

        :return: max_idx - Most confident detection result's index among duplicate detection results.
                 duplicate_idx - duplicate detection result's index.

        """
        max_idx = -1
        duplicate_idx = []

        for idx in range(len(LP_data)):
            cx_len = LP_data[idx][2][0] - cur_LPd[2][0]
            cy_len = LP_data[idx][2][1] - cur_LPd[2][1]
            c_distance = math.sqrt((cx_len * cx_len) + (cy_len * cy_len))

            if c_distance < dist_thresh and cur_LPd[0] != LP_data[idx][0]:
                duplicate_idx.append(idx)

        max_confidance = cur_LPd[1]
        if len(duplicate_idx) > 0:
            max_idx = cur_idx
        for idx in range(len(duplicate_idx)):
            if LP_data[duplicate_idx[idx]][1] > max_confidance:
                max_confidance = LP_data[duplicate_idx[idx]][1]
                max_idx = duplicate_idx[idx]

        duplicate_idx.append(cur_idx)
        duplicate_idx.sort()

        if max_idx != -1:
            return max_idx, duplicate_idx
        else:
            return -1, duplicate_idx

