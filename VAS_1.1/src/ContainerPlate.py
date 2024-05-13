"""
This class is for detecting container ROI and recognizing container id by yolov4 object detection.
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


class ContainerPlate :
    
    def __new__(self):
        if not hasattr(self, 'instance'):
            self.instance = super(ContainerPlate, self).__new__(self)

        return self.instance
    
    def SetEndSignalROI(self, cam5_json_data):
        """ This function is for setting start signal ROI.

        :input param: ROI rect cneter, rect_width, rect_height .

        """   
        self.end_rect_center = (cam5_json_data[0], cam5_json_data[1]) 
        self.rect_lx = cam5_json_data[2] 
        self.rect_ly = cam5_json_data[3] 
        self.end_from_line = cam5_json_data[4]
        self.end_to_line = cam5_json_data[5]
        self.boom_barrier_line = cam5_json_data[6]
    
    def LoadCPNetworks(self, cpd_paths, cpd_thres, cidr_paths, cidr_thres) : 
        """ This function is for loading networs about CPD and CID Recognition.

        :input param: paths for network loading, threshold for each network.


        """        
        self.CPD_net, self.CPD_meta, self.CPD_colors = darknet.load_network(
            cpd_paths[0],
            cpd_paths[2],
            cpd_paths[1],
            batch_size=1
        )
        self.CPD_thres = cpd_thres
        self.CPD_width = darknet.network_width(self.CPD_net)
        self.CPD_height = darknet.network_height(self.CPD_net)

        #CIDR network load
        self.CIDR_net = darknet.load_net_custom(cidr_paths[0].encode("ascii"), cidr_paths[1].encode("ascii"), 0, 1)  # batch size = 1
        self.CIDR_meta = darknet.load_meta(cidr_paths[2].encode("ascii"))
        self.CIDR_thres = cidr_thres
        self.CIDR_width = darknet.network_width(self.CIDR_net)
        self.CIDR_height = darknet.network_height(self.CIDR_net)

    
    def SideDetection(self, img, truck_section, truck):
        """ This function is for detecting classes with CPD network from the side camera.

        :input param: side img, side video width, side video height, truck section(ex, "left_U", "right_U"), truck instance.
        :return: flag - if something is detected, return true.
                 detectios - detection result.
                 roi_pts - roi points of the detection result (format : left, top, right, bottom).
                 trimmed_imgs - trimmed side cid area images.
                 is_gap - if container gap ('cont_center') is detected, return true.
        """
        cap_h, cap_w, _ = img.shape

        ### detection start.
        detections, _ = darknet.detect_NN(img, self.CPD_net, self.CPD_meta, self.CPD_thres)
        is_gap = False

        if len(detections) ==0 :
            return False, None, None, None, is_gap


        if truck.is_twin_truck is False :
            ### container center check.
            is_gap = darknet.isGap(detections)
        
        roi_pts = darknet.point_cvt(detections, cap_w/self.CPD_width, cap_h/self.CPD_height)
        
        trimmed_imgs = []
        for idx in range(len(detections)):
            if detections[idx][0] == 'side':
                temp = darknet.im_trim(img, roi_pts[idx]) 
                if temp is not None:
                    if truck_section == 'left_U':
                        truck.save_img(truck_section, img, roi_pts[idx], 10)

                    elif truck_section == 'right_U':
                        truck.save_img(truck_section, img, roi_pts[idx], 10)
                    trimmed_imgs.append(temp)
        
        if len(detections) > 0:
            return True, detections, roi_pts, trimmed_imgs, is_gap

        return False, None, None, None, is_gap
    
    def TopRearDetection(self, img, end_queue, boombarrier_queue, prev_cp, truck_section, truck):
        """ 
        This function is for detecting classes with CPD network from the side camera.
        This function also check the end signal2(using 'truck_back').
        In some cases, detection result using top camera include other lane's truck, to prevent this problem, we used filtering.   
        :input param: top rear img, prev_cy(prev truck_back's y coordinate), truck section(ex, "top"), truck instance.
        :return: flag - if something is detected, return true.
                 detectios - detection result.
                 roi_pts - roi points of the detection result (format : left, top, right, bottom).
                 trimmed_imgs - trimmed side cid area images.
                 img_labels - trimmed img's labels.
        """
        cap_h, cap_w, _ = img.shape
        detections_before_filtering, _ = darknet.detect_NN(img, self.CPD_net, self.CPD_meta, self.CPD_thres)

        if len(detections_before_filtering) == 0 :
            return False, None, None, None, None, end_queue, boombarrier_queue, prev_cp

        roi_pts_before_filtering = darknet.point_cvt(detections_before_filtering, cap_w/self.CPD_width, cap_h/self.CPD_height)

        roi_pts = []
        detections = []

        ### filtering
        for idx in range(len(detections_before_filtering)):
            roi_pts.append(roi_pts_before_filtering[idx])
            detections.append(detections_before_filtering[idx])

        # check availability for Chassis Position in img
        is_Cont = False
        is_Truck = False
        Cont_pt = None
        Truck_Pt = None

        # select the max dim detection result
        cx = 999999
        cy = 999999
        if truck_section == 'toprear':
            y_diff = -1

            for idx in range(len(detections)):
                # for Chassis Position
                if detections[idx][0] == 'True' or detections[idx][0] == 'False':
                    is_Cont = True
                    Cont_pt = roi_pts[idx]
                elif detections[idx][0] == 'Truck_back':
                    is_Truck = True
                    Truck_Pt = roi_pts[idx]


                # for Door Direction
                if detections[idx][0] == 'True' and float(detections[idx][1]) > 50:
                    truck.DD_true_count = truck.DD_true_count + 1
                if detections[idx][0] == 'False' and float(detections[idx][1]) > 50:
                    truck.DD_false_count = truck.DD_false_count + 1

                # for end signal
                if detections[idx][0] == 'Truck_back' and float(detections[idx][1]) > 50:
                    left, top, right, bottom = roi_pts[idx]
                    cx = left + (right - left)/2
                    cy = top + (bottom - top)/2
                    
                    y_diff = abs(prev_cp[1] - cy)

                    prev_cp[0] = cx
                    prev_cp[1] = cy

        # Chassis Position check condition
        if is_Cont == True and is_Truck == True:
            truck.is_chassis_front_pos(Cont_pt, Truck_Pt, 0)


        # boom barrier queue insert condition
        if (self.boom_barrier_line < cy) and (cap_h > cy) and (cx != 999999 and cy != 999999):
            boombarrier_queue.insert(0, True)

        # end queue insert condition
        if (((self.end_rect_center[1] - self.rect_ly) < cy and (self.end_rect_center[1] + self.rect_ly) > cy) and 
            ((self.end_rect_center[0] - self.rect_lx) < cx and (self.end_rect_center[0] + self.rect_lx) > cx)) and \
             (cx != 999999 and cy != 999999) and (y_diff < 10): 
            end_queue.insert(0, True)

        if (end_queue == [True, True]):
            truck.is_chassis_Length(cy, self.end_rect_center[1])
            truck.is_Door_Direction(truck.DD_true_count, truck.DD_false_count, 10)




        trimmed_imgs = []
        img_labels = []

        if truck_section == 'toprear':
            self.CP_toprear_detect = detections

            for idx in range(len(detections)):
                if detections[idx][0] == 'top' or detections[idx][0] == 'back' :
                    temp = darknet.im_trim(img, roi_pts[idx])

                    if temp is not None:
                        if detections[idx][0] == 'top':
                            img_labels.append('top')
                            truck.save_img('top', img, roi_pts[idx], 10)
                        elif detections[idx][0] == 'back':
                            img_labels.append('back')
                            truck.save_img('back', img, roi_pts[idx], 10)
                            
                        trimmed_imgs.append(temp)


        if len(detections) > 0:
            return True, detections, roi_pts, trimmed_imgs, img_labels, end_queue, boombarrier_queue, prev_cp
        else :
            return False, None, None, None, None, end_queue, boombarrier_queue, prev_cp
    
    def is_car_left(self, Tr_detections, Tr_pts, prev_truck):
        for idx in range(len(Tr_detections)) :
            if Tr_detections[idx][0] == 'Truck_back' :
                _, top, _, bottom = Tr_pts[idx]
                back_cy = top + (bottom-top)/2
                if (back_cy < self.end_to_line ) and (back_cy > self.end_from_line):
                    prev_truck = True

        return prev_truck
    
    def num_Identification(self, truck_section, trimmed_image):
        """ 
        This function is for recognition of Container ID.
        This function replaces the detection result with the correct form of Container ID via Container ID Rule.

        :input param: truck section(ex. left_U, top), trimmed CID images.
        :return: Container ID

        """
        output_result = None

        CPR_box = darknet.performDetect(trimmed_image, self.CIDR_net, self.CIDR_meta, self.CIDR_thres)

        if truck_section == "left_U" : 
            output_result = CP_rule('side', CPR_box, trimmed_image)

        elif truck_section == "right_U" : 
            output_result = CP_rule('side', CPR_box, trimmed_image)

        elif truck_section == "top" : 
            output_result = CP_rule('top', CPR_box, trimmed_image)

        elif truck_section == "back" :
            output_result = CP_rule('back', CPR_box, trimmed_image)

        if (output_result is not None) and (len(output_result) == 15 or len(output_result) == 11) \
            and str(list(output_result.keys())[0]).isalpha() and str(list(output_result.keys())[1]).isalpha() and \
                str(list(output_result.keys())[2]).isalpha() and str(list(output_result.keys())[3]).isalpha(): # 15- side, back, 11- top

            return output_result
        else :
            return None


class person(object) :
        def __init__(self, name) :
            self.name = name
        def __str__(self) :
            return self.name
        def __repr__(self) :
            return "'"+self.name+"'"


def CP_rule(CP_class, CP_data, trimmed_image):
    """ This method is for sorting detected character

    1. Remove duplicated character detected.
    2. Sort container ID in order by part of container
        -1. Distinguish whether container ID is horizontal or portrait.
        -2. After determining whether the character corresponds to the same line by character location, save the character for each line.
        -3. Sort the lines.
        -4. Sort the character in each line.

        **Exception handling
            - SIDE(left, right) in portrait container ROI
                - If the number of line is one or more than 2
                - If the number of character in first line is more than 11
                - If the number of character in second line is more than 4
                - If the number of character in first line is less than 9 and the number of character in second line is more than 9 

            - SIDE(left, right) in horizontal container ROI
                - If the number of line is one

            - TOP
                - If first character is number, this container ID is reversed one.

            - BACK
                -None

    :input param: CP_class - part of container that ID is detected
                  CP_data - detected ID that has label, confidence, and detected location
                  trimmed_image - detected container ID ROI image
    :return: sorted container ID

    """

    canditate_idxes = []
    remain_idxes = []
    remove_idxes = []
    for idx in range(len(CP_data)):
        max_idx, selected_CP_idx = select_CP_data(CP_data[idx], idx, CP_data, 7)
        remain_idxes.append(max_idx)
        if len(selected_CP_idx) > 1:
            canditate_idxes.extend(selected_CP_idx)
    
    remove_idxes = list(set(canditate_idxes))
    remain_idxes = list(set(remain_idxes))
    
    for i in remain_idxes : 
        if i == -1 : 
            remain_idxes.remove(-1)

    for idx in range(len(remain_idxes)):
        remove_idxes.remove(remain_idxes[idx])

    remove_idxes.sort(reverse=True)
    for idx_remove in range(len(remove_idxes)):
        del CP_data[remove_idxes[idx_remove]]


    recog_result = {}
    ruled_CP_data = CP_data
    line_threshold = 30
    if len(ruled_CP_data) ==0 :
        return None
    
    if CP_class == 'side':
        rows, cols,_ = trimmed_image.shape

        if rows > cols :
            is_portrait = True
        else : 
            is_portrait = False
        if is_portrait : 
            lines = []
            letters_in_a_line = []
            letters_in_a_line.append(ruled_CP_data[0])
            lines.append(letters_in_a_line)


            for i in range(1, len(ruled_CP_data)) :
                is_in_a_line = False
                for l in range(len(lines)):

                    # determine whether two characters is in same line or not
                    if abs(lines[l][0][2][0] - ruled_CP_data[i][2][0]) < line_threshold : # in the line\
                        lines[l].append(ruled_CP_data[i])
                        is_in_a_line = True
                        break
                if is_in_a_line == False :  # another line
                    letters_in_a_line = []
                    letters_in_a_line.append(ruled_CP_data[i])
                    lines.append(letters_in_a_line)

            if len(lines) < 2 or len(lines) > 2: return None


            # sort the lines
            lines.sort(key = lambda x : x[0][2][0])
            if len(lines[0]) < 9 and len(lines[1]) > 9:
                lines.reverse()

            if len(lines[0]) > 11 : return None


            # sort the characters in each line
            for l in range(len(lines)):
                lines[l].sort(key=lambda x : x[2][1])
            if len(lines[1]) >4:
                lines[1] = lines[1][0:4]
            

            for l in range(len(lines)):
                for t in range(len(lines[l])):
                    recog_result.update({person(str(lines[l][t][0].decode('utf-8'))) : lines[l][t][1]})
               

        if not is_portrait :  #landscape
            if len(ruled_CP_data) < 15 :
                return

            lines = []
            letters_in_a_line = []
            letters_in_a_line.append(ruled_CP_data[0])
            lines.append(letters_in_a_line)

            for i in range(1, len(ruled_CP_data)) :
                is_in_a_line = False
                for l in range(len(lines)):

                    # determine whether two characters is in same line or not
                    if abs(lines[l][0][2][1] - ruled_CP_data[i][2][1]) < line_threshold : # in the line\
                        lines[l].append(ruled_CP_data[i])
                        is_in_a_line = True
                        break
                if is_in_a_line == False :  # another line
                    letters_in_a_line = []
                    letters_in_a_line.append(ruled_CP_data[i])
                    lines.append(letters_in_a_line)
            if len(lines) < 2 : return None
            
            # sort the lines
            lines.sort(key = lambda x : x[0][2][1])
            
            # sort the characters in each line
            for l in range(len(lines)):
                lines[l].sort(key=lambda x : x[2][0])
                for t in range(len(lines[l])):
                    recog_result.update({person(str(lines[l][t][0].decode('utf-8'))) : lines[l][t][1]})


    elif CP_class == 'top':

        lines = []
        letters_in_a_line = []
        letters_in_a_line.append(ruled_CP_data[0])
        lines.append(letters_in_a_line)
        for i in range(1, len(ruled_CP_data)) :
            is_in_a_line = False

            for l in range(len(lines)):
                # determine whether two characters is in same line or not
                if abs(lines[l][0][2][1] - ruled_CP_data[i][2][1]) < line_threshold : # in the line\
                    lines[l].append(ruled_CP_data[i])
                    is_in_a_line = True
                    break
            if is_in_a_line == False :  # another line
                letters_in_a_line = []
                letters_in_a_line.append(ruled_CP_data[i])
                lines.append(letters_in_a_line)

        # sort the lines
        lines.sort(key = lambda x : x[0][2][1])
        
        # sort the characters in each line
        for l in range(len(lines)):
            lines[l].sort(key=lambda x : x[2][0])
            for t in range(len(lines[l])):
                recog_result.update({person(str(lines[l][t][0].decode('utf-8'))) : lines[l][t][1]})

        # check if container ID is reversed one
        if str(list(recog_result.keys())[0]).isdigit():
            reversed_result={}
            for k,v in recog_result.items():
                dict_element={k:v}
                dict_element.update(reversed_result)
                reversed_result=dict_element
            recog_result = reversed_result

    elif CP_class == 'back':

        lines = []
        letters_in_a_line = []
        letters_in_a_line.append(ruled_CP_data[0])
        lines.append(letters_in_a_line)

        for i in range(1, len(ruled_CP_data)) :
            is_in_a_line = False
            for l in range(len(lines)):
                
                # determine whether two characters is in same line or not
                if abs(lines[l][0][2][1] - ruled_CP_data[i][2][1]) < line_threshold : # in the line\
                    lines[l].append(ruled_CP_data[i])
                    is_in_a_line = True
                    break

            if is_in_a_line == False :  # another line
                letters_in_a_line = []
                letters_in_a_line.append(ruled_CP_data[i])
                lines.append(letters_in_a_line)
                
        # sort the lines
        lines.sort(key = lambda x : x[0][2][1])
        
        # sort the characters in each line
        for l in range(len(lines)):
            lines[l].sort(key=lambda x : x[2][0])
            for t in range(len(lines[l])):
                recog_result.update({person(str(lines[l][t][0].decode('utf-8'))) : lines[l][t][1]})

    return recog_result



def select_CP_data(cur_CPd, cur_idx, CP_data, dist_thresh):
    """ 
    
    This function is for eliminating duplicate detection results.

    :input param: cur_CPd - one of the detection results
                  cur_idx - the cur_CPd's index.
                  CP_data - the detection results.
                  dist_thresh - the detection results are removed when they are in the dist_thresh
    :return: max_idx - Most confident detection result's index among duplicate detection results
             duplicate_idx - duplicate detection result's index.

    """
    max_idx = -1
    duplicate_idx = []

    for idx in range(len(CP_data)):
        cx_len = CP_data[idx][2][0] - cur_CPd[2][0]
        cy_len = CP_data[idx][2][1] - cur_CPd[2][1]
        c_distance = math.sqrt((cx_len * cx_len) + (cy_len * cy_len))

        if c_distance < dist_thresh and cur_CPd[0] != CP_data[idx][0]:
            duplicate_idx.append(idx)

    max_confidance = cur_CPd[1]
    if len(duplicate_idx) > 0:
        max_idx = cur_idx
    for idx in range(len(duplicate_idx)):
        if CP_data[duplicate_idx[idx]][1] > max_confidance:
            max_confidance = CP_data[duplicate_idx[idx]][1]
            max_idx = duplicate_idx[idx]

    duplicate_idx.append(cur_idx)
    duplicate_idx.sort()

    if max_idx != -1:
        return max_idx, duplicate_idx
    else:
        return -1, duplicate_idx
