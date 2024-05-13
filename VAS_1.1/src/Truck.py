"""
This class is for saving attributes that each truck has.
For example, whether truck has twin container, detected ID list, whether detected ID is reliable.
"""

import ContainerPlate
import LicensePlate
import darknet
import cv2
import copy
import sys
from statistics import median

class Truck :

    Truck_Chassis_pos = None
    front_Chassis = []

    lp_candidates = []
    cp_candidates = []
    cp_candidates2 = []

    cp_candidates_left = []
    cp_candidates_left2 = []
    cp_candidates_right = []
    cp_candidates_right2 = []
    cp_candidates_top = []
    cp_candidates_top2 = []

    cp_candidates_back = []
    cp_candidates_back2 = []

    list_of_dic_LP = []
    list_of_dic_CID_left = []
    list_of_dic_CID_right = []
    list_of_dic_CID_top = []
    list_of_dic_CID_back = []
    list_of_dic_CID_left2 = []
    list_of_dic_CID_right2 = []
    list_of_dic_CID_top2 = []
    list_of_dic_CID_back2 = []
    is_precise_LPID = False
    is_precise_CID_left = False
    is_precise_CID_right = False
    is_precise_CID_top = False
    is_precise_CID_back = False
    is_precise_CID_left2 = False
    is_precise_CID_right2 = False
    is_precise_CID_top2 = False
    is_precise_CID_back2 = False

    containerID = []
    containerID2 = []
    LicenseID = []

    truck_front_flag = False
    front_img = None
    left_img = None
    right_img = None
    top_img = None
    rear_img = None

    left_img2 = None
    right_img2 = None
    top_img2 = None

    front_pt = []
    left_pt = []
    right_pt = []
    top_pt = []
    rear_pt = []
    
    left_pt2 = []
    right_pt2 = []
    top_pt2 = []



    num_gap_frames = 0

    is_twin_truck = False
    is_40ft_truck = None
    is_correct_DD = None
    truck_cnt = 0

    top= None
    debug = True


    def __new__(self):
        if not hasattr(self, 'instance'):
            self.instance = super(Truck, self).__new__(self)

        self.on_frame_cnt = 0
        self.pre_end_frame_cnt = 0
        self.start_end_frame_cnt = 0
        self.cam8_video = [] #f
        self.cam1_video = [] #r
        self.cam3_video = [] #l
        self.cam5_video = [] #tr



        self.cam1_det_info = []
        self.cam3_det_info = []

        # for stitching
        self.cam_Ldset = []
        self.cam_Luset = []
        self.cam_Rdset = []
        self.cam_Ruset = []
        self.cam_TRset = []

        self.det_Luset = [-1, [], -1]
        self.det_Ruset = [-1, [], -1]
        self.det_TRset = [-1, [], -1]

        self.cnt_Ru = 0
        self.cnt_Lu = 0
        self.cnt_Tr = 0

        self.cam_Lu_cadidates = []
        self.cam_Lu_startQ = []
        self.cam_Ru_cadidates = []
        self.cam_Ru_startQ = []
        # for stitching


        self.DD_true_count = 0
        self.DD_false_count = 0

        #for auto annotation
        self.detections_for_annotation = []
        self.roi_points_for_annotation = []
        self.images_for_annotation = []
        self.net_numbers = []
        self.times_for_annotation = []

        return self.instance



    def vote_lp_histogram(self):
        """ This function is for selecting appropriate License Plate ID at end signal

        Select License Plate ID that has max confidence in each ciphers
        :return: final License Plate ID   self.Truck.lp_candidates

        """
        LP=[]
        LP_Rate_set=[]

        for i in range(len(self.list_of_dic_LP)):
            lp_letter = max(self.list_of_dic_LP[i].keys(), key =(lambda k: self.list_of_dic_LP[i][k]))
            LP.append(lp_letter)
            LP_Rate_set.append(self.list_of_dic_LP[i][lp_letter])

        LP_Rate = int(sum(LP_Rate_set) / len(LP_Rate_set) * 100)

        return LP, LP_Rate

    def vote_cp_histogram2(self, is_second):
        """ This function is for selecting appropriate Container ID at end signal

        Select Container ID that has max confidence in each direction of truck and in each ciphers
        :input param: is second container of twin truck or not 
        :return: final Container ID

        """


        CID_max_list = [[{None:0} for col in range(4)] for row in range(15)]
        if is_second is False :
            for letter_i in range(len(self.list_of_dic_CID_left)):
                if len(self.list_of_dic_CID_left[letter_i]) != 0:
                    key_max = max(self.list_of_dic_CID_left[letter_i], key =(lambda k: self.list_of_dic_CID_left[letter_i][k]))
                    CID_max_list[letter_i][0] = {key_max: self.list_of_dic_CID_left[letter_i][key_max]}
                else:
                    CID_max_list[letter_i][0] = {None:0}

            for letter_i in range(len(self.list_of_dic_CID_right)):
                if len(self.list_of_dic_CID_right[letter_i]) != 0:
                    key_max = max(self.list_of_dic_CID_right[letter_i], key =(lambda k: self.list_of_dic_CID_right[letter_i][k]))
                    CID_max_list[letter_i][1] = {key_max: self.list_of_dic_CID_right[letter_i][key_max]}
                else:
                    CID_max_list[letter_i][1] = {None:0}

            for letter_i in range(len(self.list_of_dic_CID_top)):
                if len(self.list_of_dic_CID_top[letter_i]) != 0:
                    key_max = max(self.list_of_dic_CID_top[letter_i], key =(lambda k: self.list_of_dic_CID_top[letter_i][k]))
                    CID_max_list[letter_i][2] = {key_max: self.list_of_dic_CID_top[letter_i][key_max]}
                else:
                    CID_max_list[letter_i][2] = {None:0}

            for letter_i in range(len(self.list_of_dic_CID_back)):
                if len(self.list_of_dic_CID_back[letter_i]) != 0:
                    key_max = max(self.list_of_dic_CID_back[letter_i], key =(lambda k: self.list_of_dic_CID_back[letter_i][k]))
                    CID_max_list[letter_i][3] = {key_max: self.list_of_dic_CID_back[letter_i][key_max]}
                else:
                    CID_max_list[letter_i][3] = {None:0}


        else : 
            for letter_i in range(len(self.list_of_dic_CID_left2)):
                if len(self.list_of_dic_CID_left2[letter_i]) != 0:
                    key_max = max(self.list_of_dic_CID_left2[letter_i], key =(lambda k: self.list_of_dic_CID_left2[letter_i][k]))
                    CID_max_list[letter_i][0] = {key_max: self.list_of_dic_CID_left2[letter_i][key_max]}
                else:
                    CID_max_list[letter_i][0] = {None:0}
           
            for letter_i in range(len(self.list_of_dic_CID_right2)):
                if len(self.list_of_dic_CID_right2[letter_i]) != 0:
                    key_max = max(self.list_of_dic_CID_right2[letter_i], key =(lambda k: self.list_of_dic_CID_right2[letter_i][k]))
                    CID_max_list[letter_i][1] = {key_max: self.list_of_dic_CID_right2[letter_i][key_max]}
                else:
                    CID_max_list[letter_i][1] = {None:0}
          
            for letter_i in range(len(self.list_of_dic_CID_top2)):
                if len(self.list_of_dic_CID_top2[letter_i]) != 0:
                    key_max = max(self.list_of_dic_CID_top2[letter_i], key =(lambda k: self.list_of_dic_CID_top2[letter_i][k]))
                    CID_max_list[letter_i][2] = {key_max: self.list_of_dic_CID_top2[letter_i][key_max]}
                else:
                    CID_max_list[letter_i][2] = {None:0}
          
            for letter_i in range(len(self.list_of_dic_CID_back2)):
                if len(self.list_of_dic_CID_back2[letter_i]) != 0:
                    key_max = max(self.list_of_dic_CID_back2[letter_i], key =(lambda k: self.list_of_dic_CID_back2[letter_i][k]))
                    CID_max_list[letter_i][3] = {key_max: self.list_of_dic_CID_back2[letter_i][key_max]}
                else:
                    CID_max_list[letter_i][3] = {None:0}



        return self.select_CP_LastVoting(CID_max_list)




    def select_CP_LastVoting(self, twoDArray):
        
        CID = []
        CP_Rate_set = []

        try:
            for row_idx in range(len(twoDArray)):

                ### get the max CP
                temp_dic = {}
                for col_idx in range(len(twoDArray[0])):

                    if len(twoDArray[row_idx][col_idx]) != 0:
                        key = str( list(twoDArray[row_idx][col_idx].keys())[0] )
                        value = float( list(twoDArray[row_idx][col_idx].values())[0] )
                        if temp_dic.get(key) is None:
                            temp_dic[ key ] = value/4
                        else:
                            temp_dic[ key ] = temp_dic[key] + value/4

                max_key = max(temp_dic.keys(), key=(lambda k: temp_dic[k]))
                CID.append(max_key)


                ### calculate CP Rate
                rate_sum = 0
                rate_cnt = 0
                for col_idx in range(len(twoDArray[0])):
                    max_key_value = twoDArray[row_idx][col_idx].get(max_key)
                    if max_key_value is not None:
                        rate_sum += max_key_value
                        rate_cnt += 1

                CP_Rate_set.append(rate_sum/rate_cnt)
            CP_Rate = int(sum(CP_Rate_set) / len(CP_Rate_set) * 100)
        except:
            # CID = []
            # CP_Rate = None
            CID = []
            CP_Rate = 0

        return CID, CP_Rate


    def save_img(self, truck_section, img, roi_pt, border) :
        height, width, _ = img.shape
        
        ### front ###
        if truck_section == 'front':
            if (roi_pt[1] > (0+border) and roi_pt[3] < (height-border) and roi_pt[0] > (0+border) and roi_pt[2] < (width-border)):
                if self.front_img is None:
                    self.front_img = img
                    self.front_pt = roi_pt
                else:
                    newcx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, roi_pt)
                    origincx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, self.front_pt)

                    # better new
                    if (origincx_imgcx_diff > newcx_imgcx_diff):
                        self.front_img = img
                        self.front_pt = roi_pt
        elif truck_section == "front_not_lp": #if LPD not detected
            if (roi_pt[1] > (0+border) and roi_pt[3] < (height-border) and roi_pt[0] > (0+border) and roi_pt[2] < (width-border)):
                if self.front_img is None:
                    self.front_img = img
                    self.front_pt = roi_pt

        ### left ###
        elif truck_section == 'left_U' :
            if (roi_pt[1] > (0+border) and roi_pt[3] < (height-border) and roi_pt[0] > (0+border) and roi_pt[2] < (width-border)):
                
                if not self.is_twin_truck : 
                    if self.left_img is  None :
                        self.left_img = img
                        self.left_pt = roi_pt
                    else: 
                        newcx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, roi_pt)
                        origincx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, self.left_pt)

                        # better new
                        if (origincx_imgcx_diff > newcx_imgcx_diff):
                            self.left_img = img
                            self.left_pt = roi_pt

                else:
                    if self.left_img2 is  None :
                        self.left_img2 = img
                        self.left_pt2 = roi_pt
                    else: 
                        newcx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, roi_pt)
                        origincx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, self.left_pt2)

                        # better new
                        if (origincx_imgcx_diff > newcx_imgcx_diff):
                            self.left_img2 = img
                            self.left_pt2 = roi_pt

        ### right ###
        elif truck_section == 'right_U' :
            if (roi_pt[1] > (0+border) and roi_pt[3] < (height-border) and roi_pt[0] > (0+border) and roi_pt[2] < (width-border)):
                
                if not self.is_twin_truck : 
                    if self.right_img is  None :
                        self.right_img = img
                        self.right_pt = roi_pt
                    else: 
                        newcx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, roi_pt)
                        origincx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, self.right_pt)

                        # better new
                        if (origincx_imgcx_diff > newcx_imgcx_diff):
                            self.right_img = img
                            self.right_pt = roi_pt

                else:
                    if self.right_img2 is  None :
                        self.right_img2 = img
                        self.right_pt2 = roi_pt
                    else: 
                        newcx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, roi_pt)
                        origincx_imgcx_diff, _ = self.calc_cxcy_diff(width/2, height/2, self.right_pt2)

                        # better new
                        if (origincx_imgcx_diff > newcx_imgcx_diff):
                            self.right_img2 = img
                            self.right_pt2 = roi_pt

        ### top ###
        elif truck_section == 'top' :
            if (roi_pt[1] > (0+border) and roi_pt[3] < (height-border) and roi_pt[0] > (0+border) and roi_pt[2] < (width-border)):
                
                if not self.is_twin_truck : 
                    if self.top_img is  None :
                        self.top_img = img
                        self.top_pt = roi_pt
                    else: 
                        _, newcy_imgcy_diff = self.calc_cxcy_diff(width/2, height*2/3, roi_pt)
                        _, origincy_imgcy_diff = self.calc_cxcy_diff(width/2, height*2/3, self.top_pt)

                        # better new
                        if (origincy_imgcy_diff > newcy_imgcy_diff):
                            self.top_img = img
                            self.top_pt = roi_pt

                else:
                    if self.top_img2 is  None :
                        self.top_img2 = img
                        self.top_pt2 = roi_pt
                    else: 
                        _, newcy_imgcy_diff = self.calc_cxcy_diff(width/2, height*2/3, roi_pt)
                        _, origincy_imgcy_diff = self.calc_cxcy_diff(width/2, height*2/3, self.top_pt2)

                        # better new
                        if (origincy_imgcy_diff > newcy_imgcy_diff):
                            self.top_img2 = img
                            self.top_pt2 = roi_pt

        ### rear ###
        elif truck_section == 'back' :
            if (roi_pt[1] > (0+border) and roi_pt[3] < (height-border) and roi_pt[0] > (0+border) and roi_pt[2] < (width-border)):
                if self.rear_img is None:
                    self.rear_img = img
                    self.rear_pt = roi_pt
                else: 
                    _, newcy_imgcy_diff = self.calc_cxcy_diff(width/2, height/2, roi_pt)
                    _, origincy_imgcy_diff = self.calc_cxcy_diff(width/2, height/2, self.rear_pt)

                    # better new
                    if (origincy_imgcy_diff > newcy_imgcy_diff):
                        self.rear_img = img
                        self.rear_pt = roi_pt



    def find_ChassisPosition(self, TF2CF_threshold):

        ### FA / two cont
        if self.is_twin_truck:
            self.Truck_Chassis_pos = "FA"
            return self.Truck_Chassis_pos

        ### E / no cont
        if len(self.containerID) == 0:
            self.Truck_Chassis_pos = "E"
            return self.Truck_Chassis_pos

        ### 20ft C / CL is short
        if self.is_40ft_truck is False:
            self.Truck_Chassis_pos = "C"
            return self.Truck_Chassis_pos

        ### 40ft C / long CL and ISO first digit
        if self.is_40ft_truck : # self.containerID[0][-4] == "4":
            self.Truck_Chassis_pos = "C"
            return self.Truck_Chassis_pos

        ### "A", "M" / find CB2TB_has_Gap info from cam5
        CB2TB_has_Gap = self.find_CB2TB_has_Gap_from_TopRear()
        if CB2TB_has_Gap == None: return None
        else:
            if not CB2TB_has_Gap:
                self.Truck_Chassis_pos = "A"
            else:
                ### "A", "C" / find TB2CF_has_Gap info from cam1, cam3 / Input: detection list, list search kernel, TF2CF ratio threashold / Output: TF2CFratio
                TF2CF1_has_Gap, ratio_1set = self.find_TF2CF_has_Gap_from_side(self.cam1_det_info, 3, TF2CF_threshold)
                TF2CF3_has_Gap, ratio_3set = self.find_TF2CF_has_Gap_from_side(self.cam3_det_info, 3, TF2CF_threshold)
                TF2CF_has_Gap = None

                if TF2CF1_has_Gap is None and TF2CF3_has_Gap is None:
                    TF2CF_has_Gap = None
                else:
                    if TF2CF1_has_Gap is None:
                        TF2CF_has_Gap =  TF2CF3_has_Gap
                    elif TF2CF3_has_Gap is None:
                        TF2CF_has_Gap = TF2CF1_has_Gap
                    else:
                        if TF2CF1_has_Gap == TF2CF3_has_Gap:
                            TF2CF_has_Gap = TF2CF1_has_Gap
                        else:
                            print("Err: left right C_Pos is different!")
                            TF2CF_has_Gap = TF2CF1_has_Gap #None

                if TF2CF_has_Gap is not None:
                    if TF2CF_has_Gap == True: 
                        self.Truck_Chassis_pos = "M"
                    else: 
                        self.Truck_Chassis_pos = "F"
                else:
                    print("C_pos TF2CF_has_Gap None err")
                    self.Truck_Chassis_pos = None

        return self.Truck_Chassis_pos





    def find_TF2CF_has_Gap_from_side(self, detections, kernel_width, threshold):
        '''
        calc the compressed info from side camera using kernel
        kernel is marked from selected classes (0: Truck_head / 1: Cont_border)
        '''
        compress_detections = []
        for idx in range(len(detections) - kernel_width + 1):
            side_hist = {'Truck_head':0 ,'Cont_border':0}

            # make side hist using kernel
            for iter_idx in range(kernel_width):
                for elem in detections[idx + iter_idx]:
                    if elem == 'Truck_head' or elem == 'Cont_border':
                        side_hist[elem] = side_hist[elem] + 1

            # kernel compression from histogram: lagest number, else, same then left idx from side_hist
            kernel_idx = self.get_maxIdx_dict_histogram(side_hist)
            compress_detections.append(kernel_idx)

        compress_detections = self.cvt_border2frontback(compress_detections)
        # print("after compress_detections: ", compress_detections)

        TruckF2ContF_ratio = self.calc_medianIdx_from_compressedList(compress_detections)
        # print("TruckF2ContF_ratio: ", TruckF2ContF_ratio)
        if TruckF2ContF_ratio is None:
            return None, None

        if threshold < TruckF2ContF_ratio:
            return True, TruckF2ContF_ratio #"gap"
        else:
            return False, TruckF2ContF_ratio #"no gap"



    def cvt_border2frontback(self, compress_detections):

        first_1_flag = False
        indent_flag = False

        current_elem = None
        current_count = 0

        for idx in range(len(compress_detections)):
            elem = compress_detections[idx]

            if elem != current_elem:
                current_elem = elem
                current_count = 0
            else:
                current_count += 1


            if (current_elem == 1) and (first_1_flag == False): first_1_flag = True
            if (current_elem == -1) and (first_1_flag == True) and (indent_flag == False): indent_flag = True
            if (current_elem == 1) and (first_1_flag == True) and (indent_flag == True): 
                compress_detections[idx] = 2

        return compress_detections






    def get_maxIdx_dict_histogram(self, dict_histogram):
        max_value = 0
        max_Idx = -1
        list_histogram = list(dict_histogram.items())
        for idx in range(len(list_histogram)):
            
            if list_histogram[idx][1] > max_value:
                max_Idx = idx

        return max_Idx


    def calc_medianIdx_from_compressedList(self, compressed_list):
        idx_list_Truck_head = []
        idx_list_Cont_front = []
        idx_list_Cont_back = []
        for idx, value in enumerate(compressed_list):

            if value == 0:
                idx_list_Truck_head.append(idx)
            elif value == 1:
                idx_list_Cont_front.append(idx)
            elif value == 2:
                idx_list_Cont_back.append(idx)

        if len(idx_list_Truck_head) != 0 and len(idx_list_Cont_front) != 0 and len(idx_list_Cont_back) != 0:
            Truck_head_Idx = median(idx_list_Truck_head)
            Cont_front_Idx = median(idx_list_Cont_front)
            Cont_back_Idx = median(idx_list_Cont_back)

            # print(Truck_head_Idx, " ", Cont_front_Idx, " ", Cont_back_Idx)
            return (Cont_front_Idx-Truck_head_Idx) / (Cont_back_Idx-Truck_head_Idx)

        else:
            print("Err: calc list idx from compressed list")
            return None
    
    def find_CB2TB_has_Gap_from_TopRear(self):
        fore_count = self.front_Chassis.count(True)
        back_count = self.front_Chassis.count(False)

        if fore_count > back_count:
            return True
        else:
            return False




    def calc_cxcy_diff(self, x_stand, y_stand, roi_pt):
        cx = roi_pt[0] + (roi_pt[2] - roi_pt[0])/2
        cy = roi_pt[1] + (roi_pt[3] - roi_pt[1])/2
        
        cx_imgcx_diff = abs(x_stand - cx) 
        cy_imgcy_diff = abs(y_stand - cy)

        return  cx_imgcx_diff, cy_imgcy_diff




    def isTwinTruck(self, is_gap):
        """ This function is for checking if this truck has twin container

        If the number of detected gap of two container is more than threshold, it has twin container. Otherwise, it has single container.
        :result variable: self.is_twin_truck

        """
        if is_gap is True:
            self.num_gap_frames +=1
        else :
            if self.num_gap_frames >= 3 :
                self.is_twin_truck = True
        
        
    def is_chassis_Length(self, cy, thres):
        if cy > thres: 
            self.is_40ft_truck = True
        else:
            self.is_40ft_truck = False

    def is_Door_Direction(self, c_front, c_back, threshold):
        if c_front > threshold or c_back > threshold:
            if c_front > c_back:
                self.is_correct_DD = True
            else:
                self.is_correct_DD = False

    def is_chassis_front_pos(self, Cont_pt, Truck_Pt, threshold):
        Truck_cp = (Truck_Pt[3] - Truck_Pt[1])/2
        Cont_cp = (Cont_pt[3] - Cont_pt[1])/2
        Cont2Truck_indent = (( Truck_cp - Cont_cp ) - ( Truck_cp - Truck_Pt[1] + Cont_pt[3] - Cont_cp ))  /  ( Truck_cp - Cont_cp )
        if Cont2Truck_indent > threshold:
            # print("�", Cont2Truck_indent)        
            self.front_Chassis.append(False)
        else:
            # print("L", Cont2Truck_indent)        
            self.front_Chassis.append(True)

            
    def hasPreciseID(self, id_, truck_section):
        """ This function is for checking if it has reliable ID or not for license plate and each part of container at every frame that ID is detected.

        For license plate and each part of container
        1. Make ID list that has average confidence 
        2. Average confidence is number that divided by the number of detected frames for the accumulated confidence of a specific id character.
        3. Check if ID list is reliable with confidence of whether average confidence and the number of candidates is more than threshold value.

        :input param: detected ID, truck direction that ID detected
        :result variable: self.is_precise_LPID, 
                            self.is_precise_CID_left, self.is_precise_CID_left2, 
                            self.is_precise_CID_right, self.is_precise_CID_right2
                            self.is_precise_CID_top, self.is_precise_CID_top2, 
                            self.is_precise_CID_back, self.is_precise_CID_back2

        """
        list_of_dic = []
        id_candidates = []
        if truck_section == 'front':
            list_of_dic = self.list_of_dic_LP
            id_candidates = self.lp_candidates
        elif truck_section == 'left' : 
            list_of_dic = self.list_of_dic_CID_left
            id_candidates = self.cp_candidates_left
        elif truck_section == 'left2' : 
            list_of_dic = self.list_of_dic_CID_left2
            id_candidates = self.cp_candidates_left2
        elif truck_section == 'right' : 
            list_of_dic = self.list_of_dic_CID_right
            id_candidates = self.cp_candidates_right
        elif truck_section == 'right2' : 
            list_of_dic = self.list_of_dic_CID_right2
            id_candidates = self.cp_candidates_right2
        elif truck_section == 'top' : 
            list_of_dic = self.list_of_dic_CID_top
            id_candidates = self.cp_candidates_top
        elif truck_section == 'top2' : 
            list_of_dic = self.list_of_dic_CID_top2
            id_candidates = self.cp_candidates_top2
        elif truck_section == 'back' : 
            list_of_dic = self.list_of_dic_CID_back
            id_candidates = self.cp_candidates_back
        elif truck_section == 'back2' : 
            list_of_dic = self.list_of_dic_CID_back2
            id_candidates = self.cp_candidates_back2
        
 
        # recursive average
        for letter_i in range(len(id_)):
            
            if letter_i >= len(list_of_dic):
                if self.rule_verification( str(list(id_.keys())[letter_i]), letter_i , truck_section):
                    init_dic = {str(list(id_.keys())[letter_i]) : list(id_.values())[letter_i] / len(id_candidates) }
                    list_of_dic.append(init_dic)
                else:
                    list_of_dic.append({None:0})


            else : 
                if self.rule_verification( str(list(id_.keys())[letter_i]), letter_i , truck_section):
                    list_of_dic[letter_i] = {key: value * (len(id_candidates)-1) / len(id_candidates) for key, value in list_of_dic[letter_i].items()}
                    if str(list(id_.keys())[letter_i]) in list_of_dic[letter_i] : 
                        list_of_dic[letter_i][str(list(id_.keys())[letter_i])] += list(id_.values())[letter_i] / len(id_candidates)
                    else :
                        list_of_dic[letter_i][str(list(id_.keys())[letter_i])] = list(id_.values())[letter_i] / len(id_candidates)
                else:
                    continue

        return


    def rule_verification(self, char, idx, truck_section):

        if truck_section == 'front': return True
        else:
            if idx == 0 or idx == 1 or idx == 2 or idx == 13: return char.isalpha()
            
            elif idx == 3:
                if char == 'U' or char == 'J' or char == 'Z': return True
                else: return False
            
            elif idx == 4 or idx == 5 or idx == 6 or idx == 7 or idx == 8 or idx == 9 or idx == 10: return char.isdigit()
            else: return True



    def delete_info(self):
        self.on_frame_cnt = 0
        self.pre_end_frame_cnt = 0
        self.start_end_frame_cnt = 0
        self.cam8_video = [] #f
        self.cam1_video = [] #r
        self.cam3_video = [] #l
        self.cam5_video = [] #tr

        #for auto annotation
        self.detections_for_annotation = []
        self.roi_points_for_annotation = []
        self.images_for_annotation = []
        self.net_numbers = []
        self.times_for_annotation = []

        self.cam1_det_info = []
        self.cam3_det_info = []


        # for stitching
        self.cam_Ldset = []
        self.cam_Luset = []
        self.cam_Rdset = []
        self.cam_Ruset = [] 
        self.cam_TRset = [] 

        self.det_Luset = [-1, [], -1]
        self.det_Ruset = [-1, [], -1]
        self.det_TRset = [-1, [], -1]

        self.cam_Lu_cadidates = []
        self.cam_Lu_startQ = []
        self.cam_Ru_cadidates = []
        self.cam_Ru_startQ = []
        # for stitching


        self.front_Chassis = []
        self.Truck_Chassis_pos = None
        
        self.containerID = []
        self.containerID2 = []
        self.LicenseID = []

        self.cp_candidates = []
        self.cp_candidates_top = []
        self.cp_candidates_back = []
        self.cp_candidates_left = []
        self.cp_candidates_right = []
        self.cp_candidates2 = []
        self.cp_candidates_top2 = []
        self.cp_candidates_back2 = []
        self.cp_candidates_left2 = []
        self.cp_candidates_right2 = []
 
        self.lp_candidates = []
        self.is_twin_truck = False
        self.is_40ft_truck = None
        self.is_correct_DD = None

        self.DD_true_count = 0
        self.DD_false_count = 0

        self.truck_front_flag = False
        self.front_img = None
        self.left_img = None
        self.right_img = None
        self.top_img = None
        self.rear_img = None

        self.left_img2 = None
        self.right_img2 = None
        self.top_img2 = None


        self.front_pt = []
        self.left_pt = []
        self.right_pt = []
        self.top_pt = []
        self.rear_pt = []

        self.left_pt2 = []
        self.right_pt2 = []
        self.top_pt2 = []



        self.num_gap_frames = 0
        self.list_of_dic_LP = []
        self.list_of_dic_CID_left = []
        self.list_of_dic_CID_right = []
        self.list_of_dic_CID_top = []
        self.list_of_dic_CID_back = []
        self.list_of_dic_CID_left2 = []
        self.list_of_dic_CID_right2 = []
        self.list_of_dic_CID_top2 = []
        self.list_of_dic_CID_back2 = []
        self.is_precise_LPID = False
        self.is_precise_CID_left = False
        self.is_precise_CID_right = False
        self.is_precise_CID_top = False
        self.is_precise_CID_back = False
        self.is_precise_CID_left2 = False
        self.is_precise_CID_right2 = False
        self.is_precise_CID_top2 = False
        self.is_precise_CID_back2 = False
        import gc
        gc.collect()

