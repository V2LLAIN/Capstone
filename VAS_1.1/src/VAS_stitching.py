import cv2
import os
from ctypes import *
import math
import random
import numpy as np
import time
import json
import darknet
import time
import random
import sys
import statistics
import multiprocessing as mp


import VAS_manager

class VAS_stitching:

    def __init__(self, config_dir, cut_unit, HL, HR, HL_inv, HR_inv):
        self.L_K, self.L_D = self.load_distorMat(config_dir)
        self.HL = HL
        self.HR = HR
        self.HL_inv = HL_inv
        self.HR_inv = HR_inv

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

        self.unit = cut_unit




    def load_distorMat(self, dir):
        json_data=open(dir, encoding='utf-8').read()
        data_template = json.loads(json_data)

        L_K = np.load(data_template['Intrinsic_matrix'])
        L_D = np.load(data_template['Distortion_matrix'])

        return L_K, L_D


    def set_imageSet(self, right_up_img, right_down_img, left_up_img, left_down_img):
        left_up_img = cv2.remap(left_up_img, self.mMap_x, self.mMap_y, cv2.INTER_LINEAR)
        left_down_img = cv2.remap(left_down_img, self.mMap_x, self.mMap_y, cv2.INTER_LINEAR)
        right_up_img = cv2.remap(right_up_img, self.mMap_x, self.mMap_y, cv2.INTER_LINEAR )
        right_down_img = cv2.remap(right_down_img, self.mMap_x, self.mMap_y, cv2.INTER_LINEAR)

        self.cam_Luset.append(left_up_img)
        self.cam_Ldset.append(left_down_img)
        self.cam_Ruset.append(right_up_img)
        self.cam_Rdset.append(right_down_img)


    def set_Stitching_Info(self, width, height):
        self.L_w = int(width)
        self.L_h = int(height)

        # make distortion table
        self.mMap_x, self.mMap_y = cv2.initUndistortRectifyMap(self.L_K, self.L_D, None, self.L_K, (self.L_w, self.L_h), 5)

        return self.mMap_x, self.mMap_y


    def get_container_size_info(self, truck_iso, is_twin_truck, is_40ft_truck):
        print('truck_iso: ', truck_iso, ' is_twin_truck: ', is_twin_truck, ' is_40ft_truck: ', is_40ft_truck)
        if len(truck_iso[0]) < 4:
            return 4
        else:
            ### 20 twin
            if is_twin_truck:
                return 3
            ### 45ft
            elif is_40ft_truck and truck_iso[0][-4] == "4" and truck_iso[0][-3] == "5":
                return 2
            ### 40ft
            elif is_40ft_truck and truck_iso[0][-4] == "4" :
                return 2
            ### 20 single
            else:
                return 4

    def do_Stitching(self, cam1_images, cam2_images, cam3_images, cam4_images, cam5_images, det_Ru, det_Lu, \
        rear_img, HTr, is_twin_truck, seal_module, net_info, cont_size_switch):
        '''
        Stitching main function
        '''

        Stitching_Start = time.time()
        MP_flag = False
        # Stitching_Start = time.time()
        if MP_flag:
            Mngr = mp.Manager()
            stitching_result = Mngr.list([None, None, None, None, None, None, None])
        else:
            stitching_result = [None, None, None, None, None, None, None]


        ### for rear side detection ###
        img_HTr = None
        if rear_img is not None:
            img_HTr= self.make_HTr(rear_img, HTr, seal_module)
            if img_HTr is not None:
                stitching_result[0] = img_HTr
        if False:
            if img_HTr is not None:
                img_HTr_resize = cv2.resize(img_HTr, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("img_HTr_resize", img_HTr_resize)

        ### for top stitching H matrix ###
        # top_view_H = self.make_Top_view_H(int(rear_img.shape[0]), int(rear_img.shape[1]))


        ### do sequence alignment ###
        try:
            R_up_Info, R_down_Info, R_is_Front = self.sequence_alignment(cam1_images, cam2_images, det_Ru, net_info, 0)
            L_up_Info, L_down_Info, L_is_Front =  self.sequence_alignment(cam3_images, cam4_images, det_Lu, net_info, 1)
        except:
            print("!!! Sequence Alignment Err !!!")


        if MP_flag:
            RightStitching_process = mp.Process(name="RightStitching_Process", target=self.make_stitched_image_Right, args=(stitching_result, cam1_images, cam2_images, R_up_Info, R_down_Info, is_twin_truck, ))
            LeftStitching_process = mp.Process(name="LeftStitching_Process", target=self.make_stitched_image_Left, args=(stitching_result, cam3_images, cam4_images, L_up_Info, L_down_Info, is_twin_truck, ))
            RightStitching_process.daemon = True
            LeftStitching_process.daemon = True

            RightStitching_process.start()
            LeftStitching_process.start()

            RightStitching_process.join()
            LeftStitching_process.join()

            RightStitching_process.close()
            LeftStitching_process.close()
        else:
            self.make_stitched_image_Right(stitching_result, cam1_images, cam2_images, R_up_Info, R_down_Info, is_twin_truck, R_is_Front, cont_size_switch)
            self.make_stitched_image_Left(stitching_result, cam3_images, cam4_images, L_up_Info, L_down_Info, is_twin_truck, L_is_Front, cont_size_switch)
            # self.do_top_stitching(cam5_images, top_view_H, cont_size_switch, start_idx, mid_idx, end_idx, is_twin_truck)


        print(f"\tTotal do stitching processing time: {round(time.time() - Stitching_Start, 2)} [s]")
        return stitching_result

    def do_Stitching_BCT(self, cam1_images, cam3_images, cam5_images, det_Ru, det_Lu, det_TR, \
        rear_img, HTr, is_twin_truck, seal_module, cont_size_switch):
        '''
        Stitching main function
        '''

        '''
        stitching_result = [0] img_HTR (Top Rear Seal presence Image)
                           [1] Right stitched image (40ft or twin front)
                           [2] Left stitched image (40ft or twin front)
                           [3] Top stitched image (40ft or twin front)
                           [4] Right stitched image (twin back)
                           [5] Left stitched image (twin back)
                           [6] Top stitched image (twin back)
        '''


        Stitching_Start = time.time()
        stitching_result = [None, None, None, None, None, None, None]



        ### for rear side detection ###
        img_HTr = None
        if rear_img is not None:
            img_HTr= self.make_HTr(rear_img, HTr, seal_module)
            if img_HTr is not None:
                stitching_result[0] = img_HTr
        if False:
            if img_HTr is not None:
                img_HTr_resize = cv2.resize(img_HTr, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("img_HTr_resize", img_HTr_resize)
                cv2.waitKey(0)

        ### for top stitching H matrix ###
        try:
            top_view_H = self.make_Top_view_H(int(rear_img.shape[0]), int(rear_img.shape[1]))
        except:
            print('top rear error')

        try:
            if det_Ru[1] and is_twin_truck: # twin
                right_ret, right_stitched_side = self.do_side_stitching(cam1_images, cont_size_switch, det_Ru[0], int(np.median(det_Ru[1])), det_Ru[2], is_twin_truck, 2)
                stitching_result[1] = right_stitched_side[1]
                stitching_result[4] = right_stitched_side[2]
            else:
                right_ret, right_stitched_side = self.do_side_stitching(cam1_images, cont_size_switch, det_Ru[0], -1, det_Ru[2], is_twin_truck, 2)
                stitching_result[1] = right_stitched_side[0]
        except:
            print("right stitching err")

        try:
            if det_Lu[1] and  is_twin_truck:
                left_ret, left_stitched_side = self.do_side_stitching(cam3_images, cont_size_switch, det_Lu[0], int(np.median(det_Lu[1])), det_Lu[2], is_twin_truck, 1)
                stitching_result[2] = left_stitched_side[1]
                stitching_result[5] = left_stitched_side[2]
            else:
                left_ret, left_stitched_side = self.do_side_stitching(cam3_images, cont_size_switch, det_Lu[0], -1, det_Lu[2], is_twin_truck, 1)
                stitching_result[2] = left_stitched_side[0]
        except:
            print("left stitching err")

        try:
            if det_TR[1] and is_twin_truck:
                toprear_ret, toprear_stitched = self.do_top_stitching(cam5_images, top_view_H, cont_size_switch, det_TR[0], int(np.median(det_TR[1])), det_TR[2], is_twin_truck)
                stitching_result[3] = toprear_stitched[1]
                stitching_result[6] = toprear_stitched[2]
            else:
                toprear_ret, toprear_stitched = self.do_top_stitching(cam5_images, top_view_H, cont_size_switch, det_TR[0], -1, det_TR[2], is_twin_truck)
                stitching_result[3] = toprear_stitched[0]
        except:
            print("toprear stitching err")

        # if not is_twin_truck:
        #     if right_ret and right_stitched_side:
        #         stitching_result[1] = right_stitched_side[0]
        #     if left_ret and left_stitched_side:
        #         stitching_result[2] = left_stitched_side[0]
        # else:
        #     if right_ret and right_stitched_side:
        #         stitching_result[1] = right_stitched_side[1]
        #         stitching_result[4] = right_stitched_side[2]
        #     if left_ret and left_stitched_side:
        #         stitching_result[2] = left_stitched_side[1]
        #         stitching_result[5] = left_stitched_side[2]


        print(f"\tTotal do stitching processing time: {round(time.time() - Stitching_Start, 2)} [s]")
        return stitching_result

    '''
    make homographic matrix for top stitching
    '''
    def make_Top_view_H(self, h, w):

        point_list = [(673, 456) , (1853, 480) , (20, 993) , (2450, 1073)]
        kp1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])

        kp2 = np.array([[w/8, h/10*2],
                            [w/8*7, h/10*2],
                            [w/8, h/10*9],
                            [w/8*7, h/10*9]])

        # M = cv2.getPerspectiveTransform(kp1,kp2)
        H, _ = cv2.findHomography(kp1, kp2, cv2.RANSAC,4.0)

        return H

    def make_Left_view_H(self, h, w):

        point_list = [(243, 310) , (810, 316) , (216, 1606) , (773, 1646)]
        kp1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
        # kp2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        kp2 = np.array([[w/8*1, h/10*1],
                            [w/8*7, h/10*1],
                            [w/8*1, h/10*9],
                            [w/8*7, h/10*9]])

        # M = cv2.getPerspectiveTransform(kp1,kp2)
        H, _ = cv2.findHomography(kp1, kp2, cv2.RANSAC,4.0)

        return H

    def make_Right_view_H(self, h, w):

        point_list = [(243, 310) , (810, 316) , (216, 1606) , (773, 1646)]
        kp1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])

        kp2 = np.array([[w/8, h/10*2],
                            [w/8*7, h/10*2],
                            [w/8, h/10*9],
                            [w/8*7, h/10*9]])

        # M = cv2.getPerspectiveTransform(kp1,kp2)
        H, _ = cv2.findHomography(kp1, kp2, cv2.RANSAC,4.0)

        return H

    def make_HTr(self, rear_img, H, seal_module):
        img_HTr = cv2.warpPerspective(rear_img, H, (rear_img.shape[1], rear_img.shape[0]))
        img_HTr = img_HTr[int(img_HTr.shape[0]/20*1):int(img_HTr.shape[0]/19*20), int(img_HTr.shape[1]/15):int(img_HTr.shape[1]/14*15)]
        # img_HTr, seal_presence = seal_module.seal_detection(img_HTr) #seal detection disabled

        # return img_HTr, seal_presence
        return img_HTr

    def imgs_origin2distort(self, imgs, mMap_x, mMap_y):
        for idx in range(len(imgs)):

            imgs[idx] = cv2.remap(imgs[idx], mMap_x, mMap_y, cv2.INTER_LINEAR)



    '''''''''''''''''''''''''''''''''''''''
    For RIGHT (1, 2) camera
    '''''''''''''''''''''''''''''''''''''''
    def make_stitched_image_Right(self, stitching_result, up_images, down_images, up_Info, down_Info, is_twin_truck, is_Front, cont_size_switch):
        print("PID Right: ", os.getpid(), " ", mp.current_process())
        yminoffset = 400
        ymaxoffset = 300
        is_camera_right = 0


        print("\tRIHGT STITCHING INPUT:: (up_images)", len(up_images), " , (down_images)", len(down_images))

        ### for top images
        try:
            rotate_images(up_images, is_camera_right, self.L_w, self.L_h, yminoffset, ymaxoffset)
            rotate_images(down_images, is_camera_right, self.L_w, self.L_h, yminoffset, ymaxoffset)
            if False:
                for img in up_images:
                    img_resize = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("upimg_resize", img_resize)
                    if cv2.waitKey(0) == ord('q'): break

        except:
            print("!!! RIGHT Up Info Err !!!")


        ### find start - end imgs ### vertical_lines
        try:
            # up_croped_set, down_croped_set, diff_x, centor_dist = make_cropped_updownSet(up_images, down_images, up_Info, down_Info)
            up_croped_set, down_croped_set, diff_x, centor_dist = make_cropped_updownSet_no_translate(up_images, down_images, up_Info, down_Info, is_Front)
            if len(up_croped_set) == 0 or len(down_croped_set) == 0:
                print("Fail to make cropped updown sets")
                return None

        except:
            print("!!! RIGHT Find start-end Err !!!")

        # -debug
        if False:
            up_s = cv2.resize(up_croped_set[0], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            up_e = cv2.resize(up_croped_set[-1], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("up_test start", up_s)
            cv2.imshow("up_test end", up_e)

            down_s = cv2.resize(down_croped_set[0], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            down_e = cv2.resize(down_croped_set[-1], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("down_test start", down_s)
            cv2.imshow("down_test end", down_e)
            cv2.waitKey(0)

        # -debug
        if False:
            up_idx = 1
            print(len(up_croped_set))
            for img in up_croped_set:
                img_resize = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("up_cropped_set", img_resize)
                # cv2.imwrite('/home/dpw/Desktop/VAS/VAS/up/' + str(up_idx) + '.jpg', img)
                up_idx += 1
                if cv2.waitKey(0) == ord('q'): break

            down_idx = 1
            print(len(down_croped_set))
            for img in down_croped_set:
                img_resize = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("up_cropped_set", img_resize)
                # cv2.imwrite('/home/dpw/Desktop/VAS/VAS/down/' + str(down_idx) + '.jpg', img)
                down_idx += 1
                if cv2.waitKey(0) == ord('q'): break


        ### !container stitching (effected: cam_idx, Container_idx) ###
        try:
            stitched_img_up, stitched_img_down, avg_velocity = filtered_stitching_right(up_croped_set, down_croped_set, self.unit, diff_x, 0.4, cont_size_switch)
            if stitched_img_up is None or stitched_img_down is None or avg_velocity is None:
                return None

        except:
            print("!!! RIGHT Side Stitching Err !!!")

        # -debug
        if False:
            stitched_img_up_resize = cv2.resize(stitched_img_up, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            stitched_img_down_resize = cv2.resize(stitched_img_down, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("stitched_img_up_resize", stitched_img_up_resize)
            cv2.imshow("stitched_img_down_resize", stitched_img_down_resize)
            cv2.waitKey(0)




        ### do hole filling using closing ###
        try:
            val = 1
            val = val * 2 + 1
            kernelNDArray = np.ones((val, val), np.uint8)
            stitched_img_up = cv2.morphologyEx(stitched_img_up, cv2.MORPH_CLOSE, kernelNDArray)
            stitched_img_down = cv2.morphologyEx(stitched_img_down, cv2.MORPH_CLOSE, kernelNDArray)

        except:
            print("!!! RIGHT Hole filling Err !!!")

        # -debug
        if False:
            stitched_up = cv2.resize(stitched_img_up, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            stitched_down = cv2.resize(stitched_img_down, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("stitched_up", stitched_up)
            cv2.imshow("stitched_down", stitched_down)
            cv2.waitKey(0)




        ### !up-down stitching (effected: Container_idx) ###
        try:
            stitched_image = updown_Stitching(stitched_img_up, stitched_img_down, avg_velocity, 0.8, 0.8)

        except:
            print("!!! RIGHT Up-side Stitching Err !!!")
            stitched_image = None

        # -debug
        if False:
            stitched_image_resize = cv2.resize(stitched_image, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("stitched_image", stitched_image_resize)



        ### !stitched image twin division ###
        stitch_result = []
        try:
            stitch_result = self.if_twin_division(stitched_image, centor_dist, avg_velocity, up_croped_set[0].shape[1], is_twin_truck)

        except:
            print("!!! twin division Err !!!")

        # -debug
        if False:
            if is_twin_truck:
                stitched_image_Front_resize = cv2.resize(stitch_result[0], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("stitched_image_Front", stitched_image_Front_resize)
                stitched_image_Back_resize = cv2.resize(stitch_result[1], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("stitched_image_Back", stitched_image_Back_resize)
            else:
                stitched_image_resize = cv2.resize(stitch_result[0], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("stitched_image_resize", stitched_image_resize)


        if is_twin_truck:
            if len(stitch_result) == 2:
                if stitch_result[0] is not None:
                    stitching_result[1] = stitch_result[0]
                if stitch_result[1] is not None:
                    stitching_result[4] = stitch_result[1]
        else:
            if len(stitch_result) == 1:
                if stitch_result[0] is not None:
                    stitching_result[1] = stitch_result[0]








    '''''''''''''''''''''''''''''''''''''''
    do side stitching
    '''''''''''''''''''''''''''''''''''''''
    def do_side_stitching(self, frame_set, container_size_idx, start_idx, mid_idx, end_idx, is_twin_truck, camera_idx):
        side_stitched_result = []


        ### crop image using start-end frame index ###
        try:
            cropped_set = crop_frames(frame_set, start_idx, end_idx)
        except:
            print("!!! crop image using start-end frame index Err !!!")
            return False, None
        if False:
            print("crop image using start-end frame index:", start_idx, end_idx)
            frame_show(cropped_set, 0)




        ### process roi using container size info ###
        try:
            if camera_idx == 1: #left
                first_frame_indent_view, cropped_roi_set = define_Roi_left(cropped_set)

            if camera_idx == 2: #right
                first_frame_indent_view, cropped_roi_set = define_Roi_right(cropped_set)
        except:
            print("!!! process roi Err !!!")
            return False, None
        if False:
            frame_show(cropped_roi_set, 0)
            i1 = cv2.resize(cropped_roi_set[0], (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            i2 = cv2.resize(cropped_roi_set[-1], (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            cv2.imshow("cropped_roi_set start", i1)
            cv2.imshow("cropped_roi_set end", i2)
            cv2.waitKey(0)


        if camera_idx == 1: #left
            side_stitched_image = Stitching_side_left(first_frame_indent_view, cropped_roi_set, 0, container_size_idx)

        if camera_idx == 2: #right
            side_stitched_image = Stitching_side_right(first_frame_indent_view, cropped_roi_set, 0, container_size_idx)


        ### make stitched image using warped-top video set ###
        try:
            if camera_idx == 1: #left
                side_stitched_image = Stitching_side_left(first_frame_indent_view, cropped_roi_set, 0, container_size_idx)

            if camera_idx == 2: #right
                side_stitched_image = Stitching_side_right(first_frame_indent_view, cropped_roi_set, 0, container_size_idx)

        except:
            print("!!! make stitched image Err !!!")
            return False, None
        # if True:
        #     side_stitched_image_resize = cv2.resize(side_stitched_image, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("side_stitched_image", side_stitched_image_resize)
            # cv2.waitKey(0)





        ### do hole filling using closing ###
        try:
            val = 1
            val = val * 2 + 1
            kernelNDArray = np.ones((val, val), np.uint8)
            side_stitched_image = cv2.morphologyEx(side_stitched_image, cv2.MORPH_CLOSE, kernelNDArray)

        except:
            print("!!! RIGHT Hole filling Err !!!")
            return False, None
        if False:
            side_stitched_image_resize = cv2.resize(side_stitched_image, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("side_stitched_image", side_stitched_image_resize)
            cv2.waitKey(0)



        ### stitched image twin division ###
        try:
            if is_twin_truck and mid_idx != -1:
                if camera_idx == 1: #left
                    front_side_stitched_img, back_side_stitched_img = twin_division_left(side_stitched_image, start_idx, mid_idx, end_idx)

                if camera_idx == 2: #right
                    front_side_stitched_img, back_side_stitched_img = twin_division_right(side_stitched_image, start_idx, mid_idx, end_idx)


                side_stitched_result.append(side_stitched_image)
                side_stitched_result.append(front_side_stitched_img)
                side_stitched_result.append(back_side_stitched_img)
            else:
                side_stitched_result.append(side_stitched_image)
        except:
            print("!!! twin division Err !!!")
        if False:
            if is_twin_truck:
                front_stitched_side_resize = cv2.resize(front_side_stitched_img, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
                back_stitched_side_resize = cv2.resize(back_side_stitched_img, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("front_side_stitched_img", front_stitched_side_resize)
                cv2.imshow("back_side_stitched_img", back_stitched_side_resize)
                cv2.waitKey(0)


        return True, side_stitched_result



    ''''
    Do top stitching
    input: frame set for stitching, H matrix, container size info, start-end frame index
    output: top stitched image
    '''

    def do_top_stitching(self, frame_set, H, container_size_idx, start_idx, mid_idx, end_idx, is_twin_truck):
        top_stitched_result = []

        ### perform homographic transfomation ###
        try:
            frame_set_H = frame_set_H_transformation(frame_set, H)
        except:
            print("!!! perform homographic top transfomation Err !!!")
            return False, None
        if False:
            frame_show(frame_set_H, 0)




        ### crop image using start-end frame index ###
        try:
            cropped_set = crop_frames(frame_set_H, start_idx, end_idx)
        except:
            print("!!! crop top image using start-end frame index Err !!!")
            return False, None
        if False:
            frame_show(cropped_set, 0)




        ### process roi using container size info ###
        try:
            first_frame_indent_view, cropped_roi_set = define_Roi(cropped_set)
        except:
            print("!!! process top roi Err !!!")
            return False, None
        if False:
            frame_show(cropped_roi_set, 0)
            cv2.imshow("cropped_roi_set start", cropped_roi_set[0])
            cv2.imshow("cropped_roi_set end", cropped_roi_set[-1])
            cv2.waitKey(0)




        ### make stitched image using warped-top video set ###
        try:
            top_stitched_image = Stitching_top(first_frame_indent_view, cropped_roi_set, 0, container_size_idx)
        except:
            print("!!! make top stitched image Err !!!")
            return False, None
        if False:
            top_stitched_image_resize = cv2.resize(top_stitched_image, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("top_stitched_image", top_stitched_image_resize)
            cv2.waitKey(0)





        ### do hole filling using closing ###
        try:
            val = 1
            val = val * 2 + 1
            kernelNDArray = np.ones((val, val), np.uint8)
            top_stitched_image = cv2.morphologyEx(top_stitched_image, cv2.MORPH_CLOSE, kernelNDArray)

        except:
            print("!!! RIGHT Hole filling Err !!!")
            return False, None
        if False:
            top_stitched_image_resize = cv2.resize(top_stitched_image, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("top_stitched_image", top_stitched_image_resize)
            cv2.waitKey(0)



        ### stitched image twin division ###
        try:
            if is_twin_truck and mid_idx != -1:
                front_top_stitched_img, back_top_stitched_img = twin_division(top_stitched_image, start_idx, mid_idx, end_idx)
                top_stitched_result.append(top_stitched_image)
                top_stitched_result.append(front_top_stitched_img)
                top_stitched_result.append(back_top_stitched_img)
            else:
                top_stitched_result.append(top_stitched_image)
        except:
            print("!!! twin division Err !!!")
        if False:
            if is_twin_truck:
                front_stitched_top_resize = cv2.resize(front_top_stitched_img, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
                back_stitched_top_resize = cv2.resize(back_top_stitched_img, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("front_top_stitched_image", front_stitched_top_resize)
                cv2.imshow("back_top_stitched_image", back_stitched_top_resize)
                cv2.waitKey(0)


        return True, top_stitched_result


    def make_Top_view_H_BCT(self, h, w):

        point_list = [(673, 456) , (1853, 480) , (20, 993) , (2450, 1073)]
        kp1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])

        kp2 = np.array([[w/8, h/10*2],
                            [w/8*7, h/10*2],
                            [w/8, h/10*9],
                            [w/8*7, h/10*9]])

        # M = cv2.getPerspectiveTransform(kp1,kp2)
        H, _ = cv2.findHomography(kp1, kp2, cv2.RANSAC,4.0)

        return H

    def sequence_alignment(self, up_image, down_image, up_idxset, net_info, is_camera):
        '''
        align down image set using up index
        return best index
        '''
        if is_camera == 0:
            yminoffset = 400
            ymaxoffset = 300
        else:
            yminoffset = 400
            ymaxoffset = 400


        '''
        make up stitching information
        '''
        up_detections = self.border2frontback(up_idxset, 25)
        for idx in range(len(up_detections)):
            det_undist_x, det_undist_y = self.HR2Undist(up_detections[idx][2], yminoffset, ymaxoffset)
            up_detections[idx].append([det_undist_x, det_undist_y])

        _, w, _ = up_image[0].shape
        up_Info = find_cropped_info_up(up_detections, w, up_detections[0][0]) # front idx, back idx, front distance, back distance, center idx


        front_idx = up_Info[0]
        back_idx = up_Info[1]
        center_idx = up_Info[4]
        front_distance = up_Info[2]
        back_distance = up_Info[3]
        len_f2b = back_idx - front_idx
        len_f2c = center_idx - front_idx
        len_c2b = back_idx - center_idx




        '''
        find first down container border idx using up part front and back distance
        '''
        indent = 5
        i_max = indent
        n_max = 8 + 1
        is_Front = None
        det_down_idx = -1
        det_down_dist= -1
        image_shape = None # (rotate_image(down_image[0], is_camera, self.L_w, self.L_h, yminoffset, ymaxoffset)).shape

        up_best_idx = -1
        up_min_dist = -1

        ### detect pivot down border information using NN front image case 
        if abs(front_distance) <= abs(back_distance):
            is_Front = True
            det_down_idx, det_down_dist, image_shape = self.search_down_idx(down_image, net_info, front_idx, i_max, n_max, indent, is_camera, yminoffset, ymaxoffset)
            up_best_idx = front_idx
            up_min_dist = front_distance

        ### detect pivot down border information using NN back image case
        else:
            is_Front = False
            det_down_idx, det_down_dist, image_shape = self.search_down_idx(down_image, net_info, back_idx, i_max, n_max, indent, is_camera, yminoffset, ymaxoffset)
            up_best_idx = back_idx
            up_min_dist = back_distance



        '''
        find more accurate down border idx using first detected down border idx
        '''
        ### Set moving direction using roi_pts ###
        if is_camera == 0: direction_reverser = 1
        else: direction_reverser = -1

        ### get more accurate DOWN info  ###
        down_list = []
        Down_direction = 0
        interval = 4
        if int(image_shape[1]/2 - det_down_dist) >= 0: Down_direction = -1
        else: Down_direction = 1

        if Down_direction < 0:
            for i in range(5):
                distance = self.get_distance(down_image[det_down_idx + interval*i *direction_reverser], net_info, is_camera)
                down_list.append([det_down_idx + interval*i *direction_reverser, distance])

                if False:
                    print("< 0  distance: ", distance)
                    img = cv2.resize(down_image[det_down_idx + interval*i *direction_reverser], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("down_direction < 0", img)
                    cv2.waitKey(0)

        else:
            for i in range(5):
                distance = self.get_distance(down_image[det_down_idx - interval*i *direction_reverser], net_info, is_camera)
                down_list.append([det_down_idx - interval*i *direction_reverser, distance])

                if False:
                    print("> 0  distance: ", distance)
                    img = cv2.resize(down_image[det_down_idx - interval*i *direction_reverser], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("down_direction > 0", img)
                    cv2.waitKey(0)


        down_best_idx = -1
        down_min_dist = 999999
        for idx in range(len(down_list)):
            if down_list[idx][1] is not None:
                if abs(int(down_list[idx][1])) < abs(down_min_dist):
                    down_best_idx = down_list[idx][0]
                    down_min_dist = down_list[idx][1]


        if False:
            up_img_resize = cv2.resize(up_image[up_best_idx], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("up best image" + str(up_min_dist), up_img_resize)
            cv2.waitKey(0)

            down_img_resize = cv2.resize(down_image[down_best_idx], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("down best image" + str(down_min_dist), down_img_resize)
            cv2.waitKey(0)




        ### save down stitching image information ###
        det_down_idx, det_down_dist
        down_Info = []
        if is_Front:
            down_info_front_idx         = (down_best_idx)
            down_info_front_distance    = (down_min_dist)
            down_info_back_idx          = (front_idx + len_f2b)
            down_info_back_distance     = (self.get_distance(down_image[back_idx], net_info, is_camera))
            down_info_center_idx        = (front_idx + len_f2c)

            down_Info.append(down_info_front_idx)
            down_Info.append(down_info_back_idx)
            down_Info.append(down_info_front_distance)
            down_Info.append(down_info_back_distance)
            down_Info.append(down_info_center_idx)

        else:
            down_info_back_idx          = (down_best_idx)
            down_info_back_distance     = (down_min_dist)
            down_info_front_idx         = (back_idx - len_f2b)
            down_info_front_distance    = (self.get_distance(down_image[front_idx], net_info, is_camera))
            down_info_center_idx        = (back_idx - len_c2b)

            down_Info.append(down_info_front_idx)
            down_Info.append(down_info_back_idx)
            down_Info.append(down_info_front_distance)
            down_Info.append(down_info_back_distance)
            down_Info.append(down_info_center_idx)

        for idx in range(len(down_Info)):
            if down_Info[idx] is None: down_Info[idx] = -1
            else: down_Info[idx] = int(down_Info[idx])


        if False:
            print("UP info: ", up_Info)
            Fup_img_resize = cv2.resize(up_image[up_Info[0]], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            Bup_img_resize = cv2.resize(up_image[up_Info[1]], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("up Front best", Fup_img_resize)
            cv2.imshow("up Back best", Bup_img_resize)
            cv2.waitKey(0)

            print("DOWN info: ", down_Info)
            Fdown_img_resize = cv2.resize(down_image[down_Info[0]], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            Bdown_img_resize = cv2.resize(down_image[down_Info[1]], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("down Front best", Fdown_img_resize)
            cv2.imshow("down Back best", Bdown_img_resize)
            cv2.waitKey(0)


        return up_Info, down_Info, is_Front





    def search_down_idx(self, down_image, net_info, start_idx, i_max, n_max, indent, is_camera, yminoffset, ymaxoffset):
        '''
        find first down container border idx from up part front and back distance
        detect pivot down border information using NN front image case 

        return down index, distance of detected roi from pre-defined baseline
        '''
        height, width = None, None
        breaker = False
        for _i in range(i_max): # _i in |indent * _n + _i|
            for _n in range(n_max): # _n in |indent * _n + _i|. max_n is 8
                # print("start_idx: ", start_idx, "(indent * _n + _i): ", (indent * _n + _i))

                idx = start_idx + (indent * _n + _i) # rignt
                if (0 < idx) and (idx < len(down_image)):
                    # rotate image processing
                    img = rotate_image(down_image[idx], is_camera, self.L_w, self.L_h, yminoffset, ymaxoffset)

                    # infer to NN. if (Cont_border) detected, then return idx.
                    height, width, _ = img.shape
                    img, detections, roi_pts, center_pts = forward_NN(img, net_info[0], net_info[1], net_info[2], width, height, net_info[4], net_info[5], net_info[3])
                    if len(detections) == 1:
                        down_image_cp_x = int( roi_pts[0][0] +  (roi_pts[0][2] - roi_pts[0][0]) / 2 )
                        if (detections[0][0] == 'Cont_border') and \
                            (width/3 - (roi_pts[0][2] - roi_pts[0][0])) > 0 and \
                            ((roi_pts[0][3] - roi_pts[0][1]) - height/2) > 0:

                            breaker = True
                            down_idx = idx

                            if False:
                                drawed_image = darknet.draw_boxes(detections, img, (0,255,0), roi_pts)
                                img = cv2.resize(drawed_image, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                                cv2.imshow("- idx", img)
                                cv2.waitKey(0)

                            break
                        else: continue


                idx = start_idx - (indent * _n + _i)   # left
                if (0 < idx) and (idx < len(down_image)):
                    # rotate image processing
                    img = rotate_image(down_image[idx], is_camera, self.L_w, self.L_h, yminoffset, ymaxoffset)

                    # infer to NN. if (Cont_border) detected, then return idx.
                    height, width, _ = img.shape
                    img, detections, roi_pts, center_pts = forward_NN(img, net_info[0], net_info[1], net_info[2], width, height, net_info[4], net_info[5], net_info[3])
                    if len(detections) == 1:
                        down_image_cp_x = int( roi_pts[0][0] +  (roi_pts[0][2] - roi_pts[0][0]) / 2 )
                        if (detections[0][0] == 'Cont_border') and \
                            (width/3 - (roi_pts[0][2] - roi_pts[0][0])) > 0 and \
                            ((roi_pts[0][3] - roi_pts[0][1]) - height/2) > 0:

                            breaker = True
                            down_idx = idx

                            if False:
                                drawed_image = darknet.draw_boxes(detections, img, (0,255,0), roi_pts)
                                img = cv2.resize(drawed_image, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                                cv2.imshow("- idx", img)
                                cv2.waitKey(0)

                            break
                        else: continue

            if breaker == True: break

        return down_idx, down_image_cp_x, img.shape



    def get_distance(self, img, net_info, is_camera):
        '''
        get distance from NN roi to image rectangle
        return: distance
        '''
        distance = None

        if is_camera == 0:
            yminoffset = 400
            ymaxoffset = 300
        else:
            yminoffset = 400
            ymaxoffset = 400


        # rotate image processing
        img = rotate_image(img, is_camera, self.L_w, self.L_h, yminoffset, ymaxoffset)

        # infer to NN. if (Cont_border) detected, then return idx.
        height, width, _ = img.shape
        img, detections, roi_pts, center_pts = forward_NN(img, net_info[0], net_info[1], net_info[2], width, height, net_info[4], net_info[5], net_info[3])
        if len(detections) == 1:
            cp_x = int( roi_pts[0][0] +  (roi_pts[0][2] - roi_pts[0][0]) / 2)

            # if (detections[0][0] == 'Cont_border'):
            if (detections[0][0] == 'Cont_border') and \
                (width/2 - (roi_pts[0][2] - roi_pts[0][0])) > 0 and \
                ((roi_pts[0][3] - roi_pts[0][1]) - height/2) > 0:
                distance = (cp_x - width/2)

                if False:
                    drawed_image = darknet.draw_boxes(detections, img, (0,255,0), roi_pts)
                    img = cv2.resize(drawed_image, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("(debug) distance", img)
                    cv2.waitKey(0)

        return distance


    def process_downSet_using_NN(self, down_images, interval, net_info, is_camera):

        ### for Right Camera (1,2)  down images ###
        if is_camera == 0: # Right(1,2)
            yminoffset = 400
            ymaxoffset = 300

            try:
                rotate_images(down_images, is_camera, self.L_w, self.L_h, yminoffset, ymaxoffset)
                if False:
                    for img in down_images:
                        img_resize = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                        cv2.imshow("downimg_resize", img_resize)
                        if cv2.waitKey(0) == ord('q'): break

                down_detections = Limgs_foward_NN(down_images, interval, net_info[0], net_info[1], net_info[2], net_info[4], net_info[5], net_info[3])
                down_detections = self.border2frontback(down_detections, 25)
                down_Info = find_cropped_Info_Rdown(down_images, down_detections)


                # -debug
                if False:
                    print("RIGHT down_Info: ", down_Info)
                    for idx in range(len(down_detections)):
                        down_s = cv2.resize(down_images[down_detections[idx][0]], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                        cv2.imshow("Right(1,2) down_test", down_s)
                        if cv2.waitKey(0) == ord('q'): break

                return down_Info
            except:
                print("!!! RIGHT Down Info Err !!!")




        ### for Left Camera (3,4)  down images ###
        elif is_camera == 1: # Left 3,4)
            yminoffset = 400
            ymaxoffset = 400

            try:
                rotate_images(down_images, is_camera, self.L_w, self.L_h, yminoffset, ymaxoffset)
                if False:
                    for img in down_images:
                        img_resize = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                        cv2.imshow("downimg_resize", img_resize)
                        if cv2.waitKey(0) == ord('q'): break

                down_detections = Limgs_foward_NN(down_images, interval, net_info[0], net_info[1], net_info[2], net_info[4], net_info[5], net_info[3])
                down_detections = self.border2frontback(down_detections, 25)
                down_Info = find_cropped_Info_Ldown(down_images, down_detections)


                # -debug
                if False:
                    print("LEFT down_Info: ", down_Info)
                    for idx in range(len(down_detections)):
                        down_s = cv2.resize(down_images[down_detections[idx][0]], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                        cv2.imshow("Left(3,4) down_test", down_s)
                        if cv2.waitKey(0) == ord('q'): break

                return down_Info

            except:
                print("!!! LEFT Down Info Err !!!")



            return None




    def if_twin_division(self, stitched_image, centor_dist, avg_velocity, indent, is_twin_truck):
        stitch_set = []

        if is_twin_truck:
            affordable_area = int(stitched_image.shape[1] / 10)
            x_length = int( avg_velocity * abs(centor_dist))

            if x_length > 0:
                stitched_image_Front = stitched_image[: , :int(indent-avg_velocity + x_length + affordable_area)]
                stitched_image_Back = stitched_image[:, int(indent-avg_velocity + x_length - affordable_area): ]
            else:
                stitched_image_Front = stitched_image[:, : int(indent-avg_velocity + stitched_image.shape[1] - x_length + affordable_area)]
                stitched_image_Back = stitched_image[:, int(indent-avg_velocity + stitched_image.shape[1] - x_length - affordable_area) :]

            stitch_set.append(stitched_image_Front)
            stitch_set.append(stitched_image_Back)
        else:
            stitch_set.append(stitched_image)

        return stitch_set



    def border2frontback(self, detections, thres):
        '''
        convert from Cont_border class to Cont_front and Cont_back

        return: converted detections 
        input: original detections, indent thresold
        '''
        cur_idx = -1
        change_flag = False

        if False:
            print("detections: ")
            for data in detections:
                print(data)

        ### find initial border(front) ###
        for idx in range(len(detections)):
            if detections[idx][1] == "Cont_border":
                cur_idx = detections[idx][0]
                break


        ### cvt the border 2 front & back ###
        for idx in range(len(detections)):
            if (detections[idx][1] == "Cont_border"):
                if change_flag == False:
                    if (detections[idx][0] - cur_idx) < thres:
                        detections[idx][1] = "Cont_front"
                        cur_idx = detections[idx][0]
                    else:
                        change_flag = True
                        detections[idx][1] = "Cont_back"

                else:
                    detections[idx][1] = "Cont_back"

        return detections



    def HR2Undist(self, roi_pts, yminoffset, ymaxoffset):
        left, top, right, bottom = roi_pts
        cx = int(left + (right - left)/2)
        cy = int(top +(bottom - top)/2)


        ### pt_H 2 pt_origin (1570, 894, 3)
        pt_H = np.array([self.L_w + 30 - (cy + 380), (cx + 114), 1])
        pt = np.dot(self.HR_inv, pt_H)
        x = int(pt[0]/pt[2])
        y = int(pt[1]/pt[2])

        ### pt_origin 2 pt_undisstort
        undist_x = int(self.mMap_y[y, x])
        undist_y = self.mMap_x.shape[1] - int(self.mMap_x[y, x]) - yminoffset

        # ### pt_origin 2 pt_undistort
        # undist_x = self.mMap_x.shape[0] - int(self.mMap_y[y, x])
        # undist_y = self.mMap_x.shape[1] - int(self.mMap_x[y, x]) - yminoffset

        return undist_x, undist_y



    def HL2Undist(self, roi_pts, yminoffset, ymaxoffset):
        '''
        convert Homographied roi pt to undistorted roi pt for speed improvment

        return undistorted roi pt
        input Homographied roi pt
        '''
        left, top, right, bottom = roi_pts
        cx = int(left + (right - left)/2)
        cy = int(top +(bottom - top)/2)


        ### pt_H 2 pt_origin
        pt_H = np.array([cx + 50, cy + 600, 1]) #crop

        pt = np.dot(self.HL_inv, pt_H) #H
        x = int(pt[1]/pt[2])            #flip&transpose
        y = self.L_h - int(pt[0]/pt[2])

        ### pt_origin 2 pt_undistort
        undist_x = self.L_h - int(self.mMap_y[y, x])
        undist_y = int(self.mMap_x[y, x]) - yminoffset


        return undist_x, undist_y



    def delete_info(self):

        print("before")
        print(sys.getsizeof(self.cam_Ldset), " ", sys.getsizeof(self.cam_Ldset[0]), " ", len(self.cam_Ldset))
        print(sys.getsizeof(self.cam_Luset))
        print(sys.getsizeof(self.cam_Rdset))
        print(sys.getsizeof(self.cam_Ruset))

        for img in (self.cam_Ldset): img = None
        for img in (self.cam_Luset): img = None
        for img in (self.cam_Rdset): img = None
        for img in (self.cam_Ruset): img = None

        self.cam_Ldset = []
        self.cam_Luset = []
        self.cam_Rdset = []
        self.cam_Ruset = []
        self.cam_TRset = []

        self.det_Luset = [-1, [], -1]
        self.det_Ruset = [-1, [], -1]
        self.det_TRset = [-1, [], -1]

        print("after")
        print(sys.getsizeof(self.cam_Ldset))
        print(sys.getsizeof(self.cam_Luset))
        print(sys.getsizeof(self.cam_Rdset))
        print(sys.getsizeof(self.cam_Ruset))






'''''''''''''''''''''''''''''''''''''''
Image Stitching Function
'''''''''''''''''''''''''''''''''''''''

### image rotation ###
def rotate_images(images, camera_idx, L_w, L_h, minoffset, maxoffset):
    """ 
    This method is for rotating the images.
    
    :input param: images of the truck, camera index(left or right), width, height, offsets for cropping
    :return: rotated image list 

    """
    yminoffset = minoffset
    ymaxoffset = maxoffset

    for idx in range(len(images)):
        if camera_idx == 0: # 1 2
            images[idx] = cv2.rotate(images[idx], cv2.ROTATE_90_COUNTERCLOCKWISE)

            # height, width, channel = images[idx].shape
            # matrix = cv2.getRotationMatrix2D((width/2, height/2), -2, 1)
            # images[idx] = cv2.warpAffine(images[idx], matrix, (width, height))

            images[idx] = images[idx][yminoffset:L_w - ymaxoffset, :]


            # images_resize = cv2.resize(images[idx], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("test for save img", images_resize)
            # if cv2.waitKey(0)&0xff == ord ('q'):
            #     cv2.imwrite("images.jpg",images[idx])



        else: # 3 4
            images[idx] = cv2.rotate(images[idx], cv2.ROTATE_90_CLOCKWISE)

            # height, width, channel = images[idx].shape
            # matrix = cv2.getRotationMatrix2D((width/2, height/2), 2, 1)
            # images[idx] = cv2.warpAffine(images[idx], matrix, (width, height))

            images[idx] = images[idx][yminoffset:L_w - ymaxoffset, :]

            # images_resize = cv2.resize(images[idx], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("test for save img", images_resize)
            # if cv2.waitKey(0)&0xff == ord ('q'):
            #     cv2.imwrite("images.jpg",images[idx])


def rotate_image(image, camera_idx, L_w, L_h, minoffset, maxoffset):
    yminoffset = minoffset
    ymaxoffset = maxoffset

    if camera_idx == 0: # 1 2
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = image[yminoffset:L_w - ymaxoffset, :]

    else: # 3 4
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = image[yminoffset:L_w - ymaxoffset, :]

    return image

### cutting image unit ###
def make_unit_images(before, cut_unit):
    """ 
    This method is making the basic images list
    
    :input param: image list(before) of the truck and interval for cutting the list
    :return: cut image list(after)
    """
    after = []
    for idx in range(len(before)):
        if idx % cut_unit == 0:
            after.append(before[idx])

    return after


def find_cropped_info_up(detections, width, indent):
    ''' 
    find front, centor and back information(best matched front, centor and back frame idx using distance of centor distance)

    input: detection from NN, image width
    return: front, centor, back information of frame set
    '''
    best_front_dist = 500000
    front_idx = -1
    front_sign = 1

    best_back_dist = 500000
    back_idx = -1
    back_sign = 1


    center_idx = None
    center_idx_set = []

    left_width_offset = 0.4
    for idx in range(len(detections)):
        if detections[idx][1] == 'Cont_front' and idx < len(detections)/2:
            front_x_dist = int(width*left_width_offset) -  detections[idx][3][0]
            if front_x_dist >= 0:
                front_sign = 1
            else:
                front_sign = -1
            front_x_dist = abs(front_x_dist)

            if front_x_dist < best_front_dist:
                best_front_dist = front_x_dist
                front_idx = idx


        if detections[idx][1] == 'Cont_back' and idx > len(detections)/2:
            back_x_dist = int(width*left_width_offset) -  detections[idx][3][0]
            if back_x_dist >= 0:
                back_sign = 1
            else:
                back_sign = -1
            back_x_dist = abs(back_x_dist)

            if back_x_dist < best_back_dist:
                best_back_dist = back_x_dist
                back_idx = idx

        if (detections[idx][1] == 'Cont_center'):
            center_idx_set.append(detections[idx][0])

    if len(center_idx_set) >= 3:
        center_idx = statistics.median(center_idx_set)
    else: center_idx = -1

    return [detections[front_idx][0], detections[back_idx][0], front_sign * best_front_dist, back_sign * best_back_dist, int(center_idx)]


# find idx info of container
def find_cropped_Info_Rdown(imgs, detections):

    """
    This method finds where container is and return that index information of the image list.
    
    :input param: image list(before) of the truck and information of network #3
    :return: index information that indicates where the container is.
    """
    img_ = imgs[0]
    height, width, _ = img_.shape
    cur_idx = -1
    change_flag = False

    best_front_dist = 500000
    front_idx = -1
    front_sign = 1

    best_back_dist = 500000
    back_idx = -1
    back_sign = 1

    center_idx = None
    center_idx_set = []

    left_width_offset = 0.5
    for i in range(len(detections)):
        if (detections[i][1] == 'Cont_front') :
            front_x_dist = abs( int(width*left_width_offset) - detections[i][2][0])  #
            if (int(width*left_width_offset) - detections[i][2][0]) >= 0:
                front_sign = 1
            else:
                front_sign = -1

            if front_x_dist < best_front_dist:
                best_front_dist = front_x_dist
                front_idx = detections[i][0]

        if (detections[i][1] == 'Cont_back') :
            back_x_dist = abs( int(width*left_width_offset) - detections[i][2][0])
            if (int(width*left_width_offset) - detections[i][2][0]) >= 0:
                back_sign = 1
            else:
                back_sign = -1

            if back_x_dist < best_back_dist:
                best_back_dist = back_x_dist
                back_idx = detections[i][0]

        if (detections[i][1] == 'Cont_center'):
            center_idx_set.append(detections[i][0])


    if len(center_idx_set) >= 3:
        center_idx = statistics.median(center_idx_set)
    else: center_idx = -1


    return [front_idx, back_idx, front_sign * best_front_dist, back_sign * best_back_dist, int(center_idx)]



def Limgs_foward_NN(imgs, split_index, net, meta, thres, net_width, net_height, net_colors):
    '''
    forward images set to NN

    input split_index(for speed improvement), netinfo(net, meta, thres, net_width, net_height, net_colors)
    return detected result
    '''
    down_detections = []

    for idx in range(len(imgs)):
        if idx % split_index == 0:
            img_ = imgs[idx]
            height, width, _ = img_.shape
            img_, detections, roi_pts, center_pts = forward_NN(img_, net, meta, thres, width, height, net_width, net_height, net_colors)


            if len(detections) != 0:
                if detections[0][0] == 'Cont_border' or detections[0][0] == 'Cont_center':
                    down_detections.append( [idx, detections[0][0], center_pts[0]] )
                else: continue

            if False:
                if detections is not None:
                    drawed_image = darknet.draw_boxes(detections, imgs[idx], (0,255,0), roi_pts)
                    drawed_image_resize = cv2.resize(drawed_image, dsize=(0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("debug left NN", drawed_image_resize)
                    print(f"{detections}\n")
                    cv2.waitKey(0)

    return down_detections



# find idx info of container
def find_cropped_Info_Ldown(imgs, detections):

    """
    This method finds where container is and return that index information of the image list.
    
    :input param: image list(before) of the truck and information of network #3
    :return: index information that indicates where the container is.
    """
    img_ = imgs[0]
    height, width, _ = img_.shape
    cur_idx = -1
    change_flag = False

    best_front_dist = 500000
    front_idx = -1
    front_sign = 1

    best_back_dist = 500000
    back_idx = -1
    back_sign = 1

    center_idx = None
    center_idx_set = []
    left_width_offset = 0.4
    for i in range(len(detections)):
        if (detections[i][1] == 'Cont_front') :
            front_x_dist = abs( int(width*left_width_offset) - detections[i][2][0])  #
            if (int(width*left_width_offset) - detections[i][2][0]) >= 0:
                front_sign = 1
            else:
                front_sign = -1

            if front_x_dist < best_front_dist:
                best_front_dist = front_x_dist
                front_idx = detections[i][0]

        if (detections[i][1] == 'Cont_back') :
            back_x_dist = abs( int(width*left_width_offset) - detections[i][2][0])
            if (int(width*left_width_offset) - detections[i][2][0]) >= 0:
                back_sign = 1
            else:
                back_sign = -1

            if back_x_dist < best_back_dist:
                best_back_dist = back_x_dist
                back_idx = detections[i][0]

        if (detections[i][1] == 'Cont_center'):
            center_idx_set.append(detections[i][0])


    if len(center_idx_set) >= 3:
        center_idx = statistics.median(center_idx_set)
    else: center_idx = -1

    return [front_idx, back_idx, front_sign * best_front_dist, back_sign * best_back_dist, int(center_idx)]





def forward_NN(img, net, meta, thres, cap_w, cap_h, net_width, net_height, net_colors):

    detections, frame_resized = darknet.detect_NN(img, net, meta, thres)
    roi_pts = darknet.point_cvt(detections, cap_w/net_width, cap_h/net_height)

    center_pts = []
    for idx in range(len(roi_pts)):
        left, top, right, bottom = roi_pts[idx]
        cp_x = int( left +  (right - left) / 2 )
        cp_y = int( top + (bottom - top) / 2 )
        center_pts.append((cp_x, cp_y))

    return img, detections, roi_pts, center_pts


def make_cropped_Set(ups, downs, up_Info, up_front_idx, down_front_idx, imageset_len):

    up_front_idx = up_Info[0]

    up_cropped_set = ups[up_front_idx : up_front_idx + imageset_len]
    down_cropped_set = downs[down_front_idx : down_front_idx + imageset_len]

    indent = 5
    centor_dist = abs(up_Info[4] - up_Info[0]) + indent

    difference_x = 0

    return up_cropped_set, down_cropped_set, difference_x, centor_dist


def make_cropped_updownSet(ups, downs, up_Info, down_Info):

    """
    This function stores only the portion of the container in the list for stitching,
    based on index information about the container location that it receives.
    
    :input param: image list and index information of upper and down cam.
    :return: cropped image list of upper and down cam, difference of x axis of up and down.
    """

    up_front_idx = up_Info[0]
    up_back_idx = up_Info[1]
    up_front_dist = up_Info[2]
    up_back_dist = up_Info[3]

    down_front_idx = down_Info[0]
    down_back_idx = down_Info[1]
    down_front_dist = down_Info[2]
    down_back_dist = down_Info[3]

    # the case of no information of upper cam or lower cam or even both.
    if (up_front_idx == -1 and up_back_idx == -1) or (down_front_idx == -1 and down_back_idx == -1):
        return [], [], -1

    # the case that upper cam's container front does not detect
    if up_front_idx == -1 and up_back_idx != -1 and down_front_idx != -1 and down_back_idx != -1:
        up_front_idx = up_back_idx - (down_back_idx - down_front_idx)
        up_front_dist = -300

    # the case that upper cam's container back does not detect
    if up_front_idx != -1 and up_back_idx == -1 and down_front_idx != -1 and down_back_idx != -1:
        up_back_idx = up_front_idx + (down_back_idx - down_front_idx)
        up_back_idx = -300

    # the case that lower cam's container front does not detect
    if up_front_idx != -1 and up_back_idx != -1 and down_front_idx == -1 and down_back_idx != -1:
        down_front_idx = down_back_idx - (up_back_idx - up_front_idx)
        down_front_dist = -300

    # the case that lower cam's container back does not detect
    if up_front_idx != -1 and up_back_idx != -1 and down_front_idx != -1 and down_back_idx == -1:
        down_back_idx = down_front_idx + (up_back_idx - up_front_idx)
        down_back_dist = -300

    # the case that upper and lower cam's container front do not detect
    if up_front_idx == -1 and up_back_idx != -1 and down_front_idx == -1 and down_back_idx != -1:
        if up_back_idx > 120 and down_back_idx > 120:
            up_front_idx = up_back_idx - 120
            down_front_idx = down_back_idx - 120
            up_front_dist = -300
            down_front_dist = -300

    # the case that upper and lower cam's container back do not detect
    if up_front_idx != -1 and up_back_idx == -1 and down_front_idx != -1 and down_back_idx == -1:
        up_back_idx = len(ups)-1
        down_back_idx = len(downs)-1
        up_back_dist = 0
        down_back_dist = 0


    abs_up_front_dist = abs(up_front_dist)
    abs_up_back_dist = abs(up_back_dist)
    abs_down_front_dist = abs(down_front_dist)
    abs_down_back_dist = abs(down_back_dist)


    ### decide num of images and indent for centor deciding in twin case
    indent = 0
    num_imgs = -1
    nList = [abs_up_front_dist, abs_up_back_dist, abs_down_front_dist, abs_down_back_dist]
    if (up_back_idx - up_front_idx) > (down_back_idx - down_front_idx):
        num_imgs = abs(up_back_idx - up_front_idx) + indent
        centor_dist_front = abs(up_Info[4] - up_front_idx) + indent
        centor_dist_back = abs(up_Info[4] - up_back_idx)  + indent
    else:
        num_imgs = abs(down_back_idx - down_front_idx) + indent
        centor_dist_front = abs(down_Info[4] - down_front_idx) + indent
        centor_dist_back = abs(down_Info[4] - down_back_idx) + indent

    if num_imgs % 2 == 1: num_imgs += 1


    ### select where to crop image and get diffrent x-axis using num of images
    front_x_dist = math.sqrt(int(math.pow(up_front_dist - down_front_dist, 2)))
    back_x_dist = math.sqrt(int(math.pow(up_back_dist - down_back_dist, 2)))

    # print(f"UP front ({(up_front_dist)}) vs UP back ({(down_front_dist)}) [upinfo: {up_Info}]")
    # print(f"DOWN front ({(up_back_dist)}) vs DOWN back ({(down_back_dist)}) [downinfo: {down_Info}]")
    # print(f"Front ({(front_x_dist)}) vs Back ({(back_x_dist)})")

    centor_dist = None
    # front correction
    if front_x_dist < back_x_dist:
        centor_dist = centor_dist_front
        up_cropped_set = ups[up_front_idx:up_front_idx+num_imgs]
        down_cropped_set = downs[down_front_idx:down_front_idx+num_imgs]

        if (up_front_dist >= 0) and (down_front_dist >= 0): #(+,+)
            if abs_up_front_dist > abs_down_front_dist:
                diff_x = front_x_dist
                # print("f(+,+) / a ", diff_x)
            else:
                diff_x = -1 * front_x_dist
                # print("f(+,+) / b ", diff_x)


        elif (up_front_dist > 0) and (down_front_dist < 0): #(+,-)
            if abs_up_front_dist > abs_down_front_dist:
                diff_x = front_x_dist
                # print("f(+,-) / a ", diff_x)

            else:
                diff_x = front_x_dist  ### 6500
                # print("f(+,-) / b ", diff_x)

        elif (up_front_dist < 0) and (down_front_dist > 0): #(-,+)
            if abs_up_front_dist > abs_down_front_dist:
                diff_x =  -1 * front_x_dist
                # print("f(-,+) / a ", diff_x)
            else:
                diff_x =  -1 * front_x_dist
                # print("f(-,+) / b ", diff_x)

        elif (up_front_dist <0) and (down_front_dist < 0): #(-,-)
            if abs_up_front_dist > abs_down_front_dist:
                diff_x =  -1 * front_x_dist
                # print("f(-,-) / a ", diff_x)
            else:
                diff_x = front_x_dist
                # print("f(-,-) / b ", diff_x)

        difference_x = diff_x


    else: # back correction
        centor_dist = -centor_dist_back
        up_cropped_set = ups[up_back_idx-num_imgs:up_back_idx]
        down_cropped_set = downs[down_back_idx-num_imgs:down_back_idx]

        if (up_back_dist >= 0) and (down_back_dist >= 0): #(+,+)
            if abs_up_back_dist > abs_down_back_dist:
                diff_x =  back_x_dist       ### 8300
                # print("b(+,+) / a ", diff_x)         
            else:
                diff_x =  -1* back_x_dist   ### 3200 x 10800 o
                # print("b(+,+) / b ", diff_x)

        elif (up_back_dist > 0) and (down_back_dist < 0): #(+,-)
            if abs_up_back_dist > abs_down_back_dist:
                diff_x = back_x_dist
                # print("b(+,-) / a ", diff_x)
            else:
                diff_x = back_x_dist
                # print("b(+,-) / b ", diff_x)

        elif (up_back_dist < 0) and (down_back_dist > 0): #(-,+)
            if abs_up_back_dist > abs_down_back_dist:
                diff_x = -1*  back_x_dist
                # print("b(-,+) / a ", diff_x)
            else:
                diff_x =  -1* back_x_dist
                # print("b(-,+) / b ", diff_x)

        elif (up_back_dist <0) and (down_back_dist < 0): #(-,-)
            if abs_up_back_dist > abs_down_back_dist:
                diff_x = -1*  back_x_dist
                # print("b(-,-) / a ", diff_x)
            else:
                diff_x = back_x_dist
                # print("b(-,-) / b ", diff_x)

        difference_x = diff_x

    # print("output difference_x: ", difference_x)

    return up_cropped_set, down_cropped_set, difference_x, centor_dist

def make_cropped_updownSet_no_translate(ups, downs, up_Info, down_Info, is_Front):

    up_front_idx = up_Info[0]
    up_back_idx = up_Info[1]

    down_front_idx = down_Info[0]
    down_back_idx = down_Info[1]

    centor_dist = None
    difference_x = 0

    if is_Front:
        num_imgs = up_back_idx - up_front_idx
        centor_dist = up_Info[4] - up_front_idx

        up_cropped_set = ups[up_front_idx:up_front_idx+num_imgs]
        down_cropped_set = downs[down_front_idx:down_front_idx+num_imgs]
    else:
        num_imgs = down_back_idx - down_front_idx
        centor_dist = up_back_idx - up_Info[4]

        up_cropped_set = ups[up_back_idx-num_imgs:up_back_idx]
        down_cropped_set = downs[down_back_idx-num_imgs:down_back_idx]

    return up_cropped_set, down_cropped_set, difference_x, centor_dist

def find_vertical_lines(imgs):
    x_offset = 0.3
    y_offset = 0.3
    vertical_lines = []
    for idx in range(len(imgs)):
        temp_lines = []
        img = imgs[idx].copy()

        img = cv2.resize(img, dsize=(0, 0), fx=x_offset, fy=y_offset, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize=3)

        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

        if len(lines) != 0 :
            for line in lines:
                x1,y1,x2,y2 = line[0]
                if int(x2-x1) < 1 : # vertical line
                    temp_lines.append(( int(x1/x_offset) , int(y1/y_offset) , int(x2/x_offset) , int(y2/y_offset)))

        vertical_lines.append(temp_lines)

    return vertical_lines


# Do stitching 
def filtered_stitching_right(ups, downs, unit, x_trans, resize_scale, cont_size_switch):
    """ 
    This method does stitching.
    
    :input param: image set of upper and down camera, difference of x axis of up and down, resizing scale
    :return: stitched image of upper and down cam, average velocity information(average speed of the truck)
    """


    if len(ups) == len(downs):
        for idx in range(len(ups)):
            ups[idx] = cv2.resize(ups[idx], dsize=(0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
            downs[idx] = cv2.resize(downs[idx], dsize=(0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
    else:
        pass

    velocity_ret, velocity = calc_velocity_side(ups[0].shape[1], len(ups), cont_size_switch)
    indent = find_indent_Right(ups[0], downs[0])

    up_stitched_img = Stitching_ImgsR(ups, velocity, 1, indent)
    down_stitched_img = Stitching_ImgsR(downs, velocity, 1, indent)


    return up_stitched_img, down_stitched_img, velocity



# Do stitching
def filtered_stitching_left(ups, downs, unit, x_trans, resize_scale, cont_size_switch):
    """ 
    This method does stitching.
    
    :input param: image set of upper and down camera, difference of x axis of up and down, resizing scale
    :return: stitched image of upper and down cam, average velocity information(average speed of the truck)
    """

    if len(ups) == len(downs):
        for idx in range(len(ups)):
            ups[idx] = cv2.resize(ups[idx], dsize=(0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
            downs[idx] = cv2.resize(downs[idx], dsize=(0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
    else:
        pass


    velocity_ret, velocity = calc_velocity_side(ups[0].shape[1], len(ups), cont_size_switch)
    indent = find_indent_Left(ups[0], downs[0])

    up_stitched_img = Stitching_ImgsL(ups, velocity, 1, indent)
    down_stitched_img = Stitching_ImgsL(downs, velocity, 1, indent)


    return up_stitched_img, down_stitched_img, velocity





# Do stitching
def filtered_stitching_right_H(ups, downs, unit, x_trans, resize_scale):
    """ 
    This method does stitching.
    
    :input param: image set of upper and down camera, difference of x axis of up and down, resizing scale
    :return: stitched image of upper and down cam, average velocity information(average speed of the truck)
    """


    if len(ups) == len(downs):
        for idx in range(len(ups)):
            ups[idx] = cv2.resize(ups[idx], dsize=(0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
            downs[idx] = cv2.resize(downs[idx], dsize=(0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
    else:
        pass

    resize_time = time.time()
    up_H, up_avg_velocity = Find_Stitching_Info("Up", ups, unit)
    down_H, down_avg_velocity = Find_Stitching_Info("Down", downs, unit)

    H_time = time.time()
    # print("\tinstitching - H_time\t", H_time - resize_time)

    avg_velocity = (up_avg_velocity + down_avg_velocity) / 2
    if len(ups) == 0 or len(downs) == 0:
        return None, None, None

    indent = find_indent_Right(ups[0], up_H, downs[0], down_H)
    # print("avg_velocity: ", avg_velocity, "   indent: ", indent)
    up_stitched_img = Stitching_ImgsR_fromH(ups, up_H, avg_velocity, 1, indent)
    down_stitched_img = Stitching_ImgsR_fromH(downs, down_H, avg_velocity, 1, indent)


    return up_stitched_img, down_stitched_img, avg_velocity


# Do stitching 
def filtered_stitching_left_H(ups, downs, unit, x_trans, resize_scale):
    """ 
    This method does stitching.
    
    :input param: image set of upper and down camera, difference of x axis of up and down, resizing scale
    :return: stitched image of upper and down cam, average velocity information(average speed of the truck)
    """


    if len(ups) == len(downs):
        for idx in range(len(ups)):
            ups[idx] = cv2.resize(ups[idx], dsize=(0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
            downs[idx] = cv2.resize(downs[idx], dsize=(0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
    else:
        pass


    up_H, up_avg_velocity = Find_Stitching_Info("Up", ups, unit)
    down_H, down_avg_velocity = Find_Stitching_Info("Down", downs, unit)
    avg_velocity = (up_avg_velocity + down_avg_velocity) / 2
    if len(ups) == 0 or len(downs) == 0:
        return None, None, None

    indent = find_indent_Left(ups[0], downs[0])
    up_stitched_img = Stitching_ImgsL_fromH(ups, up_H, avg_velocity, 1, indent)
    down_stitched_img = Stitching_ImgsL_fromH(downs, down_H, avg_velocity, 1, indent)

    return up_stitched_img, down_stitched_img, avg_velocity





def find_indent_Right(up_img, down_img):

    up_border = get_indent_Right(up_img)
    down_border = get_indent_Right(down_img)

    if up_border >= down_border: return up_border
    else: return down_border

def find_indent_Left(up_img, down_img):

    up_border = get_indent_Left(up_img)
    down_border = get_indent_Left(down_img)
    if up_border >= down_border: return up_border
    else: return down_border

'''
make stitched image using warped-left video set
'''
def Stitching_side_left(first_frame_indent_view, imgs, indentation, container_size_idx):

    indent = indentation
    num_imgs = len(imgs)
    h_, w_, c_ = imgs[0].shape
    velocity_ret, velocity = calc_velocity_side(w_, num_imgs, container_size_idx)
    # if velocity_ret == False: return

    offset = first_frame_indent_view.shape[1] # w
    print("----------------------------------")
    for idx in range(0, num_imgs):
        if idx == 0:
            # blank = np.zeros( ( offset + (math.ceil(num_imgs) - 1)*velocity + (h_ - velocity - indent) , w_, c_), np.uint8)
            blank = np.zeros( (h_ , offset + (math.ceil(num_imgs))*velocity , c_), np.uint8)
            _, blank_weight, _ = blank.shape
            blank[:, :offset] = first_frame_indent_view
        #############################################################################################

        warped = imgs[idx][:, indent:velocity + indent]
        blank[:,offset + velocity*idx : offset + velocity*idx + warped.shape[1]] = warped

        #############################################################################################

        if False:
            blank_resize = cv2.resize(blank, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
            print(idx + 1, " / ", num_imgs, " ", warped.shape, " ", blank.shape, " ", velocity, " ", velocity * idx)
            cv2.imshow("warped", warped)
            cv2.imshow("blank", blank_resize)
            cv2.waitKey(0)

    return blank

'''
make stitched image using warped-right video set5
'''
def Stitching_side_right(first_frame_indent_view, imgs, indentation, container_size_idx):

    indent = indentation
    num_imgs = len(imgs)
    h_, w_, c_ = imgs[0].shape
    velocity_ret, velocity = calc_velocity_side(w_, num_imgs, container_size_idx)
    # if velocity_ret == False: return

    offset = first_frame_indent_view.shape[1] # w

    for idx in range(0, num_imgs):
        if idx == 0:
            # blank = np.zeros( ( offset + (math.ceil(num_imgs) - 1)*velocity + (h_ - velocity - indent) , w_, c_), np.uint8)
            blank = np.zeros( (h_ , offset + (math.ceil(num_imgs))*velocity , c_), np.uint8)
            _, blank_weight, _ = blank.shape
            blank[:, blank_weight - offset:] = first_frame_indent_view
        #############################################################################################

        warped = imgs[idx][:, imgs[idx].shape[1] -  velocity - indent: imgs[idx].shape[1] - indent ]
        blank[:, blank_weight - offset - velocity*(idx) - warped.shape[1] : blank_weight - offset - velocity*(idx)] = warped

        #############################################################################################

        if False:
            blank_resize = cv2.resize(blank, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
            print(idx + 1, " / ", num_imgs, " ", warped.shape, " ", blank.shape, " ", velocity, " ", velocity * idx)
            cv2.imshow("warped", warped)
            cv2.imshow("blank", blank_resize)
            cv2.waitKey(0)

    return blank


# make stitched image using H
def Stitching_ImgsR_fromH(imgs, H, avg_velocity, unit, indent):

    """
    This method does stitching by using homography.
    The Stitching is done by moving images with the homography matrix as much as average velocity.
    
    :input param: image set of upper and down camera, homography info., average velocity
    :return: one stitched image(up or down)
    """

    avg_vel = int(avg_velocity)

    while(True):
        if (len(imgs)-1) % unit != 0: imgs.pop()
        else: break

    num_imgs = len(imgs)
    for idx in range(0, num_imgs, unit):
        width = imgs[idx ].shape[0]
        height = imgs[idx ].shape[1]
        if idx == 0:
            init = cv2.warpPerspective(imgs[idx ], H, (width, height))
            blank = np.zeros( (init.shape[0],  (math.ceil(num_imgs) - 1)*avg_vel  +  (init.shape[1] - indent - avg_vel*unit) , init.shape[2]), np.uint8)

        _, blank_weight, _ = blank.shape

        ####################################################################################################################################

        if idx != (num_imgs - 1):
            warped = cv2.warpPerspective(imgs[idx ], H, (width, height))[:,-(avg_vel*unit + indent):-indent]
            blank[:, blank_weight-avg_vel*(idx+1) : blank_weight-avg_vel*(idx)] = warped
        else:
            warped = cv2.warpPerspective(imgs[idx ], H, (width, height))[:,:-(avg_vel*unit + indent)]
            blank[:, : warped.shape[1]] = warped

        ####################################################################################################################################

        # if idx != (num_imgs - 1):
        #     warped = cv2.warpPerspective(imgs[idx ], H, (width, height))[:,indent:indent + avg_vel*unit]
        #     blank[:,avg_vel * idx : avg_vel * idx + warped.shape[1]] = warped
        # elif idx == (num_imgs - 2) : continue
        # else:
        #     warped = cv2.warpPerspective(imgs[idx ], H, (width, height))

        #     for i in range(warped.shape[0]):
        #         for j in range(warped.shape[1]):
        #             if (warped.item(i, j, 0) != 0 and warped.item(i, j, 1) != 0 and warped.item(i, j, 2) != 0 ):
        #                 blank.itemset(i, j + avg_vel * (idx-2), 0, warped.item(i, j, 0))
        #                 blank.itemset(i, j + avg_vel * (idx-2), 1, warped.item(i, j, 1))
        #                 blank.itemset(i, j + avg_vel * (idx-2), 2, warped.item(i, j, 2))

        ####################################################################################################################################



        if False:
            blank_resize = cv2.resize(blank, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            print(idx + 1, " / ", num_imgs, " ", warped.shape, " ", blank.shape, " ", avg_vel, " ", avg_vel * idx)
            cv2.imshow("warped", warped)
            cv2.imshow("blank", blank_resize)
            cv2.waitKey(0)

    return blank



# make stitched image using H
def Stitching_ImgsL_fromH(imgs, H, avg_velocity, unit, indent):

    """
    This method does stitching by using homography.
    The Stitching is done by moving images with the homography matrix as much as average velocity.
    
    :input param: image set of upper and down camera, homography info., average velocity
    :return: one stitched image(up or down)
    """

    avg_vel = int(avg_velocity)

    while(True):
        if (len(imgs)-1) % unit != 0:
            imgs.pop()
        else: break

    num_imgs = len(imgs)
    for idx in range(0, num_imgs, unit):
        if idx == 0:
            init = cv2.warpPerspective(imgs[idx], H, (imgs[idx].shape[0], imgs[idx].shape[1]))
            blank = np.zeros((init.shape[0],  (math.ceil(num_imgs) - 1)*avg_vel  +  (init.shape[1] - indent - avg_vel*unit) , init.shape[2]), np.uint8)



        ####################################################################################################################################

        if idx != (num_imgs - 1):
            warped = cv2.warpPerspective(imgs[idx ], H, (imgs[idx ].shape[0], imgs[idx ].shape[1]))[:,indent:avg_vel*unit + indent]
            blank[:,avg_vel * idx : avg_vel * idx + warped.shape[1]] = warped
        else:
            warped = cv2.warpPerspective(imgs[idx ], H, (imgs[idx ].shape[0], imgs[idx ].shape[1]))[:,avg_vel*unit + indent:]
            blank[:,avg_vel * idx : ] = warped

        ####################################################################################################################################

        if  True:
            blank_resize = cv2.resize(blank, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            print(idx + 1, " / ", num_imgs, " ", warped.shape, " ", blank.shape, " ", avg_vel, " ", avg_vel * idx)
            cv2.imshow("warped", warped)
            cv2.imshow("blank", blank_resize)
            cv2.waitKey(0)

    return blank


# make stitched image using H
def Stitching_ImgsR(imgs, avg_velocity, unit, indent):

    """
    This method does stitching by using homography.
    The Stitching is done by moving images with the homography matrix as much as average velocity.
    
    :input param: image set of upper and down camera, homography info., average velocity
    :return: one stitched image(up or down)
    """

    avg_vel = int(avg_velocity)

    while(True):
        if (len(imgs)-1) % unit != 0: imgs.pop()
        else: break

    num_imgs = len(imgs)
    for idx in range(0, num_imgs, unit):
        width = imgs[idx ].shape[0]
        height = imgs[idx ].shape[1]
        if idx == 0:
            init = imgs[idx]
            blank = np.zeros( (init.shape[0],  (math.ceil(num_imgs) - 1)*avg_vel  +  (init.shape[1] - indent - avg_vel*unit) , init.shape[2]), np.uint8)

        _, blank_weight, _ = blank.shape

        ####################################################################################################################################

        if idx != (num_imgs - 1):
            warped = imgs[idx][:,-(avg_vel*unit + indent):-indent]
            blank[:, blank_weight-avg_vel*(idx+1) : blank_weight-avg_vel*(idx)] = warped
        else:
            warped = imgs[idx][:,:-(avg_vel*unit + indent)]
            blank[:, : warped.shape[1]] = warped

        ####################################################################################################################################

        if False:
            blank_resize = cv2.resize(blank, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            print(idx + 1, " / ", num_imgs, " ", warped.shape, " ", blank.shape, " ", avg_vel, " ", avg_vel * idx)
            cv2.imshow("warped", warped)
            cv2.imshow("blank", blank_resize)
            cv2.waitKey(0)

    return blank



# make stitched image using H
def Stitching_ImgsL(imgs, avg_velocity, unit, indent):

    """
    This method does stitching by using homography.
    The Stitching is done by moving images with the homography matrix as much as average velocity.
    
    :input param: image set of upper and down camera, homography info., average velocity
    :return: one stitched image(up or down)
    """

    avg_vel = int(avg_velocity)

    while(True):
        if (len(imgs)-1) % unit != 0:
            imgs.pop()
        else: break

    num_imgs = len(imgs)
    for idx in range(0, num_imgs, unit):
        if idx == 0:
            init = imgs[idx]
            blank = np.zeros((init.shape[0],  (math.ceil(num_imgs) - 1)*avg_vel  +  (init.shape[1] - indent - avg_vel*unit) , init.shape[2]), np.uint8)



        ####################################################################################################################################

        if idx != (num_imgs - 1):
            warped = imgs[idx][:,indent:avg_vel*unit + indent]
            blank[:,avg_vel * idx : avg_vel * idx + warped.shape[1]] = warped
        else:
            warped = imgs[idx][:,avg_vel*unit + indent:]
            blank[:,avg_vel * idx : ] = warped

        ####################################################################################################################################

        if  False:
            blank_resize = cv2.resize(blank, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            print(idx + 1, " / ", num_imgs, " ", warped.shape, " ", blank.shape, " ", avg_vel, " ", avg_vel * idx)
            cv2.imshow("warped", warped)
            cv2.imshow("blank", blank_resize)
            cv2.waitKey(0)

    return blank

def indent_count(img):
    hole_count = 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if ( img.item(i, j, 0) == 0 ) and ( img.item(i, j, 1) == 0 ) and ( img.item(i, j, 2) == 0 ):
                hole_count += 1

    return hole_count



def get_indent_Right(img):
    candidate_array = get_indent_candidate_array(img.shape[0], 8)

    hole_count_array = []
    hole_count = 0
    for candidiate_h in candidate_array:
        for j in range(img.shape[1]):
            if ( img.item(candidiate_h, img.shape[1]-1-j, 0) == 0 ) and ( img.item(candidiate_h, img.shape[1]-1-j, 1) == 0 ) and ( img.item(candidiate_h, img.shape[1]-1-j, 2) == 0 ):
                hole_count += 1
            else:
                hole_count_array.append(hole_count)
                hole_count = 0
                break

    median_hole_cnt = statistics.median(hole_count_array)
    for hole in hole_count_array:
        if (abs(hole - median_hole_cnt) > 50) or (hole > img.shape[1]/2):
            del[hole]
    max_hole_cnt = max(hole_count_array)

    hole_count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if ( img.item(i, img.shape[1]-1-j, 0) == 0 ) and ( img.item(i, img.shape[1]-1-j, 1) == 0 ) and ( img.item(i, img.shape[1]-1-j, 2) == 0 ):
                hole_count += 1
                if hole_count > img.shape[1] / 2:
                    hole_count = 0
                    break
            else:
                if hole_count > max_hole_cnt and hole_count < 2*max_hole_cnt: max_hole_cnt = hole_count
                hole_count = 0
                break

    return max_hole_cnt


def get_indent_Left(img):
    candidate_array = get_indent_candidate_array(img.shape[0], 8)

    hole_count_array = []
    hole_count = 0
    for candidate_h in candidate_array:
        for j in range(img.shape[1]):
            if ( img.item(candidate_h, j, 0) == 0 ) and ( img.item(candidate_h, j, 1) == 0 ) and ( img.item(candidate_h, j, 2) == 0 ):
                hole_count += 1
                if hole_count > img.shape[1]/2:
                    hole_count = 0
                    break
            else:
                hole_count_array.append(hole_count)
                hole_count = 0
                break

    median_hole_cnt = statistics.median(hole_count_array)
    for hole in hole_count_array:
        if (abs(hole - median_hole_cnt) > 50) or (hole > img.shape[1]/2):
            del[hole]
    max_hole_cnt = max(hole_count_array)

    hole_count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if ( img.item(i, j, 0) == 0 ) and ( img.item(i, j, 1) == 0 ) and ( img.item(i, j, 2) == 0 ):
                hole_count +=1
                if hole_count > img.shape[1]/2:
                    hole_count = 0
                    break
            else:
                if hole_count > max_hole_cnt and hole_count < 2*max_hole_cnt: max_hole_cnt = hole_count
                hole_count = 0
                break

    return max_hole_cnt





def get_indent_candidate_array(height, number):
    candidate_array = []
    for index in range(1, number):
        candidate_array.append(int(index/number*height))
    return candidate_array



# calculate the Stitching info (H, avg speed, stitching idx)
def Find_Stitching_Info(loc_flag, imgs, cut_unit, x_min= None, x_max= None, y_min= None, y_max = None):

    """
    This function looks for information to use when performing stitching.
    
    :input param: flag(upper or down cam), iamge list, threshold for finding average velocity.
    :return: homograpy set, average velocity, idx that has largest matching pairs.
    """

    # find matches <<
    matches_set = []
    past_key_set = []
    cur_key_set = []
    for idx in range(0, len(imgs) - 1):
        past = imgs[idx]
        current = imgs[idx+1]

        matches, keyPoints1, keyPoints2 = find_Match_Set(current, past)
        matches_set.append(matches)
        cur_key_set.append(keyPoints1)
        past_key_set.append(keyPoints2)


    # decide velocity
    avg_velocity = get_avg_Velocity(matches_set, cur_key_set, past_key_set, cut_unit)
    # print("avg_velocity: ", avg_velocity)


    # calc H and warp
    max_match_idx = -1
    fmatch_max = -1
    H_set = []
    ransac_cadidate = []
    for idx in range(len(imgs) - 1):
        past = imgs[idx]
        current = imgs[idx + 1]
        H, status, fmatch_num, ransac_cadidate = Find_filtered_H(loc_flag, current, past, matches_set[idx], cur_key_set[idx], past_key_set[idx], avg_velocity, cut_unit, ransac_cadidate)
        H_set.append(H)
        if fmatch_max < fmatch_num:
            fmatch_max = fmatch_num
            max_match_idx = idx



    selected_H = ransac_for_homography(ransac_cadidate, 10, 5)

    # # inserting another H set info.
    # try:    
    #     # if number of the largest matching pair is small than some constant value, find other good homography depending on the velocity
    #     if fmatch_max < 25: 
    #         if loc_flag == "Up":
    #             if os.path.isfile(os.path.join("../data/H/Up/", str(int(avg_velocity))+".npy")):
    #                 H_set.append(np.load(os.path.join("../data/H/Up/", str(int(avg_velocity))+".npy")))
    #             else:
    #                 H_set.append(np.load(os.path.join("../data/H/Up/", "31.npy")))
    #         elif loc_flag == "Down":
    #             if os.path.isfile(os.path.join("../data/H/Down/", str(int(avg_velocity))+".npy")):
    #                 H_set.append(np.load(os.path.join("../data/H/Down/", str(int(avg_velocity))+".npy")))
    #             else:                    
    #                 H_set.append(np.load(os.path.join("../data/H/Down/", "31.npy")))
    #         max_match_idx = -1
    #     elif fmatch_max >= 25:
    #         # Save the H set for failure Case
    #         np.save(os.path.join("../data/H/"+loc_flag, str(int(avg_velocity))) , H_set[max_match_idx])

    # except Exception as e:
    #     print("H set npy loading failed: ", e)

    return selected_H, avg_velocity


def ransac_for_homography(ransac_candidate, num_of_top_rank, num_of_selection):
    ransac_selected = []

    ### slice the RANSAC ###
    ransac_candidate = sorted(ransac_candidate, key = lambda k: k[3])
    if len(ransac_candidate) > num_of_top_rank:
        ransac_candidate = ransac_candidate[-num_of_top_rank:]
        ransac_selected = return_num_of_selection(num_of_top_rank, num_of_selection)

    else:
        num_of_top_rank = len(ransac_candidate)
        num_of_selection = len(ransac_candidate)
        for idx in range(num_of_selection):
            ransac_selected.append(idx)


    best_idx = -1
    best_residual = sys.maxsize
    for idx in ransac_selected:
        residual = calc_residual(ransac_candidate, idx)

        if best_residual > residual:
            best_idx = idx
            best_residual = residual

    return ransac_candidate[best_idx][2]



def return_num_of_selection(num_of_top_rank, num_of_selection):
    selected_list = []
    for i in range(num_of_selection):
        candidate = random.randrange(0, num_of_top_rank - 1)
        while candidate in selected_list:
            candidate = random.randrange(0, num_of_top_rank - 1)
        selected_list.append(candidate)

    return selected_list


def get_ransac_candidate_H(ransac_candidate):
    H_ransac = np.eye(3)
    for candidate in ransac_candidate:
        H_ransac = H_ransac * candidate[2]
    H_ransac = H_ransac / len(ransac_candidate)

    return ransac_candidate[0][2]


def calc_residual(ransac_candidate, idx):
    selected_H = ransac_candidate[idx][2]

    residual = 0
    for candidate in ransac_candidate:
        res_dst = H_transform(candidate[0], selected_H)
        dst = candidate[1]

        # print(len(res_dst), len(dst))
        # print(res_dst, " ", dst)
        for idx in range(len(res_dst)):
            residual += math.sqrt(pow(res_dst[idx][0] - dst[idx][0], 2) + pow(res_dst[idx][1] - dst[idx][1], 2))

    residual = residual/len(ransac_candidate)
    return residual





# feature matching
def find_Match_Set(from_, to_):

    """
    This method finds matching pairs by using ORB feature.
    
    :input param: two continuous images
    :return: matching pair info, keypoitns of each image
    """

    detector = cv2.ORB_create(300, 1.2, 1)
    # detector = cv2.ORB_create(500, 1.4, 4)

    keyPoints1, descriptors1 = detector.detectAndCompute(from_, None)
    keyPoints2, descriptors2 = detector.detectAndCompute(to_, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if len(keyPoints1) != 0 and len(keyPoints2) != 0:
        # Match descriptors.
        matches = bf.match(descriptors1, descriptors2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
    else:
        matches = []


    if False:
        matched_img = cv2.drawMatches(from_,keyPoints1,to_,keyPoints2,matches, None,flags=2)
        matched_img = cv2.resize(matched_img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("matched img", matched_img)
        cv2.waitKey(0)

    return matches, keyPoints1, keyPoints2


# find avg velocity
def get_avg_Velocity(matches_set, cur_key_set, past_key_set, cut_unit):

    """
    This method calculates the average distance between matching pairs.
    
    :input param: matching info., keypoints of current image, keypoints of previous image, and threshold for ignoring.
    :return: average distacne(average velocity)
    """
    x_Diff_set = []
    key1_set = []
    key2_set = []

    for idx in range(len(matches_set)):
        matches = matches_set[idx]
        keyPoints1 = cur_key_set[idx]
        keyPoints2 = past_key_set[idx]

        if len(matches) >= 4:
            keyPoints1_pt = ([keyPoints1[m.queryIdx].pt for m in matches])
            keyPoints2_pt = ([keyPoints2[m.trainIdx].pt for m in matches])

            x_thres = 10 * cut_unit
            y_thres = 10 * cut_unit
            for itr in range(len(keyPoints1_pt)):

                if (math.sqrt(math.pow(keyPoints1_pt[itr][1] - keyPoints2_pt[itr][1], 2)) < y_thres
                        and (math.sqrt( math.pow(keyPoints1_pt[itr][0] - keyPoints2_pt[itr][0], 2) ) > x_thres)):
                    x_Diff_set.append(math.sqrt( math.pow(keyPoints1_pt[itr][0] - keyPoints2_pt[itr][0], 2) ))

    avg_velocity = np.median(x_Diff_set)

    return avg_velocity

# find filtered H matrix
def Find_filtered_H(loc_flag, from_, to_, matches, keyPoints1, keyPoints2, avg_velocity, cut_unit, ransac_cadidate):
    """ 
    This method does filter bad matching pairs and calculate homography matrix by filtered featrues
    If the feature did not shift from side to side over the threshold, it is highly likely to be a feature extracted from a fixed background, thus excluding that feature.
    
    :input param: matching info., keypoints of current image, keypoints of previouse image, and threshold for ignoring.
    :return: Homography matrix, and length of filterd matching pair
    """

    num_filtered_matches = 0
    if len(matches) >= 4:

        keyPoints1_pt = ([keyPoints1[m.queryIdx].pt for m in matches])
        keyPoints2_pt = ([keyPoints2[m.trainIdx].pt for m in matches])

        x_thres = 10 * cut_unit
        y_thres = 10 * cut_unit
        mid_thres = 10 * cut_unit


        if loc_flag == "Up":
            indent_thresh = from_.shape[0] / 3
            goodIndex = [i for i in range(len(matches)) if (math.sqrt( math.pow(keyPoints1_pt[i][1] - keyPoints2_pt[i][1], 2) ) < y_thres)
                        and (math.sqrt( math.pow(keyPoints1_pt[i][0] - keyPoints2_pt[i][0], 2) ) > x_thres)
                        and math.fabs((math.sqrt( math.pow(keyPoints1_pt[i][0] - keyPoints2_pt[i][0], 2) ) - avg_velocity)) < mid_thres
                        and keyPoints1_pt[i][1] > indent_thresh and keyPoints2_pt[i][1] > indent_thresh]
        if loc_flag == "Down":
            indent_thresh = from_.shape[0] / 4 * 3
            goodIndex = [i for i in range(len(matches)) if (math.sqrt( math.pow(keyPoints1_pt[i][1] - keyPoints2_pt[i][1], 2) ) < y_thres)
                        and (math.sqrt( math.pow(keyPoints1_pt[i][0] - keyPoints2_pt[i][0], 2) ) > x_thres)
                        and math.fabs((math.sqrt( math.pow(keyPoints1_pt[i][0] - keyPoints2_pt[i][0], 2) ) - avg_velocity)) < mid_thres
                        and keyPoints1_pt[i][1] < indent_thresh and keyPoints2_pt[i][1] < indent_thresh]

        src = np.float32([keyPoints1_pt[i] for i in goodIndex])
        dst = np.float32([keyPoints2_pt[i] for i in goodIndex])


        num_filtered_matches = len(goodIndex)

        #-debug
        if False:
            print(num_filtered_matches)
            matched_img = draw_matches_W(from_, to_, src, dst)
            cv2.imshow("filttered matches", matched_img)
            cv2.waitKey(0)

        if num_filtered_matches >= 4:
            H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            ransac_cadidate.append([src, dst, H, num_filtered_matches])

        else:
            H, status = None, None
    else:
        H, status = None, None


    return H, status, num_filtered_matches, ransac_cadidate


def H_transform(srcs, H):

    dsts = []
    for src in srcs:
        dst = np.dot(H, [src[0], src[1], 1])
        dsts.append([int(dst[0]/dst[2]), int(dst[1]/dst[2])])

    return dsts

# draw match set Width
def draw_matches_H(left, right, pt_left, pt_right):

    if len(pt_left) != len(pt_right):
        print("err] draw_matches_W")

    L_h, L_w = left.shape[:2]

    dotted_Limg = left.copy()
    dotted_Rimg = right.copy()

    for L_idx in range(len(pt_left)):
        x_l = int(pt_left[L_idx][0])
        y_l = int(pt_left[L_idx][1])
        dotted_Limg = cv2.line(dotted_Limg, (x_l, y_l) , (x_l, y_l), (0,255,0), 3)

        x_r = int(pt_right[L_idx][0])
        y_r = int(pt_right[L_idx][1])
        dotted_Rimg = cv2.line(dotted_Rimg, (x_r, y_r), (x_r, y_r), (0, 0, 255), 3)

    matched_view = cv2.vconcat([dotted_Limg, dotted_Rimg])
    for L_idx in range(len(pt_left)):
        x_l = int(pt_left[L_idx][0])
        y_l = int(pt_left[L_idx][1])
        x_r = int(pt_right[L_idx][0])
        y_r = int(pt_right[L_idx][1])

        matched_view = cv2.line(matched_view, (x_l, y_l), (x_r, y_r + L_h), (255, 0, 0), 1)

    return matched_view


# draw match set Height
def draw_matches_W(left, right, pt_left, pt_right):

    if len(pt_left) != len(pt_right):
        print("err] draw_matches_H")

    L_h, L_w = left.shape[:2]

    dotted_Limg = left.copy()
    dotted_Rimg = right.copy()

    for L_idx in range(len(pt_left)):
        x_l = int(pt_left[L_idx][0])
        y_l = int(pt_left[L_idx][1])
        dotted_Limg = cv2.line(dotted_Limg, (x_l, y_l) , (x_l, y_l), (0,255,0), 3)

        x_r = int(pt_right[L_idx][0])
        y_r = int(pt_right[L_idx][1])
        dotted_Rimg = cv2.line(dotted_Rimg, (x_r, y_r), (x_r, y_r), (0, 0, 255), 3)


    matched_view = cv2.hconcat([dotted_Limg, dotted_Rimg])
    for L_idx in range(len(pt_left)):
        x_l = int(pt_left[L_idx][0])
        y_l = int(pt_left[L_idx][1])
        x_r = int(pt_right[L_idx][0])
        y_r = int(pt_right[L_idx][1])

        matched_view = cv2.line(matched_view, (x_l, y_l), (x_r + L_w, y_r), (255, 0, 0), 1)

    return matched_view

# resize for stitching
def resize_for_stitching(up, down):
    up_rows, up_cols, up_channel = up.shape
    down_rows, down_cols, down_channel = down.shape

    if up_cols > down_cols:
        up = cv2.resize(up, dsize=(down_cols, down_rows), interpolation=cv2.INTER_AREA)
    else:
        down = cv2.resize(down, dsize=(up_cols, up_rows), interpolation=cv2.INTER_AREA)

    return up, down

def updown_Stitching(up_stitched, down_stitched, avg_velocity, min_ratio = 0.8, max_ratio = 0.8):

    """
    This method finds where container is and return that index information of the image list.
    
    :input param: image list(before) of the truck and information of network #3
    :return: index information that indicates where the container is.
    """


    # # resize for extracting the features
    up_stitched = cv2.resize(up_stitched, dsize=(0, 0), fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    down_stitched = cv2.resize(down_stitched, dsize=(0, 0), fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)

    # set value
    up_h, up_w, _ = up_stitched.shape
    down_h, down_w, _ = down_stitched.shape

    up_template = up_stitched[ int(up_h * min_ratio)  :up_h,:]
    down_template = down_stitched[:down_h - int(down_h * max_ratio) ,:]



    # find match set using ORB
    matches, keyPoints1, keyPoints2 = find_Match_Set(up_template, down_template)
    keyPoints1_pt = ([keyPoints1[m.queryIdx].pt for m in matches])
    keyPoints2_pt = ([keyPoints2[m.trainIdx].pt for m in matches])
    median_x_distance = 0
    median_y_distance = 11
    if len(keyPoints1_pt) != 0 and len(keyPoints2_pt) != 0:
        # feature filtering and find median distance using x movement
        goodIndex_1 = [i for i in range(len(matches)) if (math.sqrt( math.pow(keyPoints1_pt[i][0] - keyPoints2_pt[i][0], 2) ) < (avg_velocity))]
        if len(goodIndex_1) != 0:
            keyPoints1_pt_x_filter = np.float32([keyPoints1_pt[i] for i in goodIndex_1])
            keyPoints2_pt_x_filter = np.float32([keyPoints2_pt[i] for i in goodIndex_1])

            x_dist_set = []
            y_dist_set = []
            for idx in range(len(keyPoints1_pt_x_filter)):
                x_dist_set.append( keyPoints1_pt_x_filter[idx][0] - keyPoints2_pt_x_filter[idx][0] ) # x
                y_dist_set.append( keyPoints1_pt_x_filter[idx][1] - keyPoints2_pt_x_filter[idx][1] ) # y

                median_x_distance = int(np.median(x_dist_set))
                median_y_distance = int(np.median(y_dist_set))



    # make up-down stitched image
    # x_translate = -1 * median_x_distance
    # M = np.float32([[1, 0, x_translate], [0, 1, 0]]) 
    # up_stitched = cv2.warpAffine(up_stitched, M, (up_w, up_h))

    y_trans = (down_h - int(down_h * max_ratio)) - median_y_distance
    if up_stitched.shape[1] > down_stitched.shape[1]:
        blank = np.zeros((up_stitched.shape[0] + down_stitched.shape[0] - y_trans, up_stitched.shape[1] , 3), np.uint8)

    else:
        blank = np.zeros((up_stitched.shape[0] + down_stitched.shape[0] - y_trans, down_stitched.shape[1] , 3), np.uint8)

    blank[:up_stitched.shape[0],: up_stitched.shape[1]] = up_stitched
    blank[up_stitched.shape[0]: ,: down_stitched.shape[1]] = down_stitched[y_trans:down_stitched.shape[0], :]



    return blank

'''
calc velocity using own velocity model fomula below
    [container_length = container_velocity(num_of_unit - 1) + container_width]
'''
def calc_velocity_side(cropped_w, num_imgs, container_size_idx):
    velocity = int((get_container_length_pixel_side(container_size_idx) - cropped_w) / (num_imgs - 1))
    if velocity <= 0: return False, velocity
    return True, velocity

'''
calc velocity using own velocity model fomula below
    [container_length = container_velocity(num_of_unit - 1) + container_width]
'''
def calc_velocity_top(cropped_h, num_imgs, container_size_idx):
    velocity = int((get_container_length_pixel_top(container_size_idx) - cropped_h) / (num_imgs - 1))
    if velocity <= 0: return False, velocity
    return True, velocity




'''
calc pixel coordinate container length using transformation 
between absolute coordinate system and pixel coordinate system
(ref: https://en.wikipedia.org/wiki/Intermodal_container)
'''
def get_container_length_pixel_top(container_size_idx):

    pixel_l = -1
    if container_size_idx == 1: #45ft high cubic
        pixel_l = (13.716/2.438) * 1371 # w_pixel
    elif container_size_idx == 2: #40ft
        pixel_l = (12.129/2.438) * 1228 # w_pixel
    elif container_size_idx == 3: #20ft twin
        pixel_l = (6.058*2/2.438) * 1050 # w_pixel
    elif container_size_idx == 4: #20ft single
        pixel_l = (6.058/2.438) * 1050 # w_pixel

    return pixel_l

'''
calc pixel coordinate container length using transformation 
between absolute coordinate system and pixel coordinate system
(ref: https://en.wikipedia.org/wiki/Intermodal_container)
'''
def get_container_length_pixel_side(container_size_idx):

    pixel_l = -1
    if container_size_idx == 1: #45ft high cubic
        pixel_l = (13.716/2.896) * 1250 # w_pixel
    elif container_size_idx == 2: #40ft
        pixel_l = (12.129/2.896) * 1250 # w_pixel
    elif container_size_idx == 3: #20ft twin
        pixel_l = (6.058*2/2.591) * 1118 # w_pixel
    elif container_size_idx == 4: #20ft single
        pixel_l = (6.058/2.591) * 1118 # w_pixel

    return pixel_l


def twin_division(stitched_result, start_idx, mid_idx, end_idx):
    img_rows = stitched_result.shape[0]
    border =  int(img_rows / 10 * 1)
    mid_ratio = (mid_idx - start_idx) / (end_idx - start_idx)


    front_stitched_result = stitched_result[:int(img_rows * mid_ratio) + border,:]
    back_stitched_result = stitched_result[int(img_rows * mid_ratio) - border:,:]

    return front_stitched_result, back_stitched_result

def twin_division_left(stitched_result, start_idx, mid_idx, end_idx):
    img_cols = stitched_result.shape[1]
    border =  int(img_cols / 10 * 1)
    mid_ratio = (mid_idx - start_idx) / (end_idx - start_idx)

    front_stitched_result = stitched_result[:,:int(img_cols * mid_ratio) + border]
    back_stitched_result = stitched_result[:, int(img_cols * mid_ratio) - border:]

    return front_stitched_result, back_stitched_result

def twin_division_right(stitched_result, start_idx, mid_idx, end_idx):
    img_cols = stitched_result.shape[1]
    border =  int(img_cols / 10 * 1)
    mid_ratio = (mid_idx - start_idx) / (end_idx - start_idx)

    front_stitched_result = stitched_result[:,:int(img_cols * mid_ratio) + border]
    back_stitched_result = stitched_result[:, int(img_cols * mid_ratio) - border:]

    return front_stitched_result, back_stitched_result

def frame_set_H_transformation(frame_set, H):
    H_frame_set = []
    for idx in range(len(frame_set)):
        frame = frame_set[idx]
        img_HTr = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))
        H_frame_set.append(img_HTr)

    return H_frame_set



def crop_frames(frame_set, start_idx, end_idx):
    cropped_set = frame_set[start_idx : end_idx]
    return cropped_set

def frame_show(frame_set, fps):

    for idx in range(len(frame_set)):
        frame = frame_set[idx]

        frame_resize = cv2.resize(frame, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("frame_resize", frame_resize)
        key = cv2.waitKey(fps) & 0xFF
        if key == 27:#ESC
            break

'''
define roi using pre-defined w, y roi range
'''
def define_Roi_left(cropped_frame_set):
    cropped_roi_frame_set = []
    h_, w_, c_ =  cropped_frame_set[0].shape

    for idx in range(len(cropped_frame_set)):


        process_frame = cropped_frame_set[idx][
            :,  #y 
            int(w_*5/10):int(w_)  #x
            ]

        cropped_roi_frame_set.append(process_frame)

    first_frame_indent_view =  cropped_frame_set[0][
            :,  #y
            # int(w_*5/10):int(w_)  #x
            :int(w_/10*5)  #x
            ]

    return first_frame_indent_view, cropped_roi_frame_set

def define_Roi_right(cropped_frame_set):
    cropped_roi_frame_set = []
    h_, w_, c_ =  cropped_frame_set[0].shape

    for idx in range(len(cropped_frame_set)):


        process_frame = cropped_frame_set[idx][
            :,  #y
            :int(w_/10*5)  #x
            ]

        cropped_roi_frame_set.append(process_frame)

    first_frame_indent_view =  cropped_frame_set[0][
            :,  #y
            int(w_*5/10):int(w_)  #x
            ]

    return first_frame_indent_view, cropped_roi_frame_set

'''
define roi using pre-defined w, y roi range
'''
def define_Roi(cropped_frame_set):
    cropped_roi_frame_set = []
    h_, w_, c_ =  cropped_frame_set[0].shape

    for idx in range(len(cropped_frame_set)):


        process_frame = cropped_frame_set[idx][
            int(h_/10*5):int(h_/10*9),  #y
            int(w_/10):int(w_/10*9)  #x
            ]

        cropped_roi_frame_set.append(process_frame)

    first_frame_indent_view =  cropped_frame_set[0][
            :int(h_/10*5),  #y
            int(w_/10):int(w_/10*9)  #x
            ]

    return first_frame_indent_view, cropped_roi_frame_set

'''
show frame video for debug
'''
def frame_show(frame_set, fps):

    for idx in range(len(frame_set)):
        frame = frame_set[idx]

        frame_resize = cv2.resize(frame, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("frame_resize", frame_resize)
        key = cv2.waitKey(fps) & 0xFF
        if key == 27:#ESC
            break


'''
make stitched image using warped-top video set
'''
def Stitching_top(first_frame_indent_view, imgs, indentation, container_size_idx):

    indent = indentation
    num_imgs = len(imgs)
    h_, w_, c_ = imgs[0].shape
    velocity_ret, velocity = calc_velocity_top(h_,num_imgs, container_size_idx)
    # if velocity_ret == False: return

    offset = first_frame_indent_view.shape[0]

    for idx in range(0, num_imgs):
        if idx == 0:
            blank = np.zeros( ( offset + (math.ceil(num_imgs) - 1)*velocity + (h_ - velocity - indent) , w_, c_), np.uint8)
            blank[:offset, :] = first_frame_indent_view
        #############################################################################################

        if idx != (num_imgs - 1):
            warped = imgs[idx][indent:velocity + indent, :]
            blank[offset + velocity*idx : offset + velocity*idx + warped.shape[0],:] = warped
        else:
            warped = imgs[idx][velocity + indent:, :]
            blank[offset + velocity*idx : ,:] = warped

        #############################################################################################

        if False:
            blank_resize = cv2.resize(blank, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
            print(idx + 1, " / ", num_imgs, " ", warped.shape, " ", blank.shape, " ", velocity, " ", velocity * idx)
            cv2.imshow("warped", warped)
            cv2.imshow("blank", blank_resize)
            cv2.waitKey(0)

    return blank
