'''
Class Auto Gate system
'''
import multiprocessing
from ctypes import *
import os
import numpy as np
import cv2
import glob
import time
from scipy.linalg import inv, norm
import copy
import sys

import json
from pprint import pprint
from datetime import datetime
import ftplib
import requests
import multiprocessing as mp
import statistics
import asyncio
import GPUtil
import psutil
import threading

import darknet
import LicensePlate
import ContainerPlate
import Seal
import Truck
import VAS_manager
import VAS_stitching
import VAS_viewer
import VAS_interface
from threading import Thread


class VAS_system():

    def __new__(self):

        print("[VAS system v0,1]")
        if not hasattr(self, 'instance'):
            self.instance = super(VAS_system, self).__new__(self)

        # mp.set_start_method('spawn')
        self.config_dir = "/home/admin/codes/VAS_1.1/data/config_L1.json"

        ### data processing ###
        self.LP_ = LicensePlate.LicensePlate()
        self.CP_ = ContainerPlate.ContainerPlate()
        self.SEAL_ = Seal.Seal()
        self.Truck = Truck.Truck()

        self.CamRetList = [False, False, False, False]

        ### interface ###
        self.interface = VAS_interface.VAS_interface(self.config_dir)

        ### manager ###
        self.manager = VAS_manager.VAS_manager()

        self.SENDING_FLAG, self.VIEWER_FLAG, self.DEBUG_FLAG, self.LANGUAGE_FLAG = self.manager.load_system_flag(
            self.config_dir)
        self.Net1_paths, self.Net2_paths, self.Net3_paths, self.Net4_paths, self.SealD_paths, self.SealR_paths = self.manager.load_Net_data(
            self.config_dir, self.LANGUAGE_FLAG)
        self.net1_thres = 0.3
        self.net2_thres = 0.3
        self.net3_thres = 0.3
        self.net4_thres = 0.3
        self.sealD_thres = 0.3
        self.sealR_thres = 0.3
        self.LoadNetworks(self, self.Net1_paths, self.net1_thres, self.Net2_paths, self.net2_thres, self.Net3_paths,
                          self.net3_thres, self.Net4_paths, self.net4_thres, self.SealD_paths, self.sealD_thres,
                          self.SealR_paths, self.sealR_thres)

        ### stitching ###
        self.stitcher = VAS_stitching.VAS_stitching(self.config_dir, 1, self.manager.HL, self.manager.HR,
                                                    self.manager.HL_inv, self.manager.HR_inv)

        self.cam8_json_data = self.manager.load_cam8_data(self.config_dir)
        self.cam5_json_data = self.manager.load_cam5_data(self.config_dir)

        self.LP_.SetStartSignalROI(self.cam8_json_data)
        self.CP_.SetEndSignalROI(self.cam5_json_data)

        self.default_skip, self.recog_skip, self.recog_skip_Tr, self.recog_skip_LR, self.max_frame_thres1, self.max_frame_thres2 = self.manager.load_system_data(
            self.config_dir)
        self.skip_frame_status_check = self.manager.status_check_interval(self.config_dir)  # 18000 seconds == 10 min
        ### viewer ###
        if self.VIEWER_FLAG:
            self.viewer = VAS_viewer.VAS_viewer(640, 480)
            self.viewer.set_view_position(0, 640, 1400, 0, 640)
        else:
            self.viewer = None


        self.indent_updown = 30
        return self.instance

    """

    Main operation of the VAS system

    """

    def system_Operation(self, Video_Info, MODE):
        truck_detection_mode = False
        back_boomBarreir_isClose = False

        # elif MODE == 1:
        cam_paths, start_frame = self.manager.load_IPvideo_path(Video_Info), 0
        # print(f"cam_path is {cam_paths}")

        ### Stitching
        # if self.STITCHING_FLAG:
        mMap_x, mMap_y = self.stitcher.set_Stitching_Info(1920,
                                                          1080)

        ### variable initialization
        resultSave_path = self.manager.load_resultSave_path(self.config_dir, Video_Info)
        self.resultSave_file = open(resultSave_path, 'w')

        ### for start end signal.
        start_queue = []
        end_queue = []
        boombarrier_queue = []
        prev_cp = [sys.maxsize, sys.maxsize]
        max_frame = self.max_frame_thres1

        ### for frame skip.
        front_skip = self.default_skip
        left_skip = 4
        right_skip = 4
        top_rear_skip = 4
        ### for processing time check
        time_Array = []




        # self.camStatus(self.CamRetList)
        video_stream = VideoStream(cam_paths)
        self.CamRetList = video_stream.status
        while True:
            if False in self.CamRetList:
                break
        self.gpuStatus()
        self.camStatus(self.CamRetList)
        frame_num = 0
        while self.CamRetList == [True] * 4:
            if self.DEBUG_FLAG: fpsCheckStart = time.time()
            self.cam1_image_Cr, self.cam3_image_Cr, self.cam5_image_Cr, self.cam8_image_Cr = video_stream.imgs
            frame_num += 1
            ### convert image using H
            F_img, Ru_img, Lu_img, Tr_img = self.manager.BCT_frame_convertor(self.stitcher, mMap_x, mMap_y,
                                                                             self.cam8_image_Cr, self.cam1_image_Cr,
                                                                             self.cam3_image_Cr, self.cam5_image_Cr)
            if self.VIEWER_FLAG: self.viewer.set_frames(F_img, Ru_img, Lu_img, Tr_img)
            self.interface.set_image_Size(F_img, Ru_img, Lu_img, Tr_img)

            ###
            # End signal Catcher / [ON] -> [OFF] / (end signal: precise ID exist, exceed the max frame)
            ###
            if (len(end_queue) >= 2) or (self.Truck.on_frame_cnt > max_frame):
                self.Truck.pre_end_frame_cnt += 1
            #     print(self.Truck.pre_end_frame_cnt)
            # if self.Truck.pre_end_frame_cnt > 50:  # wait two seconds after catching end signal with TR
                end_queue = []
                boombarrier_queue = []
                prev_cp = [sys.maxsize, sys.maxsize]

                truck_detection_mode = False

                front_skip = 1
                left_skip = 4
                right_skip = 4
                top_rear_skip = 4

                # if self.DEBUG_FLAG :
                #     self.manager.calc_avgTime(time_Array)
                #     time_Array = []
                #     self.manager.per_truck_time_Set.append(int(round(self.manager.avg_time * self.manager.avg_frame_length / 1000)))
                #     self.manager.per_truck_fps_Set.append(self.manager.avg_fps)
                #     print("(debug) AVG TIME:", round(statistics.mean(self.manager.per_truck_time_Set), 2), "[s] / Per Truck Time Set[s]: ", self.manager.per_truck_time_Set)
                #     print("(debug) AVG FPS:", round(statistics.mean(self.manager.per_truck_fps_Set), 2), "[fps] / Per Truck Fps Set[fps]: ", self.manager.per_truck_fps_Set)

            ###
            # change signal processing(end->start) & Start signal Catcher / [OFF]
            ###
            if truck_detection_mode == False:

                if (len(self.Truck.lp_candidates) > 0 or len(self.Truck.cp_candidates) > 0 or len(
                        self.Truck.cp_candidates2) > 0):

                    ### vote using histogram from LP, CP ###
                    self.interface.vote_histogram(self.Truck)

                    ### Chassis position ###
                    TF2CF_threshold = 0.15
                    if self.DEBUG_FLAG: self.Truck.find_ChassisPosition(TF2CF_threshold)

                    ### print & save VAS result ###
                    if self.DEBUG_FLAG: self.manager.print_Result(self.Truck)
                    if self.DEBUG_FLAG: self.interface.save_info2file(self.Truck, self.resultSave_file, ",")

                    ### send the data ###
                    if self.SENDING_FLAG:
                        self.interface.local_path = self.interface.folder_empty(self.interface.fileserver_dir, "VAS",
                                                                                self.interface.lane)
                        self.interface.remote_path = self.interface.filserver_url + self.interface.local_path.replace(
                            self.interface.fileserver_dir, "")
                        print("\t::LOCAL PATH: ", self.interface.local_path)
                        print("\t::REMOTE PATH: ", self.interface.remote_path, "\n")

                        # ctx = mp.get_context('fork')
                        output_folder, signal_date, signal_time = self.interface.make_recognition_output_folder(
                            self.Truck)
                        # self.do_makeVideo_output(output_folder, signal_date, signal_time)
                        ctx = mp.get_context('fork')
                        video_output_process = ctx.Process(name="SubProcess", target=self.do_makeVideo_output,
                                                           args=(output_folder, signal_date, signal_time,))
                        video_output_process.start()

                    ### stitching ###
                    if len(self.Truck.cp_candidates) > 0 or len(self.Truck.cp_candidates2) > 0:
                        cont_size_info = self.stitcher.get_container_size_info(self.Truck.containerID,
                                                                               self.Truck.is_twin_truck,
                                                                               self.Truck.is_40ft_truck)

                        stitch_result = self.stitcher.do_Stitching_BCT(
                            self.Truck.cam_Ruset, self.Truck.cam_Luset, self.Truck.cam_TRset, \
                            self.Truck.det_Ruset, self.Truck.det_Luset, self.Truck.det_TRset, self.Truck.rear_img,
                            self.manager.HTr, self.Truck.is_twin_truck, self.SEAL_, \
                            cont_size_info
                        )
                        if self.SENDING_FLAG:
                            self.do_makeStitch_output(stitch_result, output_folder, signal_date, signal_time)
                            # stitcing_output_process = mp.Process(name="SubProcess-Stitching", target=self.do_makeStitch_output, args=(stitch_result, output_folder, signal_date, signal_time,))
                            # stitcing_output_process.start()
                    # os.system('bash /home/admin/codes/VAS/src/vas_kill.sh')

                    ### deallocation truck information ###
                    back_boomBarreir_isClose = False
                    self.Truck.delete_info()
                    available_mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
                    print(f'Available Memory: {available_mem}')
                    if int(available_mem) < 20:
                        return False
                else:
                    if self.Truck.on_frame_cnt > max_frame:
                        self.Truck.delete_info()
                        back_boomBarreir_isClose = False

                ### Start signal Catcher ###
                if frame_num % front_skip == 0:
                    check_mode_flag = self.LP_.CheckMode(F_img, self.viewer)
                    # if key == 113:#q
                    #     print(check_mode_flag, start_queue)
                    #     check_mode_flag = True

                    if check_mode_flag == True:

                        ### Check if the previous truck has gone.
                        ret_top_rear, top_rear_detections, top_rear_roi_pts, _, _, end_queue, boombarrier_queue, prev_cp = self.CP_.TopRearDetection(
                            Tr_img, end_queue, boombarrier_queue, prev_cp, "toprear", self.Truck)
                        prev_truck_flag = False
                        if ret_top_rear:
                            prev_truck_flag = self.CP_.is_car_left(top_rear_detections, top_rear_roi_pts,
                                                                   prev_truck_flag)

                        if self.VIEWER_FLAG and top_rear_detections is not None:
                            self.viewer.Tr_img = darknet.draw_boxes(top_rear_detections, Tr_img, (0, 255, 0),
                                                                    top_rear_roi_pts)

                        if prev_truck_flag == False:
                            start_queue.insert(0, check_mode_flag)

                    ### prev car x / cur car o
                    if (start_queue == [True, True]):
                        if self.DEBUG_FLAG: print("\n\n(debug)", str(self.Truck.truck_cnt + 1), "cnt start frame :",
                                                  frame_num)

                        if self.SENDING_FLAG:
                            self.interface.VAS_Truckarrived()

                        ## start signal on
                        max_frame = self.max_frame_thres1

                        truck_detection_mode = True
                        start_queue = []

                        self.Truck.truck_cnt += 1
                        first_lp = False



            ###
            #  Process detection & recognition / [ON]
            ###
            elif truck_detection_mode == True:
                self.Truck.on_frame_cnt += 1
                self.Truck.start_end_frame_cnt += 1

                ###
                # TOP & REAR Detection | cam5
                ###
                if frame_num % top_rear_skip == 0:
                    self.Truck.cam_TRset.append(Tr_img)

                    ret_top_rear, top_rear_detections, top_rear_roi_pts, top_rear_cpid_imgs, cpid_img_labels, end_queue, boombarrier_queue, prev_cp = self.CP_.TopRearDetection(
                        Tr_img, end_queue, boombarrier_queue, prev_cp, "toprear", self.Truck)

                    ### if something is detected.
                    if ret_top_rear:
                        for i in range(len(top_rear_detections)):
                            ### for stitching NN alternatives
                            if top_rear_detections[0][0] == 'Cont_border' and float(top_rear_detections[0][1]) >= 30 and \
                                    self.Truck.det_TRset[0] == -1:  # and (roi_pts[0][0] + roi_pts[0][2])/2 < img_w/2:
                                self.Truck.det_TRset[0] = len(self.Truck.cam_TRset)
                            if top_rear_detections[0][0] == 'Cont_center' and float(top_rear_detections[0][1]) >= 30 and \
                                    self.Truck.det_TRset[1] == -1:  # and (roi_pts[0][0] + roi_pts[0][2])/2 < img_w/2:
                                self.Truck.det_TRset[1].append(len(self.Truck.cam_TRset))
                            if top_rear_detections[0][0] == 'Cont_border' and float(top_rear_detections[0][
                                                                                        1]) >= 30:  # and (roi_pts[0][0] + roi_pts[0][2])/2 < img_w/2:
                                self.Truck.det_TRset[2] = len(self.Truck.cam_TRset)

                        ### if the 'top' or 'back' cid area imgs exist.forward_NN
                        for i in range(len(top_rear_cpid_imgs)):
                            ### top rear skip change.
                            top_rear_skip = self.recog_skip_Tr

                            ### get CID from trimmed CID imgs.
                            top_cid = self.CP_.num_Identification(cpid_img_labels[i], top_rear_cpid_imgs[i])

                            ### save ID to truck module
                            if top_cid is not None:
                                if self.Truck.is_twin_truck is False:
                                    self.Truck.cp_candidates.append(top_cid)
                                    section = top_rear_detections[i][0]
                                    if section == "top":
                                        self.Truck.cp_candidates_top.append(top_cid)
                                        ### precise ID check
                                        self.Truck.hasPreciseID(top_cid, "top")
                                    elif section == "back":
                                        self.Truck.cp_candidates_back.append(top_cid)
                                        self.Truck.hasPreciseID(top_cid, "back")

                                else:
                                    self.Truck.cp_candidates2.append(top_cid)
                                    section = top_rear_detections[i][0]
                                    if section == "top":
                                        self.Truck.cp_candidates_top2.append(top_cid)
                                        self.Truck.hasPreciseID(top_cid, "top2")
                                    elif section == "back":
                                        self.Truck.cp_candidates_back2.append(top_cid)
                                        self.Truck.hasPreciseID(top_cid, "back2")

                    if self.VIEWER_FLAG and top_rear_detections is not None:
                        self.viewer.Tr_img = darknet.draw_boxes(top_rear_detections, Tr_img, (0, 255, 0),
                                                                top_rear_roi_pts)

                ###
                # LEFT Detection | cam3
                ###
                if frame_num % 1 == 0:
                    self.Truck.cam_Luset.append(Lu_img)

                    # ctx = mp.get_context('fork')
                    # video_output_process = ctx.Process(name="SubProcess", target=self.append_Lucam,
                    #                                    args=(Lu_img,))
                    # video_output_process.start()

                    self.Truck.cnt_Lu += 1
                #     self.Truck.cam_Ldset.append(cam4_undist)

                if frame_num % left_skip == 0:
                    # self.Truck.cam3_video.append(Lu_img)

                    # detection start
                    ret_left_U, left_U_detections, left_U_roi_pts, left_U_cid_imgs, is_gap = self.CP_.SideDetection(
                        Lu_img, "left_U", self.Truck)
                    m = multiprocessing.Manager()
                    self.Truck.cam3_det_info.append(self.extract_C_Pos(left_U_detections))  # for C_Pos
                    self.Truck.isTwinTruck(is_gap)

                    ### if something is detected.
                    if ret_left_U:
                        for i in range(len(left_U_detections)):
                            if left_U_detections[i][0] == 'Truck_head':
                                self.Truck.truck_front_flag = True
                            ### left skip change.
                            if left_U_detections[i][0] == 'Cont_border' or left_U_detections[i][0] == 'side':
                                left_skip = int(self.recog_skip_LR)

                            ### for stitching NN alternatives
                            if left_U_detections[0][0] == 'Cont_border' and float(left_U_detections[0][1]) >= 30 and \
                                    self.Truck.det_Luset[0] == -1:  # and (roi_pts[0][0] + roi_pts[0][2])/2 < img_w/2:
                                self.Truck.det_Luset[0] = len(self.Truck.cam_Luset)
                            if left_U_detections[0][0] == 'Cont_center' and float(left_U_detections[0][1]) >= 30 and \
                                    self.Truck.det_Luset[1] == -1:  # and (roi_pts[0][0] + roi_pts[0][2])/2 < img_w/2:
                                self.Truck.det_Luset[1].append(len(self.Truck.cam_Luset))

                            if left_U_detections[0][0] == 'Cont_border' and float(
                                    left_U_detections[0][1]) >= 30:  # and (roi_pts[0][0] + roi_pts[0][2])/2 < img_w/2:
                                self.Truck.det_Luset[2] = len(self.Truck.cam_Luset)

                        ### if the 'side' cid area imgs exist.
                        for i in range(len(left_U_cid_imgs)):

                            ### get CID from trimmed CID imgs.
                            left_cid = self.CP_.num_Identification("left_U", left_U_cid_imgs[i])

                            ### save ID to truck module
                            if left_cid is not None:
                                ### if 1st container.
                                if self.Truck.is_twin_truck is False:
                                    self.Truck.cp_candidates.append(left_cid)
                                    self.Truck.cp_candidates_left.append(left_cid)
                                    self.Truck.hasPreciseID(left_cid, "left")

                                ### if 2nd container.
                                else:
                                    self.Truck.is_twin_truck = True
                                    self.Truck.cp_candidates2.append(left_cid)
                                    self.Truck.cp_candidates_left2.append(left_cid)
                                    self.Truck.hasPreciseID(left_cid, "left2")

                    if self.VIEWER_FLAG and left_U_detections is not None:
                        self.viewer.Lu_img = darknet.draw_boxes(left_U_detections, Lu_img, (0, 255, 0), left_U_roi_pts)

                ###
                # RIGHT Detection | cam1
                ###
                if frame_num % 1 == 0:
                    self.Truck.cam_Ruset.append(Ru_img)
                    self.Truck.cnt_Ru += 1
                #     self.Truck.cam_Rdset.append(cam2_undist)

                if frame_num % right_skip == 0:
                    # self.Truck.cam1_video.append(Ru_img)

                    # detection start
                    ret_right_U, right_U_detections, right_U_roi_pts, right_U_cid_imgs, is_gap = self.CP_.SideDetection(
                        Ru_img, "right_U", self.Truck)
                    self.Truck.cam1_det_info.append(self.extract_C_Pos(right_U_detections))  # for C_Pos
                    self.Truck.isTwinTruck(is_gap)

                    ### if something is detected.
                    if ret_right_U:
                        for i in range(len(right_U_detections)):
                            if right_U_detections[i][0] == 'Truck_head':
                                self.Truck.truck_front_flag = True
                            ### right skip change.
                            if right_U_detections[i][0] == 'Cont_border' or right_U_detections[i][0] == 'side':
                                right_skip = int(self.recog_skip_LR)

                            ### for stitching NN alternatives
                            if right_U_detections[0][0] == 'Cont_border' and float(right_U_detections[0][1]) >= 30 and \
                                    self.Truck.det_Ruset[0] == -1:  # and (roi_pts[0][0] + roi_pts[0][2])/2 < img_w/2:
                                self.Truck.det_Ruset[0] = len(self.Truck.cam_Ruset)
                            if right_U_detections[0][0] == 'Cont_center' and float(right_U_detections[0][
                                                                                       1]) >= 30:  # and self.Truck.det_Ruset[1] == -1: # and (roi_pts[0][0] + roi_pts[0][2])/2 < img_w/2:
                                self.Truck.det_Ruset[1].append(len(self.Truck.cam_Ruset))
                            if right_U_detections[0][0] == 'Cont_border' and float(
                                    right_U_detections[0][1]) >= 30:  # and (roi_pts[0][0] + roi_pts[0][2])/2 < img_w/2:
                                self.Truck.det_Ruset[2] = len(self.Truck.cam_Ruset)

                        ### if the 'side' cid area imgs exist.
                        for i in range(len(right_U_cid_imgs)):

                            ### get CID from trimmed CID imgs.
                            right_cid = self.CP_.num_Identification("right_U", right_U_cid_imgs[i])

                            ### save ID to truck module
                            if right_cid is not None:
                                ### if 1st container.
                                if self.Truck.is_twin_truck is False:
                                    self.Truck.cp_candidates.append(right_cid)
                                    self.Truck.cp_candidates_right.append(right_cid)
                                    self.Truck.hasPreciseID(right_cid, "right")

                                ### if 2nd container.
                                else:
                                    self.Truck.is_twin_truck = True
                                    self.Truck.cp_candidates2.append(right_cid)
                                    self.Truck.cp_candidates_right2.append(right_cid)
                                    self.Truck.hasPreciseID(right_cid, "right2")

                    if self.VIEWER_FLAG and right_U_detections is not None:
                        self.viewer.Ru_img = darknet.draw_boxes(right_U_detections, Ru_img, (0, 255, 0),
                                                                right_U_roi_pts)

                ###
                #  FRONT Detection | cam8
                ###
                if frame_num % front_skip == 0:
                    # self.Truck.cam8_video.append(F_img)

                    # detection start
                    ret_front_L, front_L_detections, front_L_roi_pts, front_L_lpid_imgs, LP_type = self.LP_.FrontDetection(
                        F_img, self.Truck, 10000)

                    ### if something is detected
                    if ret_front_L:
                        for i in range(len(front_L_lpid_imgs)):
                            if first_lp == False:
                                ### if first lp is detected, max_frame changes from max_frame_thres_1 to max_frame_thres_2
                                max_frame = self.max_frame_thres2
                                self.Truck.on_frame_cnt = 0
                                first_lp = True

                            ### front skip change
                            front_skip = self.recog_skip

                            ### get LP ID from trimmed LP img.
                            lpid = self.LP_.num_Identification(LP_type, front_L_lpid_imgs[i], self.LANGUAGE_FLAG)

                            ### save ID to truck module
                            if lpid is not None:
                                self.Truck.lp_candidates.append(lpid)
                                self.Truck.hasPreciseID(lpid, "front")

                        if self.VIEWER_FLAG and front_L_detections is not None:
                            self.viewer.F_img = darknet.draw_boxes(front_L_detections, F_img, (0, 255, 0),
                                                                   front_L_roi_pts)

            if self.DEBUG_FLAG:
                fpsCheckTime = int((round((time.time() - fpsCheckStart), 3)) * 1000)  # [ms]
                fpsCheck = int(round((1000 / fpsCheckTime)))  # [fps]
                if truck_detection_mode == True:
                    time_Array.append([fpsCheckTime, fpsCheck])

            ###
            #   VIEWER
            ###
            if self.VIEWER_FLAG:
                self.viewer.draw_vasSignal(truck_detection_mode, self.Truck.truck_cnt)
                # self.viewer.draw_frameIdx(frame_num, total_frame_num, video_file_name)
                if self.DEBUG_FLAG: self.viewer.draw_fps(fpsCheckTime, fpsCheck, self.manager.avg_time,
                                                         self.manager.avg_fps, self.manager.avg_frame_length)
                self.viewer.draw_boomBarrierState(back_boomBarreir_isClose, self.CP_)
                self.viewer.draw_processRoi(truck_detection_mode, self.LP_, self.CP_, self.LP_.LP_bottom_border)
                self.viewer.show_windows(self.interface.lane)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                if key == 32:  # SPACE
                    while (True):
                        if (cv2.waitKey(0) == 32): break

        ### deallocation ###

        self.Truck.delete_info()
        # self.CamClose() changed after the issue 2022.10.02
        self.camStatus(self.CamRetList)
        # video_output_process.join()
        return False

    #     send here

    def LoadNetworks(self, net1_paths, net1_thresh, net2_paths, net2_thresh, net3_paths, net3_thresh, net4_paths,
                     net4_thresh, sealD_paths, sealD_thresh, sealR_paths, sealR_thresh):
        """ This function is for loading networks about LP and container part.

        :input param: paths for network loading, threshold for each network.

        """
        self.LP_.LoadLPNetworks(net1_paths, net1_thresh, net2_paths, net2_thresh)
        self.CP_.LoadCPNetworks(net3_paths, net3_thresh, net4_paths, net4_thresh)
        # self.SEAL_.LoadSealNetworks(sealD_paths, sealD_thresh, sealR_paths, sealR_thresh)

    def CamOpen(self, video_paths, frame_set, least_common_multiple, frame_delimiter, MODE):
        """ This function is for opening all video.

        :input param: video paths for opening the cameras, start frame

        """
        self.least_common_multiple = least_common_multiple
        self.frame_delimiter = frame_delimiter

        self.cam8 = cv2.VideoCapture(video_paths[0])
        self.cam1 = cv2.VideoCapture(video_paths[1])
        self.cam2 = cv2.VideoCapture(video_paths[2])
        self.cam3 = cv2.VideoCapture(video_paths[3])
        self.cam4 = cv2.VideoCapture(video_paths[4])
        self.cam5 = cv2.VideoCapture(video_paths[5])

        if MODE == 0:
            self.cam8.set(0, frame_set)
            self.cam1.set(0, frame_set)
            self.cam2.set(0, frame_set)
            self.cam3.set(0, frame_set)
            self.cam4.set(0, frame_set)
            self.cam5.set(0, int(frame_set * 2 / 3))

        elif MODE == 1:
            self.cam8.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            self.cam1.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            self.cam2.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            self.cam3.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            self.cam4.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            self.cam5.set(cv2.CAP_PROP_POS_FRAMES, int(frame_set * 2 / 3))

        if self.cam8.isOpened() and self.cam1.isOpened() and self.cam2.isOpened() and self.cam3.isOpened() and self.cam4.isOpened() and self.cam5.isOpened():
            return True
        else:
            return False

    def CamOpen_BCT(self, video_paths, frame_set, least_common_multiple, frame_delimiter, MODE):
        """ This function is for opening all video.

        :input param: video paths for opening the cameras, start frame

        """
        self.least_common_multiple = least_common_multiple
        self.frame_delimiter = frame_delimiter
        # print(video_paths)
        self.cam8 = cv2.VideoCapture(video_paths[0])
        self.cam1 = cv2.VideoCapture(video_paths[1])
        # self.cam2 = cv2.VideoCapture(video_paths[2])
        self.cam3 = cv2.VideoCapture(video_paths[3])
        # self.cam4 = cv2.VideoCapture(video_paths[4])
        self.cam5 = cv2.VideoCapture(video_paths[5])

        if MODE == 0:
            self.cam8.set(0, int(frame_set * 5 / 6))
            self.cam1.set(0, frame_set)
            # self.cam2.set(0, frame_set)
            self.cam3.set(0, frame_set)
            # self.cam4.set(0, frame_set)
            self.cam5.set(0, int(frame_set * 2 / 3))

        elif MODE == 1:
            self.cam8.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            self.cam1.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            # self.cam2.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            self.cam3.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            # self.cam4.set(cv2.CAP_PROP_POS_FRAMES, frame_set)
            self.cam5.set(cv2.CAP_PROP_POS_FRAMES, int(frame_set * 2 / 3))

        # if self.cam8.isOpened() and self.cam1.isOpened() and self.cam2.isOpened() and self.cam3.isOpened() and self.cam4.isOpened() and self.cam5.isOpened() :
        if self.cam8.isOpened() and self.cam1.isOpened() and self.cam3.isOpened() and self.cam5.isOpened():
            return True
        else:
            self.interface.VAS_NotiDeviceStatus_RestAPI(self.cam1.isOpened(), 'CAMR', '01')  # RightUp
            self.interface.VAS_NotiDeviceStatus_RestAPI(self.cam3.isOpened(), 'CAMR', '03')  # LeftUp
            self.interface.VAS_NotiDeviceStatus_RestAPI(self.cam5.isOpened(), 'CAMR', '05')  # TopRear
            self.interface.VAS_NotiDeviceStatus_RestAPI(self.cam8.isOpened(), 'CAMR', '08')  # LP
            return False

    def CamRead(self):
        """ This function is for reading all video image.

        frame_delimiter is used for synchronization between top&rear camera and other cameras.(top&rear video frames : 12000, other video frames : 18000
        :input param: video paths for opening the cameras, start frame
        :return: Whether or not the images are sread successfully

        """
        if self.frame_delimiter % self.least_common_multiple == 0:
            self.frame_delimiter = 0

        if self.frame_delimiter == 0 or self.frame_delimiter == 2 or self.frame_delimiter == 3 or self.frame_delimiter == 5:
            ret_cam5, self.cam5_image_Cr = self.cam5.read()
            if ret_cam5 == False:
                return False

        ret_cam8, self.cam8_image_Cr = self.cam8.read()
        if ret_cam8 == False:
            return False
        ret_cam1, self.cam1_image_Cr = self.cam1.read()
        if ret_cam1 == False:
            return False
        ret_cam2, self.cam2_image_Cr = self.cam2.read()
        if ret_cam2 == False:
            return False
        ret_cam3, self.cam3_image_Cr = self.cam3.read()
        if ret_cam3 == False:
            return False
        ret_cam4, self.cam4_image_Cr = self.cam4.read()
        if ret_cam4 == False:
            return False

        self.frame_delimiter += 1

        return True

    def CamRead_BCT(self):
        """ This function is for reading all video image.

        frame_delimiter is used for synchronization between top&rear camera and other cameras.(top&rear video frames : 12000, other video frames : 18000
        :input param: video paths for opening the cameras, start frame
        :return: Whether or not the images are sread successfully

        """
        if self.frame_delimiter % self.least_common_multiple == 0:
            self.frame_delimiter = 0

        if self.frame_delimiter == 0 or self.frame_delimiter == 2 or self.frame_delimiter == 3 or self.frame_delimiter == 5:
            self.CamRetList[2], self.cam5_image_Cr = self.cam5.read()
            if self.CamRetList[2] == False: return False

        # if self.frame_delimiter == 0 or self.frame_delimiter == 1 or self.frame_delimiter == 2 or self.frame_delimiter == 4 or self.frame_delimiter == 5 :
        #     ret_cam8, self.cam8_image_Cr = self.cam8.read()
        #     if ret_cam8 == False : return False

        # cam1 = 0, cam3 = 1, cam5 = 2, cam8 = 3
        self.CamRetList[3], self.cam8_image_Cr = self.cam8.read()
        if self.CamRetList[3] == False: return False
        self.CamRetList[0], self.cam1_image_Cr = self.cam1.read()
        if self.CamRetList[0] == False: return False

        # ret_cam1, self.cam1_image_Cr = self.cam1.read()
        # if ret_cam1 == False:
        #     return False

        # ret_cam2, self.cam2_image_Cr = self.cam2.read()
        # if ret_cam2 == False : return False
        self.CamRetList[1], self.cam3_image_Cr = self.cam3.read()
        if self.CamRetList[1] == False: return False
        # ret_cam4, self.cam4_image_Cr = self.cam4.read()
        # if ret_cam4 == False : return False

        self.frame_delimiter += 1

        return True

    def CamClose(self):
        """
        This function is for closing all video.
        """

        self.cam1.release()
        # self.cam2.release()
        self.cam3.release()
        # self.cam4.release()
        self.cam5.release()
        self.cam8.release()
        cv2.destroyAllWindows()

    def do_makeVideo_output(self, output_folder, signal_date, signal_time):
        '''
        save vas output img and videos in ftp server

        input: output forder dir, date, time info
        '''
        video_send_Start = time.time()
        # Date = str(datetime.now().year) + str('{0:02d}'.format(datetime.now().month)) + str('{0:02d}'.format(datetime.now().day))
        # Time = str('{0:02d}'.format(datetime.now().hour)) + str('{0:02d}'.format(datetime.now().minute)) + '00'

        ### Save Cropped Image and Video Output##
        self.interface.VAS_saveImage_output(self.Truck, output_folder)
        # self.interface.Vas_saveImageStitFrame(self.Truck, output_folder)
        Recognition_result_vas = self.interface.VAS_Recognition(self.Truck, signal_date, signal_time)
        # self.interface.VAS_saveVideo_output(output_folder, self.Truck, Time + "_", 33.0)
        # self.interface.VAS_saveStitVideo_output(output_folder, self.Truck, 30.0)

        print("(debug) Video Sending End & Processing Time: ", round(time.time() - video_send_Start, 2), " [s]")

    def do_makeStitch_output(self, stitch_result, output_folder, signal_date, signal_time):
        '''
        save vas stitching output image

        input: stitching result, output forder dir, date, time info
        '''
        video_send_Start = time.time()

        self.interface.VAS_saveStitchedImage_output(stitch_result, output_folder, self.Truck.is_twin_truck)
        Recognition_result_vas_stitching = self.interface.VAS_Stitching_RestAPI(self.Truck, signal_date, signal_time)

        print("(debug Stitch Video Sending End Time: ", round(time.time() - video_send_Start, 2), " [s]")

    def append_Lucam(self, frame):
        self.Truck.cam_Luset.append(frame)

    def write_frames(self, writer, frame_set):
        for idx in range(len(frame_set)):
            writer.write(frame_set[idx])

    def show_frames(self, name, mats):
        print(name + " len#: " + len(mats))
        for mat in mats:
            cv2.imshow(name, mat)
            if cv2.waitKey(0) == ord('q'): break

    def extract_C_Pos(self, detections):
        temp = []
        if detections is not None:
            for det in detections:
                temp.append(det[0])

        return temp

    def gpuStatus(self):
        hasGPU = False
        deviceID = GPUtil.getAvailable()  # [0]

        if deviceID is not None:
            hasGPU = True
        self.interface.VAS_NotiDeviceStatus_RestAPI(hasGPU, 'GPUU', '01')
        print(f'Current time is {datetime.now()}')
        # check GPU status and send to GOS(every 4min30sec)
        threading.Timer(180, self.gpuStatus).start()

    def camStatus(self, cam_ret_list):
        self.interface.VAS_NotiDeviceStatus_RestAPI(cam_ret_list[0], 'CAMR', '01')  # RightUp
        self.interface.VAS_NotiDeviceStatus_RestAPI(cam_ret_list[1], 'CAMR', '03')  # LeftUp
        self.interface.VAS_NotiDeviceStatus_RestAPI(cam_ret_list[2], 'CAMR', '05')  # TopRear
        self.interface.VAS_NotiDeviceStatus_RestAPI(cam_ret_list[3], 'CAMR', '08')  # LP


class VideoStream(object):
    def __init__(self, sources):
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads, self.status = [None] * n, [0] * n, [0] * n, [None] * n, [
            False] * n
        self.sources = sources
        self.Lu_cam_list = []
        self.Ru_cam_list = []
        self.TR_cam_list = []
        self.LP_cam_list = []
        for i, s in enumerate(sources):
            # st = f'{i + 1}/{n}: {s}... '
            cap = cv2.VideoCapture(s)
            _, self.imgs[i] = cap.read()  # guarantee first frame
            # assert cap.isOpened(), f'{st}Failed to open {s}'  
            if not cap.isOpened(): break
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            self.threads[i].start()

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]  # frame number, frame array

        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()#
            if n % 1 == 0:
                success, im = cap.retrieve()
                self.status[i] = success
                if success:
                    self.imgs[i] = im
                else:
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost

            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        imgs = self.imgs.copy()
        count = self.count
        return imgs, count

    def __len__(self):
        return len(self.sources)
