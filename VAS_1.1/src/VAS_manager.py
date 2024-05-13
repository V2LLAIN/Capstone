import cv2
import glob
import os
import json
import string
from datetime import datetime
import numpy as np

class VAS_manager():

    def __new__(self):

        if not hasattr(self, 'instance'):
            self.instance = super(VAS_manager, self).__new__(self)

        self.cols_set = [1920, 950, 894, 2592] # after H apply values will be change
        self.rows_set = [1080, 1350, 1570, 1944]
        self.fourcc = cv2.VideoWriter_fourcc(*'X264')
        self.frame_num = 0
        self.total_frame_num = 0

        self.sum_time = 0
        self.sum_fps = 0
        self.avg_time = 0
        self.avg_fps = 0
        self.avg_frame_length = 0

        self.per_truck_time_Set = []
        self.per_truck_fps_Set = []

        self.HL, self.HL_inv = self.make_HL(self)
        self.HR, self.HR_inv = self.make_HR(self)
        self.HTr = self.make_HTr(self)

        return self.instance

    def load_video_path(self, Video_Info):

        path_of_video = Video_Info[0]
        video_last_name = Video_Info[1]
        # start_frame = int(Video_Info[2])
        video = "record_2022-06-16-00.00.30.mp4"

        right_up_cam_path = path_of_video + "Camera01/" + video_last_name
        right_down_cam_path = path_of_video + "Camera02/" + video_last_name
        left_up_cam_path = path_of_video + "Camera03/" + video_last_name
        left_down_cam_path = path_of_video + "Camera04/" + video_last_name
        top_rear_cam_path = path_of_video + "Camera05/" + video_last_name
        front_cam_path = path_of_video + "Camera08/" + video_last_name
        # cam_paths = [front_cam_path, right_up_cam_path, right_down_cam_path, left_up_cam_path, left_down_cam_path, top_rear_cam_path]
        path1 = "/data/data/bct/CameraRecord(BCT)/220616_1/Camera01/" + video
        path3 = "/data/data/bct/CameraRecord(BCT)/220616_1/Camera03/" + video
        path5 = "/data/data/bct/CameraRecord(BCT)/220616_1/Camera05/" + video
        path8 = "/data/data/bct/CameraRecord(BCT)/220616_1/Camera08/" + video
        cam_paths = [path1, path3, path5, path8]
        return cam_paths


    def load_IPvideo_path(self, Video_Info):
        '''
        input ipcamera rtsp address
        retrun path of ipcamera
        '''
        username = "admin"
        password = "BCT!234$"

        left_up_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.215:554/profile3/media.smp"
        # left_down_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.216:554/profile3/media.smp"
        right_up_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.217:554/profile3/media.smp"
        # right_down_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.218:554/profile3/media.smp"
        top_rear_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.219:554/profile3/media.smp"
        front_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.220:554/profile3/media.smp"

        # #Lane2
        # left_up_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.221:554/profile3/media.smp"
        # left_down_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.222:554/profile3/media.smp"
        # right_up_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.223:554/profile3/media.smp"
        # right_down_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.224:554/profile3/media.smp"
        # top_rear_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.225:554/profile3/media.smp"
        # front_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.226:554/profile3/media.smp"

        # #Lane3
        # left_up_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.227:554/profile3/media.smp"
        # left_down_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.228:554/profile3/media.smp"
        # right_up_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.229:554/profile3/media.smp"
        # right_down_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.233:554/profile3/media.smp"
        # top_rear_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.231:554/profile3/media.smp"
        # front_cam_path = "rtsp://"+str(username)+":" + str(password) +"@192.168.205.232:554/profile3/media.smp"



        cam_paths = [right_up_cam_path, left_up_cam_path, top_rear_cam_path, front_cam_path]
        # print(cam_paths) #RTSP streaming address
        return cam_paths

    def load_resultSave_path(self, dir, Video_Info):
        video_last_name = Video_Info[1]
        json_data=open(dir, encoding='utf-8').read()
        data_template = json.loads(json_data)

        output_path = data_template['output_path'] + video_last_name + ".txt"

        return output_path
    
    def load_Net_data(self, dir, language_flag):
        json_data=open(dir, encoding='utf-8').read()
        data_template = json.loads(json_data)

        Net1_cfg = data_template['Net1_cfg']
        Net1_meta = data_template['Net1_meta']
        Net1_weight = data_template['Net1_weight']
        Net1_paths = [Net1_cfg, Net1_weight, Net1_meta]

        if language_flag == 1:
            Net2_cfg = data_template['Net2_cfg']
            Net2_meta = data_template['Net2_meta']
            Net2_weight = data_template['Net2_weight']
            Net2_paths = [Net2_cfg, Net2_weight, Net2_meta]
        elif language_flag == 2:
            Net2_cfg = data_template['Net2_eng_cfg']
            Net2_meta = data_template['Net2_eng_meta']
            Net2_weight = data_template['Net2_eng_weight']
            Net2_paths = [Net2_cfg, Net2_weight, Net2_meta]

        Net3_cfg = data_template['Net3_cfg']
        Net3_meta = data_template['Net3_meta']
        Net3_weight = data_template['Net3_weight']
        Net3_paths = [Net3_cfg, Net3_weight, Net3_meta]

        Net4_cfg = data_template['Net4_cfg']
        Net4_meta = data_template['Net4_meta']
        Net4_weight = data_template['Net4_weight']
        Net4_paths = [Net4_cfg, Net4_weight, Net4_meta]

        SealD_cfg = data_template['SealD_cfg']
        SealD_meta = data_template['SealD_meta']
        SealD_weight = data_template['SealD_weight']
        SealD_paths = [SealD_cfg, SealD_weight, SealD_meta]

        SealR_cfg = data_template['SealR_cfg']
        SealR_meta = data_template['SealR_meta']
        SealR_weight = data_template['SealR_weight']
        SealR_paths = [SealR_cfg, SealR_weight, SealR_meta]

        return Net1_paths, Net2_paths, Net3_paths, Net4_paths, SealD_paths, SealR_paths



    def load_cam8_data(self, dir):
        json_data=open(dir, encoding='utf-8').read()
        data_template = json.loads(json_data)

        start_rect_center_x = data_template['start_rect_center_x']
        start_rect_center_y = data_template['start_rect_center_y']
        start_rect_lx = data_template['start_rect_lx']
        start_rect_ly = data_template['start_rect_ly']
        LP_bottom_border = data_template['LP_bottom_border']
        start_signal_truck_dim = data_template['start_signal_truck_dim']

        return [start_rect_center_x, start_rect_center_y, start_rect_lx, start_rect_ly, LP_bottom_border, start_signal_truck_dim]


    def load_cam5_data(self, dir):
        json_data=open(dir, encoding='utf-8').read()
        data_template = json.loads(json_data)

        end_rect_center_x = data_template['end_rect_center_x']
        end_rect_center_y = data_template['end_rect_center_y']
        end_rect_lx = data_template['end_rect_lx']
        end_rect_ly = data_template['end_rect_ly']
        precede_car_from_line = data_template['precede_car_from_line']
        precede_car_to_line = data_template['precede_car_to_line']
        boom_barrier_line = data_template['boom_barrier_line']

        return [end_rect_center_x, end_rect_center_y, end_rect_lx, end_rect_ly, precede_car_from_line, precede_car_to_line, boom_barrier_line]

    def load_system_data(self, dir):
        json_data=open(dir, encoding='utf-8').read()
        data_template = json.loads(json_data)

        default_skip = data_template['default_skip']
        recog_skip = data_template['recog_skip']
        recog_skip_Tr = data_template['recog_skip_Tr']
        recog_skip_LR = data_template['recog_skip_LR']
        max_frame_thres1 = data_template['max_frame_thres1']
        max_frame_thres2 = data_template['max_frame_thres2']

        return [default_skip, recog_skip, recog_skip_Tr, recog_skip_LR, max_frame_thres1, max_frame_thres2]

    # Skip count for status check(CAMR, GPUU)
    def status_check_interval(self, dir):
        json_data=open(dir, encoding='utf-8').read()
        data_template = json.loads(json_data)

        skip_frame_count = data_template['status_check_interval']
        return skip_frame_count

    def load_system_flag(self, dir):
        json_data=open(dir, encoding='utf-8').read()
        data_template = json.loads(json_data)

        SENDING_FLAG = data_template['SENDING_FLAG']
        VIEWER_FLAG = data_template['VIEWER_FLAG']
        DEBUG_FLAG = data_template['DEBUG_FLAG']
        LANGUAGE_FLAG = data_template['LANGUAGE_FLAG']

        return [SENDING_FLAG, VIEWER_FLAG, DEBUG_FLAG, LANGUAGE_FLAG]

    def make_HL(self):
        kp1 = [
            [992, 696],
            [766, 1890],
            [723, 722],
            [608, 1882],
            [529, 712],
            [490, 1880],
            [333, 709],
            [376, 1877],
            [200, 715],
            [291, 1868],
            [22, 723],
            [177, 1848]
        ]

        kp2 = [
            [992, 696],
            [992, 1890],
            [723, 722],
            [723, 1880],
            [529, 712],
            [529, 1880],
            [333, 709],
            [333, 1880],
            [200, 715],
            [200, 1880],
            [22, 723],
            [22, 1880]
        ]
        
        kp1 = np.float32(kp1)
        kp2 = np.float32(kp2)

        H, _ = cv2.findHomography(kp1, kp2, cv2.RANSAC,4.0)
        H_inv, _ = cv2.findHomography(kp2, kp1, cv2.RANSAC,4.0)
        return H, H_inv


    def make_HR(self):
        kp1 = [
            [ 551 ,  520 ],
            [ 116 ,  446 ],
            [ 542 ,  7 ],
            [ 114 ,  108 ],
        ]
        
        kp2 = [
            [ 551 ,  500 ],
            [ 116 ,  500 ],
            [ 551 ,  27 ],
            [ 116 ,  27 ],
        ]

        for k1 in kp1:
            k1[0] = k1[0]*2
            k1[1] = k1[1]*2
        for k2 in kp2:
            k2[0] = k2[0]*2
            k2[1] = k2[1]*2

        kp1 = np.float32(kp1)
        kp2 = np.float32(kp2)

        H, _ = cv2.findHomography(kp1, kp2, cv2.RANSAC,4.0)
        H_inv, _ = cv2.findHomography(kp2, kp1, cv2.RANSAC,4.0)
        return H, H_inv

    def make_HTr(self):
        # Four corners of the book in source image (lt, ld, rt, rd)
        pts_src = np.array([[143*10/3, 190*10/3],
                            [189*10/3, 476*10/3],
                            [518*10/3, 205*10/3],
                            [463*10/3, 491*10/3]])


        # Four corners of the book in destination image. (lt, ld, rt, rd)
        w = 2592
        h = 1944
        pts_dst = np.array([[w/4, h/6*1],
                            [w/4, h/5*4],
                            [w/4*3, h/6*1],
                            [w/4*3, h/5*4]])

        # Calculate Homography
        H, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC,4.0)
        return H



    def perform_image_homography_left(self, img):
        """ This function is for converting the original image as seen from the front. 

        :input param: image.
        :return: converted image.

        """

        img_ = img.copy()
        img_ = cv2.transpose(img_)
        img_ = cv2.flip(img_, 0)
        img_ = cv2.flip(img_, 1)
        # img_ = cv2.warpPerspective(img_, self.HL, (img_.shape[1] +30, img_.shape[0] +30))
        # img_ = img_[300:1400, 100:]





        return img_

    def perform_image_homography_right(self, img):
        """ This function is for converting the original image as seen from the front. 

        :input param: image.
        :return: converted image.

        """
        img_ = img.copy()
        # img_ = cv2.warpPerspective(img_, self.HR, (img_.shape[1] +30, img_.shape[0] +30))
        img_ = cv2.transpose(img_)
        img_ = cv2.flip(img_, 1)
        
        # img_ = img_[150:1300, 100:800]#y, x

        return img_

    def print_Result(self, truck):
        print("\n\t(debug)\n\t::Result #", str(truck.truck_cnt), "cnt::")
        if truck.is_twin_truck :
            if len(truck.LicenseID) != 0:
                print(f"\t::License OCR:\t\t{truck.LicenseID[0]}\t\t{truck.LicenseID[1]} %")
            if len(truck.containerID) != 0:
                print(f"\t::Container OCR Front:\t{truck.containerID[0]}\t\t{truck.containerID[1]} %")
            if len(truck.containerID2) != 0:
                print(f"\t::Container OCR Back:\t{truck.containerID2[0]}\t\t{truck.containerID2[1]} %")
            if truck.is_40ft_truck is not None:
                if truck.is_40ft_truck: print(f"\t::Chassis Length:\t40ft")
                else: print(f"\t::Chassis Length:\t20ft")
            if truck.Truck_Chassis_pos is not None:
                print(f"\t::Chassis Position:\t{truck.Truck_Chassis_pos}")
            if truck.is_correct_DD is not None:
                print(f"\t::Door Direction:\t{str(truck.is_correct_DD)}")
            else:
                print(f"\t::Door Direction:\tNone")
        else:
            if len(truck.LicenseID) != 0:
                print(f"\t::License OCR:\t\t{truck.LicenseID[0]}\t\t{truck.LicenseID[1]} %")
            if len(truck.containerID) != 0:
                print(f"\t::Container OCR:\t{truck.containerID[0]}\t\t{truck.containerID[1]} %")
            if truck.is_40ft_truck is not None:
                if truck.is_40ft_truck: print(f"\t::Chassis Length:\t40ft")
                else: print(f"\t::Chassis Length:\t20ft")
            if truck.Truck_Chassis_pos is not None:
                print(f"\t::Chassis Position:\t{truck.Truck_Chassis_pos}")
            if truck.is_correct_DD is not None:
                print(f"\t::Door Direction:\t{str(truck.is_correct_DD)}")
            else:
                print(f"\t::Door Direction:\tNone")
        print("\n")
        
    def jpg_Count(self, dir):
        images = glob.glob(dir + '/*.jpg')
        return len(images)

    def txt_Count(self, dir):
        txts = glob.glob(dir + '/*.txt')
        return len(txts)

    def return_Image_jpg(self, dir):
        images = glob.glob(dir + '/*.jpg')
        return images

    def get_timestamp(self):
        # hh_ddmmyy= str(datetime.today().hour).zfill(2) + "_" + str(datetime.today().day).zfill(2) + str(datetime.today().month).zfill(2) + str(str(datetime.today().year)[2:]).zfill(2)
        hh_ddmmyy= str(datetime.today().day).zfill(2) + str(datetime.today().month).zfill(2) + str(str(datetime.today().year)[2:]).zfill(2)
        return hh_ddmmyy

    def set_frameInfo(self, cap):
        self.frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frameInfo(self):
        return self.frame_num, self.total_frame_num

    def alarm_video_pos(self, frame_num, total_frame_num):
        video_position = (frame_num / total_frame_num * 100)
        if video_position % 10 == 0 :
            print("(debug)Video frame position: " + str(video_position) + " %")

    def calc_avgTime(self, time_Array):
        self.avg_time = 0
        self.avg_fps = 0
        self.avg_frame_length = 0
        
        self.sum_time = 0
        self.sum_fps = 0

        for idx in range(len(time_Array)):
            self.sum_time = self.sum_time + time_Array[idx][0]
            self.sum_fps = self.sum_fps + time_Array[idx][1]

        self.avg_frame_length = len(time_Array)
        self.avg_time = int(round(self.sum_time/self.avg_frame_length))
        self.avg_fps = int(round(self.sum_fps/self.avg_frame_length))
    def BCT_frame_convertor(self, stitcher, mMap_x, mMap_y, F_img, Ru_img, Lu_img, Tr_img):

        # Lu_img = self.perform_image_homography_left(Lu_img)#LU 
        Lu_img = cv2.remap(Lu_img, mMap_x, mMap_y, cv2.INTER_LINEAR)#LU - undistort
        Lu_img = cv2.transpose(Lu_img)
        Lu_img = cv2.flip(Lu_img, 0)


        # imgl_h, imgl_w = Lu_img_2.shape[:2]
        # Left_H = stitcher.make_Right_view_H(imgl_h, imgl_w)
        # Lu_img_2 = cv2.warpPerspective(Lu_img_2, Left_H, (imgl_w, imgl_h))



        # Ru_img = self.perform_image_homography_right(Ru_img)#RU
        Ru_img = cv2.remap(Ru_img, mMap_x, mMap_y, cv2.INTER_LINEAR)#RU - undistort
        Ru_img = cv2.transpose(Ru_img)
        Ru_img = cv2.flip(Ru_img, 1)


        # imgr_h, imgr_w = Ru_img_2.shape[:2]
        # Right_H = stitcher.make_Right_view_H(imgr_h, imgr_w)
        # Ru_img_2 = cv2.warpPerspective(Ru_img_2, Right_H, (imgr_w, imgr_h))

        return F_img, Ru_img, Lu_img, Tr_img

    # def BCT_frame_convertor(self, stitcher, mMap_x, mMap_y, F_img, Ru_img, Lu_img, Tr_img):
    #     Lu_img_2 = Lu_img.copy() # Camera03
    #     Ru_img_2 = Ru_img.copy() # Camera01


    #     # Lu_img = self.perform_image_homography_left(Lu_img)#LU 
    #     Lu_img_2 = cv2.remap(Lu_img_2, mMap_x, mMap_y, cv2.INTER_LINEAR)#LU - undistort
    #     Lu_img_2 = cv2.transpose(Lu_img_2)
    #     Lu_img_2 = cv2.flip(Lu_img_2, 0)
    #     Lu_img_2 = cv2.flip(Lu_img_2, 1)
    #     Lu_img = Lu_img_2

    #     # imgl_h, imgl_w = Lu_img_2.shape[:2]
    #     # Left_H = stitcher.make_Right_view_H(imgl_h, imgl_w)
    #     # Lu_img_2 = cv2.warpPerspective(Lu_img_2, Left_H, (imgl_w, imgl_h))



    #     # Ru_img = self.perform_image_homography_right(Ru_img)#RU
    #     Ru_img_2 = cv2.remap(Ru_img_2, mMap_x, mMap_y, cv2.INTER_LINEAR)#RU - undistort
    #     Ru_img_2 = cv2.transpose(Ru_img_2)
    #     Ru_img_2 = cv2.flip(Ru_img_2, 1)
    #     Ru_img = Ru_img_2


    #     # imgr_h, imgr_w = Ru_img_2.shape[:2]
    #     # Right_H = stitcher.make_Right_view_H(imgr_h, imgr_w)
    #     # Ru_img_2 = cv2.warpPerspective(Ru_img_2, Right_H, (imgr_w, imgr_h))

    #     return F_img, Ru_img, Ru_img_2, Lu_img, Lu_img_2, Tr_img



    def frame_convertor(self, F_img, Ru_img, Rd_img, Lu_img, Ld_img, Tr_img):
        Lu_img = self.perform_image_homography_left(Lu_img)
        # Lu_img = cv2.rotate(Lu_img, cv2.ROTATE_90_CLOCKWISE)
        Ld_img = cv2.rotate(Ld_img, cv2.ROTATE_90_CLOCKWISE)

        Ru_img = self.perform_image_homography_right(Ru_img)
        # Ru_img = cv2.rotate(Ru_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        Rd_img = cv2.rotate(Rd_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return F_img, Ru_img, Rd_img, Lu_img, Ld_img, Tr_img

    def frame_undistort_convertor(self, mMap_x, mMap_y, right_up_img, right_down_img, left_up_img, left_down_img):
        left_up_img = cv2.remap(left_up_img, mMap_x, mMap_y, cv2.INTER_LINEAR)
        left_down_img = cv2.remap(left_down_img, mMap_x, mMap_y, cv2.INTER_LINEAR)



        # height, width, channel = right_up_img.shape
        # matrix_for_up = cv2.getRotationMatrix2D((width/2, height/2), -3, 1)
        # matrix_for_down = cv2.getRotationMatrix2D((width/2, height/2), -1, 1)
        # right_up_img = cv2.warpAffine(right_up_img, matrix_for_up, (width, height))
        # right_down_img = cv2.warpAffine(right_down_img, matrix_for_down, (width, height))

        right_up_img = cv2.remap(right_up_img, mMap_x, mMap_y, cv2.INTER_LINEAR)
        right_down_img = cv2.remap(right_down_img, mMap_x, mMap_y, cv2.INTER_LINEAR)

        # -debug
        if False:
            if False:
                left_up_img_resize = cv2.resize(left_up_img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                left_down_img_resize = cv2.resize(left_down_img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("left_up_img_resize", left_up_img_resize)
                cv2.imshow("left_down_img_resize", left_down_img_resize)
            if False:
                right_up_img_resize = cv2.resize(right_up_img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                right_down_img_resize = cv2.resize(right_down_img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("right_up_img_resize", right_up_img_resize)
                cv2.imshow("right_down_img_resize", right_down_img_resize)
                cv2.waitKey(0)


        return right_up_img, right_down_img, left_up_img, left_down_img


    