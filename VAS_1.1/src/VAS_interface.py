import cv2
import json
import requests
import ftplib
import os
import copy
import time
import asyncio
import base64
from difflib import get_close_matches
from datetime import datetime
import numpy as np


class VAS_interface:

    def __new__(self, config_dir):
        if not hasattr(self, 'instance'):
            self.instance = super(VAS_interface, self).__new__(self)

        # self.fourcc = cv2.VideoWriter_fourcc('a','v','c','1') # have to change due to LC
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # to here

        self.TruckArrivedObjectItem_url, self.AIToGosSendProperty_url, self.ImageStitched_url, self.DeviceStatus_url, self.lane_type, self.lane, self.FTP_server, self.FTP_user, self.FTP_password, self.FTP_port, self.fileserver_dir = self.load_config_json(
            self, config_dir)
        self.filserver_url = str(self.FTP_server) + ":" + str(self.FTP_port)
        # print('URL TEST: ',self.TruckArrivedObjectItem_url, self.AIToGosSendProperty_url, self.ImageStitched_url, self.DeviceStatus_url)
        self.authorizationID = "AI01"
        self.authorizationPW = "aaa$"
        self.authorization = self.authorizationID + ':' + self.authorizationPW
        self.authorization_bytes = self.authorization.encode('UTF-8')
        self.authorization_base64 = base64.b64encode(self.authorization_bytes)
        self.authorization_str = str('Basic ') + self.authorization_base64.decode('UTF-8')

        self.p_headers = {'Content-Type': 'application/json; charset=utf-8', 'Authorization': self.authorization_str}
        # self.test_url = "http://192.168.203.63:5001/AI"    #test rest for MIN's PC

        self.template_TruckArrivedObjectItem = self.set_template_TruckArrivedObjectItem(self)
        self.template_RecognitionObjectItem = self.set_template_RecognitionObjectItem(self)
        self.template_BoomBarrierCloseObjectItem = self.set_template_BoomBarrierCloseObjectItem(self)
        self.template_StitchingObjectItem = self.set_template_StitchingObjectItem(self)
        self.template_NotiDeviceStatusItem = self.set_template_NotiDeviceStatusItem(self)

        self.local_path = self.folder_empty(self, self.fileserver_dir, "VAS",
                                            self.lane)  # local_path:  /home/kkk/fileserver/VAS/2021/03/17/L1
        self.remote_path = self.filserver_url + self.local_path.replace(self.fileserver_dir,
                                                                        "")  # remote_path:  http://192.168.6.62:8080/VAS/2021/03/17/Lane-01
        print("LOCAL PATH: ", self.local_path)
        print("REMOTE PATH: ", self.remote_path)

        self.boom_barrier_STATE = False
        return self.instance

    def VAS_Truckarrived(self):

        truckArrived = copy.deepcopy(self.template_TruckArrivedObjectItem)

        Date = str(datetime.now().year) + str('{0:02d}'.format(datetime.now().month)) + str(
            '{0:02d}'.format(datetime.now().day))
        # Time = str('{0:02d}'.format(datetime.now().hour)) + str('{0:02d}'.format(datetime.now().minute)) + '00'
        Time = str('{0:02d}'.format(datetime.now().hour)) + str('{0:02d}'.format(datetime.now().minute)) + str(
            '{0:02d}'.format(datetime.now().second))

        ### Date ###
        truckArrived['Date'] = Date
        truckArrived['Time'] = Time
        truckArrived['LaneType'] = self.lane_type  # In-Lane
        truckArrived['LaneID'] = self.lane  # L1

        for n in range(1):
            try:
                truckArrived_res = requests.post(self.TruckArrivedObjectItem_url, headers=self.p_headers,
                                                 data=json.dumps(truckArrived))  # TruckArrivedObjectItem_url
                print("- Truck Arrived: ", truckArrived_res, "\n", truckArrived_res.content)
                print("- TruckArrived result: ", truckArrived, "\n\n")
                if truckArrived_res.status_code == 200: break

            except requests.exceptions.Timeout as errd:
                print("Timeout Error : ", errd)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting : ", errc)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.HTTPError as errb:
                print("Http Error : ", errb)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            # Any Error except upper exception
            except requests.exceptions.RequestException as erra:
                print("AnyException : ", erra)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

    def VAS_BoomBarrierClose(self):

        BoomBarrierClose = copy.deepcopy(self.template_BoomBarrierCloseObjectItem)

        ### Date ###
        BoomBarrierClose['Date'] = str(datetime.now().year) + str('{0:02d}'.format(datetime.now().month)) + str(
            '{0:02d}'.format(datetime.now().day))
        # BoomBarrierClose['Time'] = str('{0:02d}'.format(datetime.now().hour)) + str('{0:02d}'.format(datetime.now().minute)) + '00'
        BoomBarrierClose['Time'] = str('{0:02d}'.format(datetime.now().hour)) + str(
            '{0:02d}'.format(datetime.now().minute)) + str('{0:02d}'.format(datetime.now().second))
        BoomBarrierClose['LaneType'] = self.lane_type  # In-Lane
        BoomBarrierClose['LaneID'] = self.lane  # L1

        for n in range(1):
            try:
                BoomBarrierClose_res = requests.post(self.TruckArrivedObjectItem_url, headers=self.p_headers,
                                                     data=json.dumps(BoomBarrierClose))  # TruckArrivedObjectItem_url
                print("- BoomBarrier Closed: ", BoomBarrierClose_res, "\n", BoomBarrierClose_res.content)
                print("- BoomBarrier Closed result: ", BoomBarrierClose, "\n\n")
                if BoomBarrierClose_res.status_code == 200: break

            except requests.exceptions.Timeout as errd:
                print("Timeout Error : ", errd)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting : ", errc)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.HTTPError as errb:
                print("Http Error : ", errb)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            # Any Error except upper exception
            except requests.exceptions.RequestException as erra:
                print("AnyException : ", erra)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

    def VAS_Stitching_RestAPI(self, truck, date, time):

        ### Date ###
        # Date = str(datetime.now().year) + str('{0:02d}'.format(datetime.now().month)) + str('{0:02d}'.format(datetime.now().day))
        # Time = str('{0:02d}'.format(datetime.now().hour)) + str('{0:02d}'.format(datetime.now().minute)) + '00'
        # Time = str('{0:02d}'.format(datetime.now().hour)) + str('{0:02d}'.format(datetime.now().minute)) + str('{0:02d}'.format(datetime.now().second))
        Date = date
        Time = time
        StitchedItem = copy.deepcopy(self.template_StitchingObjectItem)
        Recognition_result_vas = self.saveAsJSON_VAS_stitching(StitchedItem, truck, Date, Time)

        for n in range(1):
            try:
                StitchedItem_res = requests.post(self.ImageStitched_url, headers=self.p_headers,
                                                 data=json.dumps(StitchedItem))  # ImageStitched_url
                print("- Image Stitched: ", StitchedItem_res, "\n", StitchedItem_res.content)
                print("- Image Stitched result: ", StitchedItem, "\n\n")
                if StitchedItem_res.status_code == 200: break

            except requests.exceptions.Timeout as errd:
                print("Timeout Error : ", errd)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting : ", errc)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.HTTPError as errb:
                print("Http Error : ", errb)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            # Any Error except upper exception
            except requests.exceptions.RequestException as erra:
                print("AnyException : ", erra)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

    def VAS_NotiDeviceStatus_RestAPI(self, deviceStatus, deviceType, deviceID):
        ### CAMREA ERROR CATCH                           ### GPU ERROR CATCH
        # {'LaneID': 'In-Lane01',                        # {'LaneID': 'In-Lane01',
        # 'MessageID': 'NotiDeviceStatus',               # 'MessageID': 'NotiDeviceStatus',
        # 'Body': {'DeviceType': 'CAMR',                 # 'Body': {'DeviceType': 'GPUU',
        #          'DeviceKey': 'CAMR0104',              #          'DeviceKey': 'GPUU0101',
        #          'DeviceStatus': 'I' or 'E'            #          'DeviceStatus': 'I' or 'E',
        #          'DeviceErrorCode': '' or 'CAMR404'}}  #          'DeviceErrorCode': '' or 'GPUU404'}}

        DeviceItem = copy.deepcopy(self.template_NotiDeviceStatusItem)
        DeviceItem['LaneID'] = self.lane  # L1

        if deviceStatus and deviceType == 'CAMR':
            DeviceItem['Body']['DeviceType'] = str(deviceType)
            DeviceItem['Body']['DeviceKey'] = str(DeviceItem['Body']['DeviceType']) + '0' + str(
                DeviceItem['LaneID'][1]) + str(deviceID)
            DeviceItem['Body']['DeviceStatus'] = 'I'
            DeviceItem['Body']['DeviceErrorCode'] = ''

        elif deviceStatus == False and deviceType == 'CAMR':
            DeviceItem['Body']['DeviceType'] = str(deviceType)
            DeviceItem['Body']['DeviceKey'] = str(DeviceItem['Body']['DeviceType']) + '0' + str(
                DeviceItem['LaneID'][1]) + str(deviceID)
            DeviceItem['Body']['DeviceStatus'] = 'E'
            DeviceItem['Body']['DeviceErrorCode'] = 'CAMR404'

        if deviceStatus and deviceType == 'GPUU':

            DeviceItem['Body']['DeviceType'] = str(deviceType)
            DeviceItem['Body']['DeviceKey'] = str(DeviceItem['Body']['DeviceType']) + '0' + str(
                DeviceItem['LaneID'][1]) + str(deviceID)
            DeviceItem['Body']['DeviceStatus'] = 'I'
            DeviceItem['Body']['DeviceErrorCode'] = ''

        elif deviceStatus and deviceType == 'GPUU':

            DeviceItem['Body']['DeviceType'] = str(deviceType)
            DeviceItem['Body']['DeviceKey'] = str(DeviceItem['Body']['DeviceType']) + '0' + str(
                DeviceItem['LaneID'][1]) + str(deviceID)
            DeviceItem['Body']['DeviceStatus'] = 'E'
            DeviceItem['Body']['DeviceErrorCode'] = 'GPUU404'

        for n in range(1):
            try:
                # print('self.authorizationID_str: ',self.authorizationID_str, 'self.authorizationPW_str: ',self.authorizationPW_str)

                DeviceItem_res = requests.post(self.DeviceStatus_url, headers=self.p_headers,
                                               data=json.dumps(DeviceItem))  # DeviceStatus_url

                print("- Device status check: ", DeviceItem_res, "\n", DeviceItem_res.content)
                # print("- DeviceStatus_url: ", self.DeviceStatus_url)
                print("- DeviceStatus result: ", DeviceItem, "\n\n")
                if DeviceItem_res.status_code == 200: break

            except requests.exceptions.Timeout as errd:
                print("Timeout Error : ", errd)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting : ", errc)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.HTTPError as errb:
                print("Http Error : ", errb)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            # Any Error except upper exception
            except requests.exceptions.RequestException as erra:
                print("AnyException : ", erra)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

    def make_recognition_output_folder(self, truck):
        '''
        make output forder in ftp server dir

        input truck class
        return folder dir, date&time information
        '''
        Date = str(datetime.now().year) + str('{0:02d}'.format(datetime.now().month)) + str(
            '{0:02d}'.format(datetime.now().day))
        # Time = str('{0:02d}'.format(datetime.now().hour)) + str('{0:02d}'.format(datetime.now().minute)) + '00'
        Time = str('{0:02d}'.format(datetime.now().hour)) + str('{0:02d}'.format(datetime.now().minute)) + str(
            '{0:02d}'.format(datetime.now().second))
        ### make JSON for detect and save car info using VAS ###
        folder_car = self.local_path + "/" + Time + "_" + truck.LicenseID[0] + "/"
        createFolder(folder_car)

        return folder_car, Date, Time

    ### Recognition VAS ###
    def VAS_Recognition(self, truck, date, time):
        # data input
        # Date = str(datetime.now().year) + str('{0:02d}'.format(datetime.now().month)) + str('{0:02d}'.format(datetime.now().day))
        # Time = str('{0:02d}'.format(datetime.now().hour)) + str('{0:02d}'.format(datetime.now().minute)) + '00'
        Date = date
        Time = time
        RecognitionObjectItem = copy.deepcopy(self.template_RecognitionObjectItem)
        Recognition_result_vas = self.saveAsJSON_VAS(RecognitionObjectItem, truck, Date, Time)

        ### Send the REST POST
        for n in range(1):
            try:
                RecognitionObjectItem_res = requests.post(self.AIToGosSendProperty_url, headers=self.p_headers,
                                                          data=json.dumps(
                                                              Recognition_result_vas))  # AIToGosSendProperty_url
                print("- Recognition End: ", RecognitionObjectItem_res, "\n", RecognitionObjectItem_res.content)
                # print("- AIToGosSendProperty_url: ", self.AIToGosSendProperty_url, "\n\n")
                print('- Recognition End result: ', Recognition_result_vas, "\n\n")
                if RecognitionObjectItem_res.status_code == 200: break

            except requests.exceptions.Timeout as errd:
                print("Timeout Error : ", errd)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting : ", errc)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            except requests.exceptions.HTTPError as errb:
                print("Http Error : ", errb)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

            # Any Error except upper exception
            except requests.exceptions.RequestException as erra:
                print("AnyException : ", erra)
                time.sleep(1)
                # await asyncio.sleep(1)
                continue

        return Recognition_result_vas

    def Vas_saveImageStitFrame(self, truck, folder_car):
        createFolder(os.path.join(folder_car, "stitching_test"))
        try:
            if truck.cam_Luset[truck.det_Luset[0]] is not None:
                cv2.imwrite(os.path.join(folder_car, "stitching_test") + "/leftU_first_frame.jpg",
                            truck.cam_Luset[truck.det_Luset[0]])
            if truck.cam_Luset[truck.det_Luset[2]] is not None:
                cv2.imwrite(os.path.join(folder_car, "stitching_test") + "/leftU_last_frame.jpg",
                            truck.cam_Luset[truck.det_Luset[2]])
            if truck.cam_Ruset[truck.det_Ruset[0]] is not None:
                cv2.imwrite(os.path.join(folder_car, "stitching_test") + "/rightU_first_frame.jpg",
                            truck.cam_Ruset[truck.det_Ruset[0]])
            if truck.cam_Ruset[truck.det_Ruset[2]] is not None:
                cv2.imwrite(os.path.join(folder_car, "stitching_test") + "/rightU_last_frame.jpg",
                            truck.cam_Ruset[truck.det_Ruset[2]])
            if truck.cam_TRset[truck.det_TRset[0]] is not None:
                cv2.imwrite(os.path.join(folder_car, "stitching_test") + "/TR_first_frame.jpg",
                            truck.cam_TRset[truck.det_TRset[0]])
            if truck.cam_TRset[truck.det_TRset[2]] is not None:
                cv2.imwrite(os.path.join(folder_car, "stitching_test") + "/TR_last_frame.jpg",
                            truck.cam_TRset[truck.det_TRset[2]])
            if truck.is_twin_truck:
                if truck.cam_Ruset[int(np.median(truck.det_Luset[1]))] is not None:
                    cv2.imwrite(os.path.join(folder_car, "stitching_test") + "/leftU_center_frame.jpg",
                                truck.cam_Ruset[int(np.median(truck.det_Luset[1]))])
                if truck.cam_Ruset[int(np.median(truck.det_Ruset[1]))] is not None:
                    cv2.imwrite(os.path.join(folder_car, "stitching_test") + "/rightU_center_frame.jpg",
                                truck.cam_Ruset[int(np.median(truck.det_Ruset[1]))])
                if truck.cam_Ruset[int(np.median(truck.det_TRset[1]))] is not None:
                    cv2.imwrite(os.path.join(folder_car, "stitching_test") + "/TR_center_frame.jpg",
                                truck.cam_Ruset[int(np.median(truck.det_TRset[1]))])
        except IndexError as err:
            print("Index error: ", err)

    def VAS_saveStitVideo_output(self, folder_dir, Truck, fps):
        # out8 = cv2.VideoWriter(folder_dir + "/" + timestamp + "cam8_" + ".mp4", self.fourcc, fps,
        #                        (int(self.cols_set[0]), int(self.rows_set[0])))
        out1 = cv2.VideoWriter(os.path.join(folder_dir, "stitching_test") + "/video_rightU" + ".mp4", self.fourcc, fps,
                               (int(self.cols_set[1]), int(self.rows_set[1])))
        out3 = cv2.VideoWriter(os.path.join(folder_dir, "stitching_test") + "/video_leftU" + ".mp4", self.fourcc, fps,
                               (int(self.cols_set[2]), int(self.rows_set[2])))
        out5 = cv2.VideoWriter(os.path.join(folder_dir, "stitching_test") + "/video_TR" + ".mp4", self.fourcc, 20,
                               (int(self.cols_set[3]), int(self.rows_set[3])))
        # write_frames(out8, Truck.cam8_video)
        write_frames(out1, Truck.cam_Ruset)
        write_frames(out3, Truck.cam_Luset)
        write_frames(out5, Truck.cam_TRset)
        # out8.release()
        out1.release()
        out3.release()
        out5.release()

    def VAS_saveImage_output(self, truck, folder_car):
        if truck.is_twin_truck:
            if truck.front_img is not None:
                cv2.imwrite(folder_car + "/02_20ftTwin_LP.jpg", truck.front_img)

            if truck.top_img is not None:
                cv2.imwrite(folder_car + "/04_20ftFore_Top.jpg", truck.top_img)
            if truck.left_img is not None:
                cv2.imwrite(folder_car + "/05_20ftFore_Left.jpg", truck.left_img)
            if truck.right_img is not None:
                cv2.imwrite(folder_car + "/06_20ftFore_Right.jpg", truck.right_img)

            if truck.rear_img is not None:
                cv2.imwrite(folder_car + "/07_20ftAfter_Rear.jpg", truck.rear_img)
            if truck.top_img2 is not None:
                cv2.imwrite(folder_car + "/08_20ftAfter_Top.jpg", truck.top_img2)
            if truck.left_img2 is not None:
                cv2.imwrite(folder_car + "/09_20ftAfter_Left.jpg", truck.left_img2)
            if truck.right_img2 is not None:
                cv2.imwrite(folder_car + "/10_20ftAfter_Right.jpg", truck.right_img2)

        else:
            if truck.front_img is not None:
                cv2.imwrite(folder_car + "/02_20ftTwin_LP.jpg", truck.front_img)
            if truck.rear_img is not None:
                cv2.imwrite(folder_car + "/03_20ftAfter_Rear.jpg", truck.rear_img)
            if truck.top_img is not None:
                cv2.imwrite(folder_car + "/04_20ftFore_Top.jpg", truck.top_img)
            if truck.left_img is not None:
                cv2.imwrite(folder_car + "/05_20ftFore_Left.jpg", truck.left_img)
            if truck.right_img is not None:
                cv2.imwrite(folder_car + "/06_20ftFore_Right.jpg", truck.right_img)

    def VAS_saveStitchedImage_output(self, stitched_set, folder_car, is_twin_truck):
        if is_twin_truck:
            if stitched_set[1] is not None and 0 not in stitched_set[1].shape:  # twin front top
                cv2.imwrite(folder_car + "/14_Fore_Right.jpg", stitched_set[1])
            if stitched_set[2] is not None and 0 not in stitched_set[2].shape:  # twin front right
                cv2.imwrite(folder_car + "/13_Fore_Left.jpg", stitched_set[2])
            if stitched_set[3] is not None and 0 not in stitched_set[3].shape:  # twin front left
                cv2.imwrite(folder_car + "/12_Fore_Top.jpg", stitched_set[3])

            if stitched_set[4] is not None and 0 not in stitched_set[4].shape:  # twin back top
                cv2.imwrite(folder_car + "/18_After_Right.jpg", stitched_set[4])
            if stitched_set[5] is not None and 0 not in stitched_set[5].shape:  # twin back right
                cv2.imwrite(folder_car + "/17_After_Left.jpg", stitched_set[5])
            if stitched_set[6] is not None and 0 not in stitched_set[6].shape:  # twin back left
                cv2.imwrite(folder_car + "/16_After_Top.jpg", stitched_set[6])

            if stitched_set[0] is not None and 0 not in stitched_set[0].shape:  # rear
                cv2.imwrite(folder_car + "/15_After_Rear.jpg", stitched_set[0])

        else:
            if stitched_set[1] is not None and 0 not in stitched_set[1].shape:  # twin front top
                cv2.imwrite(folder_car + "/14_Fore_Right.jpg", stitched_set[1])
            if stitched_set[2] is not None and 0 not in stitched_set[2].shape:  # twin front right
                cv2.imwrite(folder_car + "/13_Fore_Left.jpg", stitched_set[2])
            if stitched_set[3] is not None and 0 not in stitched_set[3].shape:  # twin front left
                cv2.imwrite(folder_car + "/12_Fore_Top.jpg", stitched_set[3])

            if stitched_set[0] is not None and 0 not in stitched_set[0].shape:  # rear
                cv2.imwrite(folder_car + "/11_Fore_Rear.jpg", stitched_set[0])

    def set_image_Size(self, F_img, R_img, L_img, Tr_img):
        self.cols_set = [F_img.shape[1], R_img.shape[1], L_img.shape[1], Tr_img.shape[1]]
        self.rows_set = [F_img.shape[0], R_img.shape[0], L_img.shape[0], Tr_img.shape[0]]

    def VAS_saveVideo_output(self, folder_dir, Truck, timestamp, fps):
        out8 = cv2.VideoWriter(folder_dir + "/" + timestamp + "cam8_" + ".mp4", self.fourcc, fps,
                               (int(self.cols_set[0]), int(self.rows_set[0])))
        out1 = cv2.VideoWriter(folder_dir + "/" + timestamp + "cam1_" + ".mp4", self.fourcc, fps,
                               (int(self.cols_set[1]), int(self.rows_set[1])))
        out3 = cv2.VideoWriter(folder_dir + "/" + timestamp + "cam3_" + ".mp4", self.fourcc, fps,
                               (int(self.cols_set[2]), int(self.rows_set[2])))
        out5 = cv2.VideoWriter(folder_dir + "/" + timestamp + "cam5_" + ".mp4", self.fourcc, fps,
                               (int(self.cols_set[3]), int(self.rows_set[3])))
        write_frames(out8, Truck.cam8_video)
        write_frames(out1, Truck.cam1_video)
        write_frames(out3, Truck.cam3_video)
        write_frames(out5, Truck.cam5_video)
        out8.release()
        out1.release()
        out3.release()
        out5.release()



    def print_Result(self, truck):
        print("\n\t(debug)\n\t::Result #", str(truck.truck_cnt), "cnt::")
        if truck.is_twin_truck:
            if len(truck.LicenseID) != 0:
                print(f"\t::License OCR:\t\t{truck.LicenseID[0]}\t\t{truck.LicenseID[1]} %")
            if len(truck.containerID) != 0:
                print(f"\t::Container OCR Front:\t{truck.containerID[0]}\t\t{truck.containerID[1]} %")
            if len(truck.containerID2) != 0:
                print(f"\t::Container OCR Back:\t{truck.containerID2[0]}\t\t{truck.containerID2[1]} %")
            if truck.is_40ft_truck is not None:
                if truck.is_40ft_truck:
                    print(f"\t::Chassis Length:\t40ft")
                else:
                    print(f"\t::Chassis Length:\t20ft")
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
                if truck.is_40ft_truck:
                    print(f"\t::Chassis Length:\t40ft")
                else:
                    print(f"\t::Chassis Length:\t20ft")
            if truck.Truck_Chassis_pos is not None:
                print(f"\t::Chassis Position:\t{truck.Truck_Chassis_pos}")
            if truck.is_correct_DD is not None:
                print(f"\t::Door Direction:\t{str(truck.is_correct_DD)}")
            else:
                print(f"\t::Door Direction:\tNone")
        print("\n")

    def correctLP(self, lp_num):
        if "MV" in lp_num: lp_num = lp_num[2:]
        if len(lp_num) >= 10: lp_num = lp_num[-6:]
        if "WIE" in lp_num: lp_num = lp_num[:6]
        if lp_num == "OBR46900": lp_num = "CBR4690"
        if lp_num == "OXV511": lp_num = "DXY511"
        if lp_num == "CA1969": lp_num = "CAH1969"

        return lp_num

    def correctCP(self, cp_num):
        if "ERSU" in cp_num: cp_num = cp_num.replace("ERSU", "EISU")
        if "TNKU" in cp_num: cp_num = cp_num.replace("TNKU", "INKU")
        if "MVCU" in cp_num: cp_num = cp_num.replace("MVCU", "MWCU")
        if "MACU" in cp_num: cp_num = cp_num.replace("MACU", "MWCU")
        if "ETTU" in cp_num: cp_num = cp_num.replace("ETTU", "EITU")
        if "FOLU" in cp_num: cp_num = cp_num.replace("FOLU", "EOLU")

        return cp_num


    def lpNumReconfirm(self, lp_num):
        matching_case = 1
        threshold = 0.8
        lp_candidate = ['NEX5786', 'NEG9662', 'NDP4827', 'NDG6810', 'DKY511', 'DAP7408', 'CAH1969', 'CBX7305','184369', '038003AQ', '1119438', '044808', '035205', '030109', '120110', '040403', '120192',
                        '130100001308480', '138000000809668', '180100000135056', '809327', '097536', '044809', '044801',
                        '030162232', '047202', '1301880409',
                        'NDH2031', 'CDJ7731', 'CTP301', 'NDF8293', 'NAN5778', 'CBJ4296', 'NEO2399',
                        'CAE9447', 'CDJ2649', 'MAP6455', 'NGA9581', 'CAG4051', 'AAU3950', 'AAB8783', 'AAB1075',
                        'AAB2769', 'AAB2770', 'AAB4118', 'AAB7276', 'AAC6773', 'AAC1446', 'AAE9155', 'AAH6629',
                        'AAH6635', 'AAH6636', 'AAH6681', 'AAI1189', 'AAI1191', 'AAI1193', 'AAI1197', 'AAI1207',
                        'AAI1326', 'AAI1329', 'AAI2133', 'AAI2135', 'AAI2136', 'AAI9869', 'AAJ1898', 'AAJ1899',
                        'AAJ1280', 'AAJ1370', 'AAJ1374', 'AAJ1375', 'AAJ1376', 'AAJ1534', 'AAJ1535', 'AAJ1536',
                        'AAJ1537', 'AAJ1958', 'AAJ2136', 'AAJ2148', 'AAJ2152', 'AAJ2153', 'AAJ2154', 'AAJ2155',
                        'AAJ2311', 'AAJ2312', 'AAJ2314', 'AAJ2364', 'AAJ2641', 'AAJ2642', 'AAJ2797', 'AAN1220',
                        'AAN1224', 'AAN1225', 'AAN1226', 'AAN2540', 'AAN3114', 'AAN3121', 'AAN3132', 'AAO1081',
                        'AAO1087', 'AAO1088', 'AAO1089', 'AAO1090', 'AAO1091', 'AAP1568', 'AAP1569', 'AAP1574',
                        'AAQ6588', 'AAQ6590', 'AAQ7326', 'AAQ7732', 'AAQ8031', 'AAQ8070', 'AAQ8451', 'AAQ8643',
                        'AAQ8769', 'AAQ8836', 'AAQ8974', 'AAQ9227', 'AAQ9271', 'AAQ9353', 'AAQ9635', 'AAQ9637',
                        'AAQ9644', 'AAR1130', 'AAR1269', 'AAR1314', 'AAR1319', 'AAR1514', 'AAR1588',
                        'AAR1610', 'AAR1742', 'AAR1761', 'AAR1762', 'AAR2153', 'AAR2346', 'AAR2404', 'AAR2432',
                        'AAR2825', 'AAR2860', 'AAR2923', 'AAR2938', 'AAR2998', 'AAS3621', 'AAT3684', 'AAT1714',
                        'AAT2312', 'AAT2684', 'AAT2737', 'AAT2948', 'AAT3054', 'AAT3174', 'AAT3176', 'AAT3450',
                        'AAT3603', 'AAT3634', 'AAT3637', 'AAT3761', 'AAT3780', 'AAT3960', 'AAT4118', 'AAT4183',
                        'AAT4188', 'AAT4289', 'AAT4364', 'AAT4531', 'AAT4532', 'AAT4724', 'AAT4792', 'AAU7376',
                        'AAU7377', 'AAU7379', 'AAU9512', 'AAV5049', 'AAV6781', 'AAV1181', 'AAV1772', 'AAV3332',
                        'AAV3357', 'AAV3484', 'AAV3485', 'AAV3779', 'AAV3787', 'AAV3870', 'AAV3880', 'AAV3950',
                        'AAV3970', 'AAV4051', 'AAV4060', 'AAV4090', 'AAV4264', 'AAV4378', 'AAV4711', 'AAV4748',
                        'AAV4859', 'AAV5152', 'AAV5272', 'AAV5291', 'AAV5293', 'AAV5295', 'AAV5296', 'AAV5306',
                        'AAV5402', 'AAV5515', 'AAV5538', 'AAV5593', 'AAV5882', 'AAV5987', 'AAV6010', 'AAV6061',
                        'AAV6065', 'AAV6821', 'AAV7232', 'AAW4678', 'AAW9826', 'AAW9827', 'AAW9828', 'AAW9851',
                        'AAW9853', 'AAW9854', 'AAW9855', 'AAW9856', 'AAW9857', 'AAW9858', 'AAW9955', 'AAW9956',
                        'AAW9957', 'AAW9958', 'AAZ1056', 'AAZ4212', 'AAZ4229', 'AAZ4238', 'AAZ4390', 'AAZ4397',
                        'AAZ4399', 'AAZ4408', 'AAZ4417', 'AAZ442', 'AAZ4423', 'AAZ4425', 'AAZ4442', 'AAZ4931',
                        'ABA8734', 'ABA9032', 'ABA9039', 'ABA9100', 'ABA9538', 'ABA9539', 'ABA9649', 'ABA9650',
                        'ABA9756', 'ABA9833', 'ABA9848', 'ABA9892', 'ABA9911', 'ABB5168', 'ABB1087', 'ABB25475',
                        'ABB4336', 'ABB5004', 'ABB5148', 'ABB5247', 'ABB5272', 'ABB5441', 'ABB5475', 'ABB5510',
                        'ABB5613', 'ABB5689', 'ABB5718', 'ABB5719', 'ABB5771', 'ABB6024', 'ABB6178', 'ABB6179',
                        'ABB6219', 'ABB6222', 'ABB6674', 'ABB6760', 'ABC17949', 'ABC7897', 'ABC7949', 'ABC8469',
                        'ABC7594', 'ABC7728', 'ABC7898', 'ABC7913', 'ABC7932', 'ABC7934', 'ABC7954', 'ABC8116',
                        'ABC8685', 'ABC8792', 'ABC8993', 'ABC8999', 'ABD1055', 'ABD1088', 'ABD1090', 'ABD1106',
                        'ABD1108', 'ABD1112', 'ABD1126', 'ABD1236', 'ABD1254', 'ABD1284', 'ABD2325', 'ABD2343',
                        'ABD2536', 'ABD2570', 'ABD2573', 'ABD2574', 'ABD2985', 'ABD3059', 'ABD3125', 'ABD3193',
                        'ABD3212', 'ABD3218', 'ABD3221', 'ABD3222', 'ABD3226', 'ABD3266', 'ABD3279', 'ABD3292',
                        'ABD3326', 'ABD3369', 'ABD7064', 'ABD8465', 'ABD8466', 'ABE2150', 'ABE2152', 'ABE3314',
                        'ABF6134', 'ABG3040', 'ABG4880', 'ABG4881', 'ABG5403', 'ABG5406', 'ABG5411', 'ABG5418',
                        'ABG5420', 'ABG6341', 'ABG6343', 'ABG6344', 'ABG6350', 'ABG8655', 'ABI6197', 'ABI8685',
                        'ABI8691', 'ABJ5418', 'ABJ7731', 'ABJ8350', 'ABJ8469', 'ABJ8470', 'ABJ8472', 'ABJ8479',
                        'ABJ8574', 'ABJ8591', 'ABJ8720', 'ABJ8961', 'ABJ9357', 'ABJ9538', 'ABJ9932', 'ABK5272',
                        'ABK8761', 'ABK1096', 'ABK1210', 'ABK1407', 'ABK1424', 'ABK1448', 'ABK1466', 'ABK1469',
                        'ABK1529', 'ABK1715', 'ABK1884', 'ABK1991', 'ABK2026', 'ABK2068', 'ABK2094', 'ABK2244',
                        'ABK2248', 'ABK2354', 'ABK2355', 'ABK2459', 'ABK2476', 'ABK2583', 'ABK2632', 'ABK2669',
                        'ABK2834', 'ABK3088', 'ABK3208', 'ABK3218', 'ABK3225', 'ABK3368', 'ABK3453', 'ABK3555',
                        'ABK3649', 'ABK3651', 'ABK3659', 'ABK3746', 'ABK3862', 'ABK4419', 'ABK4484', 'ABK4712',
                        'ABK5681', 'ABK6628', 'ABK6647', 'ABK6678', 'ABK6760', 'ABK6939', 'ABK6954', 'ABK7079',
                        'ABK8179', 'ABK8760', 'ABK8993', 'ABK9305', 'ABK9501', 'ABK9861', 'ABL7486', 'ABL9084',
                        'ABL2493', 'ABL7485', 'ABL7954', 'ABO2973', 'ABO6317', 'ABO8736', 'ABO8738', 'ABO8917',
                        'ABQ1071', 'ABR3648', 'ABR4364', 'ABR4365', 'ABR4384', 'ABR4441', 'ABR6169', 'ABS2517',
                        'ABS6702', 'ABW1717', 'ABW5203', 'ABW5204', 'ABW5205', 'ABW5208', 'ABX7305', 'ABX8205',
                        'ABY3537', 'ABY3538', 'ABY3539', 'ABY3540', 'ABY3541', 'ABZ8843', 'ABZ8844', 'ABZ8845',
                        'ABZ8846', 'ACA1484', 'ACA8183', 'ACA8184', 'ACA8185', 'ACA8186', 'ACA8190', 'ACC9392',
                        'ACD4280', 'ACD4281', 'ACD6545', 'ACD6546',
                        'ACD6549', 'ACD6601', 'ACG7171', 'ACH1037', 'ACH7293', 'ACJ1494', 'ACJ1498', 'ACJ1591',
                        'ACK1587', 'ACK1881', 'ACK2204', 'ACK2280', 'ACK2925', 'ACK3167', 'ACK4421', 'ACK4491',
                        'ACK4492', 'ACK5023', 'ACK6999', 'ACK8296', 'ACK8701', 'ACK9012', 'ACL3816', 'ACM2125',
                        'ACM2127', 'ACM2128', 'ACM2129', 'ACM2158', 'ACN1334', 'ACN1425', 'ACN1488', 'ACN1944',
                        'ACN2205', 'ACN3320', 'ACN3446', 'ACN3607', 'ACN4419', 'ACN4932', 'ACN5172', 'ACN5213',
                        'ACN5576', 'ACN7150', 'ACN7210', 'ACN7293', 'ACN7294', 'ACN7939', 'ACN7944', 'ACN7952',
                        'ACN8742', 'ACN8957', 'ACO1062', 'ACO1517', 'ACO1539', 'ACO2302', 'ACO4829', 'ACO6066',
                        'ACO8652', 'ACO8656', 'ACO8974', 'ACO9895', 'ACP1484', 'ACP2484', 'ACP4145', 'ACP4146',
                        'ACP4147', 'ACQ4919', 'ACQ4920', 'ACY2603', 'ACZ3504', 'ACZ3509', 'ACZ3512', 'ADA1122', 'ADA9993', 'ADA9994', 'ADA9995', 'ADB1103', 'ADB1104',
                        'ADB2452', 'ADB2453', 'ADB3279', 'ADD5980', 'ADH2007', 'ADH2892', 'ADH2893', 'ADH2894',
                        'ADH2895', 'ADH3057', 'ADJ2649', 'ADJ2650', 'ADJ2651', 'ADJ2652', 'ADJ6191', 'ADJ7402',
                        'ADJ7403', 'ADJ7406', 'ADJ7408', 'ADJ7411', 'ADJ7412', 'ADJ7719', 'ADJ7723', 'ADJ7725',
                        'ADJ7726', 'ADJ7728', 'ADJ7729', 'ADJ7730', 'ADJ7731', 'ADJ7733', 'ADJ7737', 'ADJ7739',
                        'ADJ8499', 'ADK1186', 'ADK1188', 'ADK1190', 'ADK1715', 'ADK2137', 'ADK2594', 'ADK3969',
                        'ADK3992', 'ADK5030', 'ADK5117', 'ADK5220', 'ADK5260', 'ADK6065', 'ADK6083', 'ADL8318',
                        'ADL8320', 'ADL8323', 'ADL9554', 'ADL9813', 'ADM3995', 'ADM3996', 'ADM3997', 'ADY173', 'AE003',
                        'AEA9352', 'AFA8522',
                        'AFA2637', 'AFA4744', 'AFA5872', 'AFA5963', 'AFA5965', 'AFA5969', 'AFA5970', 'AFA6238',
                        'AFA6423', 'AFA6656', 'AFA6716', 'AFA6718', 'AFA6722', 'AFA6960', 'AFA7064', 'AFA7081',
                        'AFA7218', 'AFA7226', 'AFA7255', 'AFA7322', 'AFA7434', 'AFA7563', 'AFA7564', 'AFA7611',
                        'AFA7637', 'AFA7697', 'AFA7698', 'AFA7703', 'AFA7705', 'AFA7869', 'AFA7881', 'AFA7905',
                        'AFA8372', 'AFA8381', 'AFA8382', 'AFA8497', 'AFA8568', 'AFA8572', 'AFA8830', 'AFA8896',
                        'AFA8899', 'AGA6208', 'AGA6209', 'AGA6212', 'AGA6213', 'AGA6242', 'AGA6311', 'AGA6650',
                        'AGA6651', 'AGA6700', 'AGA6831', 'AGA6891', 'AGA6949', 'AGA6959', 'AGA7154', 'AGA7155',
                        'AGA7156', 'AGA7157', 'AGA7158', 'AGA7173', 'AGA7506', 'AGA7539', 'AGA7560', 'AGA8530',
                        'AGA8532', 'AGA8533', 'AGA8535', 'AGA8536', 'AGA8639', 'AHA6636', 'AHA7693', 'AHF239',
                        'AIA4082', 'AIA4083', 'AIA3851', 'AIA4089', 'AIA4091', 'AIA7321', 'AIA8720', 'AJA7700',
                        'AJA7711', 'AJA7722', 'AKA2412', 'AKA5099', 'ALA4402', 'ALA7284', 'ALA7321', 'ALA7333',
                        'ALA7447', 'ALA7608', 'ALA7612', 'ALA7614', 'ALA7625', 'ALA7750', 'ALA7927', 'ALA7945',
                        'ALA7966', 'ALA8209', 'ALA8298', 'ALA8324', 'ALA8363', 'ALA8585', 'ALA8602', 'ALA8801',
                        'ALA8805', 'ALA8907', 'ALA8909', 'ALA9048', 'ALA9200', 'ALA9281', 'ALA9338', 'ALA9343',
                        'ALA9431', 'ALA9434', 'ALA9519', 'ALA9638', 'ALA9652', 'ALA9732', 'ALA9887',
                        'NEG3773', 'NDZ5699', 'AMA,7937', 'AMA1043', 'AMA1263', 'AMA1415', 'AMA1466', 'AMA1722',
                        'AMA1746', 'AMA1764', 'AMA2010', 'AMA2013', 'AMA2026', 'AMA210', 'AMA2145', 'AMA2159',
                        'AMA2163', 'AMA2399', 'AMA2589', 'AMA2641', 'AMA2739', 'AMA2925', 'AMA2946', 'AMA7932',
                        'AMA7937', 'AMA7939', 'ANA4145', 'ANA5145', 'ANR547', 'APA3079', 'APA3332', 'APA3681',
                        'APA3696', 'APA2804', 'APA3072', 'APA3078', 'APA3149', 'APA3310', 'APA3318', 'APA3364',
                        'APA3537', 'APA3542', 'APA3580', 'APA3604', 'APA3613', 'APA3664', 'APA3666', 'APA3668',
                        'APA3672', 'APA3705', 'APA4046', 'APA4362', 'APA4364', 'APA4365', 'APA4366', 'APA4367',
                        'APA4370', 'APA4371', 'APA4741', 'APA4742', 'APA4743', 'APA4744', 'APA4752',
                        'APA4761', 'APA4945', 'APA4946', 'APA5465', 'APA5485', 'APA5512', 'APA5626', 'APA5639',
                        'APA5762', 'APA5905', 'APA5907', 'APA5974', 'APA6009', 'APA6232', 'APA6246', 'APA6248',
                        'APA6348', 'APA7698', 'APA8542', 'APA8649', 'APA8650', 'AQA4130', 'AQA4145', 'AQA4146',
                        'AQA4149', 'AQA4154', 'AQA4155', 'AQA4162', 'AQA4164', 'AQA4165', 'AQA6132', 'AQA9353',
                        'AQA9876', 'AQA9877', 'NBG4647', 'ASA2755', 'ASA2756', 'ASA6703', 'ASA8073', 'ASA8307',
                        'ASA8535', 'ASA8724', 'ATA5099', 'ATA5762', 'ATA5764', 'ATA5767', 'ATA5770', 'ATA7058',
                        'ATA7060', 'ATI123', 'ATI212', 'ATI321', 'ATI456', 'AUS8066', 'AVA4327', 'AVA5034', 'AVA3314',
                        'AVA3316', 'AVA3317', 'AVA4320', 'AVA4350', 'AVA4396', 'AVA4397', 'AVA4398', 'AVA4452',
                        'AVA4522', 'AVA4529', 'AVA4542', 'AVA4572', 'AVA4609', 'AVA4619', 'AVA4809', 'AVA4815',
                        'AVA4817', 'AVA4975', 'AVA5051', 'AVA5074', 'AVA5075', 'AVA5130', 'AVA5132', 'AVA5152',
                        'AWA4639', 'AWA3150', 'AWA3151', 'AWA3152', 'AWA3457', 'AWA3480', 'AWA3698', 'AWA3778',
                        'AWA3943', 'AWA4035', 'AWA4145', 'AWA4267', 'AWA4443', 'AWA4696',
                        'AWA4944', 'AWA4961', 'AWA4969', 'AWA5027', 'AWA5060', 'AWA5061', 'AWA5074', 'AWA5082',
                        'AWA5205', 'AWA5208', 'AWA5280', 'AWA5297', 'AWA5532', 'AWA5643', 'AWA5765', 'AWA5771',
                        'AWA5939', 'AWA6185', 'AWA6612', 'AWA6756', 'AWA6877', 'AWA7051', 'AWA7053', 'AWA713',
                        'AWA7256', 'AWA7317', 'AWA7522', 'AWA9803', 'AXA1234', 'AXA2149', 'BBT270', 'BCY615',
                        'BKY232', 'BUA1628', 'BVN232', 'BVN241', 'BVN624', 'BVR487', 'BVR497', 'BVR876', 'BVR905',
                        'BVT270', 'BVT280', 'BVV357', 'BVV368', 'C15973', 'CA1969', 'CAA3116', 'CAA4456', 'CAA4793',
                        'CAA6035', 'CAA6159', 'CAA6836', 'CAA8746', 'CAB3223', 'CAB3709', 'CAB9841', 'CAB1638',
                        'CAB1639', 'CAB1692',
                        'CAB1962', 'CAB2345', 'CAB3163', 'CAB3244', 'CAA6158', 'CAB6874', 'CAB8674', 'CAB8984',
                        'CAB9594', 'CAB9834', 'CAB9837', 'CAB9841', 'CAC5014', 'CAC4465', 'CAC5019', 'CAC5362',
                        'CAC5365', 'CAC7456', 'CAC9633', 'CAD2183', 'CAD1182', 'CAD2187', 'CAD2630', 'CAD5954',
                        'CAD5956', 'CAD9841', 'CAE2443', 'CAE2791', 'CAE3562', 'CAE4244', 'CAE4258', 'CAE4722',
                        'CAE5157', 'CAE5362', 'CAE5442', 'CAE5448', 'CAE5450', 'CAE5456', 'CAE6631', 'CAE6633',
                        'CAE6635', 'CAE8642', 'CAE9459', 'CAE9460', 'CAE9464', 'CAF1120', 'CAF1125', 'CAF1208',
                        'CAF1268', 'CAF1941', 'CAF4090', 'CAF5115', 'CAF7269', 'CAF7271', 'CAF7486', 'CAF7500',
                        'CAF8947', 'CAF9875', 'CAG4909', 'CAG3123', 'CAG3913', 'CAG4336', 'CAG4355', 'CAG4357',
                        'CAG4363', 'CAG4627', 'CAG4633', 'CAG5332', 'CAG5747', 'CAG5999', 'CAG6658', 'CAG6999',
                        'CAG8503', 'CAG9122', 'CAH1969', 'CAH1971', 'CAH3682', 'CAH4522', 'CAH4790', 'CAH4797',
                        'CAH5215', 'CAH7350', 'CAI4062', 'CAI2194', 'CAI2296', 'CAI3031', 'CAI3128', 'CAI433',
                        'CAI4335', 'CAI6695', 'CAJ3022', 'CAJ1046', 'CAJ1047', 'CAJ2294', 'CAJ2296', 'CAJ2537',
                        'CAJ2564', 'CAJ5333', 'CAJ5334', 'CAJ5747', 'CAJ5755', 'CAK1849', 'CAK2977', 'CAK3076',
                        'CAK6259', 'CAL2938', 'CAL3036', 'CAL6751', 'CAL6759', 'CAL6830', 'CAL7559', 'CAL7731',
                        'CAL7736', 'CAL7891', 'CAM8712', 'CAM2317', 'CAM2450', 'CAM2561', 'CAM2631', 'CAM3469',
                        'CAMB8712', 'CAN2932', 'CAN3112', 'CAN3522', 'CAN5143', 'CAN7457', 'CAN7582', 'CAN8957',
                        'CAN9404', 'CAO2511', 'CAO2760', 'CAO6569', 'CAO6765', 'CAO7656', 'CAO7760', 'CAO8399',
                        'CAP1974', 'CAP2567', 'CAP2716', 'CAP4716', 'CAP4734', 'CAP4737', 'CAP6858', 'CAP7230',
                        'CAP8209', 'CAP8372', 'CAP8500', 'CAQ1102', 'CAQ6500', 'CAQ6501', 'CAQ6504', 'CAQ6505',
                        'CAQ6569', 'CAQ6835', 'CAQ8333', 'CAQ8335', 'CAQ8336', 'CAQ8341', 'CAQ8343', 'CAQ8347',
                        'CAR4853', 'CAR2136', 'CAR2142', 'CAR2145', 'CAR2352', 'CAR2353', 'CAR4190', 'CAR7761',
                        'CAR9521', 'CAS4865', 'CAS2425', 'CAS8928', 'CAT2716', 'CAT4059', 'CAT4835', 'CAU2450',
                        'CAU2454', 'CAY4637', 'CAY1072', 'CAY1322', 'CAY2101', 'CAY4447', 'CBF7678', 'CBF8812',
                        'CBJ1926', 'CBJ2747', 'CBJ3048', 'CBJ3283', 'CBJ3287', 'CBJ4369', 'CBJ4378', 'CBJ4550',
                        'CBJ4926', 'CBJ4933', 'CBJ4934', 'CBJ4936', 'CBJ4937', 'CBJ5222', 'CBJ5803', 'CBJ7262',
                        'CBJ7926', 'CBJ7928', 'CBJ8057', 'CBJ8059', 'CBJ8825', 'CBJ9187', 'CBJ9193', 'CBJ9235',
                        'CBJ9238', 'CBJ9674', 'CBN5219', 'CBN5226', 'CBN1217', 'CBN1436', 'CBN1479', 'CBN1490',
                        'CBN1555', 'CBN2043', 'CBN2044',
                        'CBN2899', 'CBN3048', 'CBN3675', 'CBN3684', 'CBN5222', 'CBN5224', 'CBN5247', 'CBN7689',
                        'CBP4729', 'CBP1366', 'CBP1371', 'CBP1373', 'CBP3523', 'CBP3527',
                        'CBP4641', 'CBP8013', 'CBQ3132', 'CBQ3457', 'CBQ6288', 'CBQ6982', 'CBX7414', 'CBX7416',
                        'CCM2129', 'CCM2125', 'CCM2158', 'CCN5576', 'CCN1425', 'CCN1488', 'CCN2101', 'CCN5213',
                        'CCN7210', 'CCN7294', 'CCN7939', 'CCN7952', 'CCN8957', 'CCO6708', 'CCO6780', 'CCO1517',
                        'CCO1539', 'CCO1957', 'CCO2302', 'CCO3246', 'CCO3827', 'CCO3970', 'CCO5059', 'CCO6066',
                        'CCO6068', 'CCO7760', 'CCO8530', 'CCO8590', 'CCO8652', 'CCO8656', 'CCO8974', 'CCO9224',
                        'CCO9663', 'CDJ2650', 'CDJ2651', 'CDJ2652', 'CDJ3728', 'CDJ6187', 'CDJ6188', 'CDJ6189',
                        'CDJ6190', 'CDJ6191', 'CDJ7402', 'CDJ7408', 'CDJ7412', 
                        'CDJ7730', 'CDJ7733', 'CDJ7736', 'CDJ7737', 'CDJ7739', 'CDJ8499', 'CDJ8542', 'CDJ8546',
                        'CDJ8564', 'CDK6065', 'CDK1188', 'CDK1715', 'CDK1716', 'CDK2265', 'CDK22665', 'CDK2346',
                        'CDK2594', 'CDK2596', 'CDK2889', 'CDK2992', 'CDK3992', 'CDK5030', 'CDK5117', 'CDK5220',
                        'CDK715', 'CDK9521', 'CDM843N', 'CDR6065', 'CFB2345', 'CK6999', 'CN1634', 'CNE601', 'CNK135',
                        'CNK153', 'CNL360', 'CNN341', 'CNN323', 'CNN8957', 'CO5157', 'CO5166', 'CO8832', 'COA7760',
                        'CP5356', 'CPT301', 'CQA6505', 'CR1669', 'CR1674', 'CR6115', 'CR6121', 'CRU589', 'CS6920',
                        'CS6958', 'CS6961', 'CSB823', 'CSG324', 'CTU943', 'CTW322', 'CTX773', 'CTY301', 'CUR676',
                        'CUT436', 'CUT580', 'CXF912', 'CXK640', 'CXK648', 'CXM679', 'CXM724', 'CXM970', 'CXN365',
                        'CXN387', 'CXR164', 'CXS658', 'CXS713', 'CXS725', 'CXS746', 'DAA7614', 'DAA9818', 'DAA9819',
                        'DAB4152', 'DAB4164', 'DAB7513', 'DAB7515', 'DAB7517', 'DAB7518', 'DAB9196', 'DAC3828',
                        'DAD6119', 'DAD7209', 'DAD7210', 'DAD7211', 'DAD7212', 'DAD7213', 'DAD7214', 'DAD7215',
                        'DAD7216', 'DAD7218', 'DAD8504', 'DAD9969', 'DAE112', 'DAE113', 'DAE114', 'DAE115', 'DAE123',
                        'DAE1924', 'DAE321', 'DAE8432', 'DAE9998', 'DAF7137', 'DAH7631', 'DAI3272', 'DAI3273',
                        'DAI7200', 'DAK2120', 'DAK2153', 'DAK2154', 'DAK2156', 'DAK2157', 'DAM1019', 'DAM1035',
                        'DAM1037', 'DAO6101', 'DAO6102', 'DAO6103', 'DAO6104', 'DAO6108', 'DAO6109', 'DAO6110',
                        'DAO6111', 'DAO6118', 'DAO6119', 'DAQ3082', 'DAQ3083', 'DAQ3084', 'DBX8844', 'DBX9306',
                        'DBX9601', 'DBX9609', 'DBX9610', 'DBX9611', 'DBX9612', 'DBX9613', 'DBY3537', 'DBY3538',
                        'DBY3539', 'DBY3540', 'DBY3541', 'DBZ8843', 'DBZ8844', 'DBZ8845', 'DBZ8846', 'DCP4145',
                        'DCP4146', 'DCP4147', 'DCP4479', 'DCQ1650', 'DCQ1651', 'DCQ1652', 'DCQ1655', 'DCQ1656',
                        'DCQ3036', 'DCQ3037', 'DCQ3038', 'DCQ3039', 'DCQ3105', 'DCQ4912', 'DCQ4919', 'DCQ4920',
                        'DCR1123', 'DDL8318', 'DDL8319', 'DDL8320', 'DDL8321', 'DDL8794', 'DDL8796', 'DDL8797',
                        'DDL8800', 'DDL8803', 'DDL8804', 'DDL9554', 'DDL9555', 'DDL9556', 'DDL9813', 'DDM3995',
                        'DDM3996', 'DDM3997', 'DEB2031', 'DEB2301', 'DEB2304', 'DEB2306', 'DEB2345', 'DIY531', 'DOD127',
                        'DSS159', 'DWJ164', 'DWJ182', 'DWK530', 'DWP995', 'DWR438', 'DWR496', 'DWT974', 'DXB404',
                        'DXB494', 'DXF856', 'DXM180', 'DXM190', 'DXN722', 'DXP726', 'DXP747', 'DXU926', 'DXW835',
                        'DXX997', 'DXY501', 'DXY511', 'DXY515', 'DXZ193', 'DXZ214', 'DXZ215', 'DXZ228', 'DXZ238',
                        'DXZ246', 'DXZ506', 'DXZ529', 'DXZ818', 'EFM1234', 'EFM2345', 'EVT763', 'FAF1751', 'FAF7990',
                        'FAW1528', 'FAW36', 'FAW37', 'FAW38', 'FAW39', 'FAW40', 'FCI814', 'FJH131', 'FJH132', 'ZRL184',
                        'NBM5516', 'GAL2835', 'GAL2836', 'GAL2935', 'GAO6892', 'GAO9250', 'CBN5697', 'GAV6804',
                        'GB3304', 'GB6161', 'GB7786', 'GB7787', 'GB7788', 'GB7789', 'GB7792', 'GB7793', 'GB7794',
                        'GB7795', 'GB7796', 'GB9517', 'GB9523', 'GDS846', 'GEZ276', 'GS1061', 'HBC4407', 'HRL780',
                        'IAA4205', 'JAL1081', 'JAL2174', 'JCT7714', 'JEF33', 'JF7013', 'JM0323', 'JM0680', 'JM0682',
                        'JM0860', 'JM1497', 'JM1686', 'JM2312', 'JM2314', 'JM2422', 'JM2457', 'JM2467', 'JM2892',
                        'JM2901', 'JM3276', 'JM3277', 'JM3278', 'JM3281', 'JM3285', 'JM3286', 'JM3319', 'JM3320',
                        'JM3321', 'JM3332', 'JM5659', 'JM5712', 'JM6143', 'JM6767', 'JM6782', 'JM7356', 'JM7672',
                        'JM7675', 'JM8127', 'JM8420', 'JM8428', 'JM8500', 'JMS3339', 'JMS3345', 'JOE202', 'JPE486',
                        'NCX8264', 'KAC9395', 'KAD3439', 'KAD4391', 'KAH5581', 'KAH5881', 'KAK5845', 'NET3270',
                        'NEW7823', 'KVU966', 'KVU970', 'KVV992', 'KVY852', 'KX133831', 'L1950', 'LA1967', 'LAB2452',
                        'LAB2457', 'LAB2460', 'LAB2463', 'LAB2467', 'LAB2470', 'LAG8576', 'LAI6174', 'LAJ6174',
                        'CBN7864', 'LIB102', 'LXA328', 'LXB812', 'LXB977', 'LXC256', 'LXC305', 'LXC471', 'LXD918',
                        'M0D906', 'MAA4077', 'MAD4798/301982', 'MAD4799/301985', 'MAD8982/320316', 'MAD8983/320319',
                        'MAF1729/479397', 'MAF1730/479397', 'MAF1731/479399', 'MAF5441', 'MAF5442/355236',
                        'MAF5443/355239', 'MAF5444/355244', 'MAF5445/355246', 'MAF5446/355243', 'MAF8981/320315',
                        'MAH2358', 'MAK1999', 'MAK5058', 'MAK5067', 'MAK5096', 'MAK6920', 'MAL1522', 'MAL4442',
                        'MAL4451', 'MAN8653', 'MAO6317', 'MAP1730', 'MAP1732/479402', 'MAP1733/479404',
                        'MAP1734/479406', 'MAV3765', 'MAX7442', 'MB136047854', 'MCG998', 'MCV9595', 'MCW7875', 'MGE622',
                        'MGE672', 'MGF621', 'MGK320', 'MOC287', 'MOF108', 'NAB3442', 'NAB3439', 'NAB3440', 'NAB3441',
                        'NAB3442', 'NAB3469', 'NAB3492', 'NAB3496', 'NAB4421', 'NAB9403', 'NAB9404', 'NAB9405',
                        'NAB9406', 'NAB9407', 'NAC7996', 'NAC8009', 'NAD5373', 'NAD5374', 'NAD6592', 'NAD6593',
                        'NAD6596', 'NAD6785', 'NAE5040', 'NAE5961', 'NAE5967', 'NAE5979', 'NAE5980', 'NAE5981',
                        'NAE5982', 'NAE8951', 'NAE9580', 'NAE9929', 'NAF7986', 'NAG4360', 'NAG4380', 'NAG4384',
                        'NAG4421', 'NAG8984', 'NAG9328', 'NAG9405', 'NAH7155', 'NAI9036', 'NAI2597', 'NAI2611',
                        'NAI7657', 'NAI8031', 'NAJ1266', 'NAJ1267', 'NAJ1314', 'NAJ1315', 'NAJ4439', 'NAJ4478',
                        'NAJ4481', 'NAJ4489', 'NAJ4607', 'NAJ5941', 'NAJ5982', 'NAJ6420', 'NAJ6422', 'NAJ6679',
                        'NAJ6680', 'NAJ6681', 'NAJ6682', 'NAJ6683', 'NAK1941', 'NAK1942', 'NAK1943', 'NAK1948',
                        'NAK3480', 'NAK9419', 'NAK9477', 'NAK9690', 'NAK9691', 'NAK9712', 'NAK9970', 'NAK9974',
                        'NAL2499', 'NAL6771', 'NAL6908', 'NAL6909', 'NAL6917', 'NAL7397', 'NAL771', 'NAL9205',
                        'NAM9135', 'NAM1219', 'NAM2908', 'NAM3541', 'NAM3742', 'NAM3929', 'NAM3930', 'NAM3931',
                        'NAM3932', 'NAM3933', 'NAM6685', 'NAM6686', 'NAM6753', 'NAM7895', 'NAM9134', 'NAM9164',
                        'NAN4644', 'NAN4915', 'NAN7124', 'NAN7125', 'NAN3515', 'NAN4746', 'NAN4748', 'NAN7541',
                        'NAO3967', 'NAO3968',
                        'NAO3969', 'NAP6771', 'NAP6784', 'NAP6807', 'NAP3833', 'NAP4649', 'NAP6748', 'NAP6786',
                        'NAP6795', 'NAP7440', 'NAP7441', 'NAP7443', 'NAP7469', 'NAQ9582', 'NAR1799', 'NAR1800',
                        'NAR2714', 'NAS2273', 'NAS2280', 'NAS2281', 'NAS2282', 'NAS2283', 'NAS2284', 'NAS5892',
                        'NAS5898', 'NAS6651', 'NAS9586', 'NAS9590', 'NAS9595', 'NAS9597', 'NAS9598', 'NAS9599',
                        'NAS9842', 'NAS9843', 'NAS9845', 'NAS9847', 'NAS9895', 'NAS9945', 'NAT2633', 'NAT2637',
                        'NAT2833', 'NAT3428', 'NAT3495', 'NAT3539', 'NAT3540', 'NAT3541', 'NAU7933', 'NAU9293',
                        'NAU9358', 'NAU9443', 'NAU9549', 'NAU9699', 'NAV1862', 'NAV1985', 'NAV1986', 'NAV5900',
                        'NAV5901', 'NAV5902', 'NAV5903', 'NAV6243', 'NAV6542', 'NAV6562', 'NAV6603', 'NAV6604',
                        'NAV7522', 'NAV9358', 'NAW2038', 'NAW2899', 'NAW4291', 'NAW4383', 'NAW4439', 'NAW4649',
                        'NAW4939', 'NAW5092', 'NAW5093', 'NAW5095', 'NAW5097', 'NAW7713', 'NAW9294', 'NAW9573',
                        'NAW9575', 'NAW9577', 'NAX2908', 'NAX5416', 'NAX6883', 'NAX6884', 'NAX6885', 'NAX6887',
                        'NAX6888', 'NAY6914', 'NAY1188', 'NAY3711', 'NAY4355', 'NAY4364', 'NAY4392', 'NAY4393',
                        'NAY4469', 'NAY4696',
                        'NAY4699', 'NAY6243', 'NAY6895', 'NAY6941', 'NAY6943', 'NAZ2177', 'NAZ2178', 'NAZ3876',
                        'NAZ3877', 'NAZ4063', 'NAZ4064', 'NAZ4065', 'NAZ4066', 'NAZ4067', 'NAZ4069', 'NAZ4070',
                        'NAZ4071', 'NAZ4072', 'NAZ4073', 'NAZ4076', 'NAZ4077', 'NAZ4078', 'NAZ4079', 'NAZ4080',
                        'NAZ4081', 'NAZ4082', 'NAZ4084', 'NAZ4085', 'NAZ4086', 'NAZ4087', 'NAZ4089', 'NAZ4090',
                        'NAZ4278', 'NAZ9590', 'NBA1770', 'NBA1880', 'NBA3597', 'NBA5397', 'NBA8825', 'NBA9651',
                        'NBA9657', 'NBB1239', 'NBB1241', 'NBB1242', 'NBB3316', 'NBB3523', 'NBB4647', 'NBB5791',
                        'NBB5831', 'NBB7647', 'NBC4340', 'NBC4356', 'NBC4357', 'NBC4358', 'NBC4407', 'NBC5260',
                        'NBC6224', 'NBC6225', 'NBC6226', 'NBC6227', 'NBC6228', 'NBC6688', 'NBC9547', 'NBD3902',
                        'NBD4506', 'NBD5230', 'NBD5831', 'NBD8328', 'NBD9242', 'NBD9305', 'NBE1071', 'NBE5084',
                        'NBE5117', 'NBE5120', 'NBE5122', 'NBE5124', 'NBE5126', 'NBE6193', 'NBE6279', 'NBE6643',
                        'NBE7497', 'NBE7511', 'NBE7533', 'NBE7534', 'NBE8829', 'NBE9464', 'NBE9467', 'NBF2074',
                        'NBF2081', 'NBF4714', 'NBF4718', 'NBF4743', 'NBF4744', 'NBF9320', 'NBG1936', 'NBG9174',
                        'NBG2790', 'NBG3378', 'NBG3916',
                        'NBG4334', 'NBG6688', 'NBG6788', 'NBG8193', 'NBG8499', 'NBG8540', 'NBG9054', 'NBG9174',
                        'NBG9193', 'NBH9493', 'NBH2386', 'NBH4519', 'NBH8193', 'NBH8574', 'NBH9002', 'NBH9193',
                        'NBH9491', 'NBI3921', 'NBI4172', 'NBI4504', 'NBI4640', 'NBI4641', 'NBI4644', 'NBI4647',
                        'NBI4747', 'NBI4749', 'NBI6753', 'NBI6944', 'NBJ1977', 'NBJ1978', 'NBJ1980', 'NBJ1981',
                        'NBJ1982', 'NBJ1983', 'NBJ2775', 'NBJ4550', 'NBJ5941', 'NBJ6873', 'NBJ6925', 'NBJ8168',
                        'NBJ8186', 'NBJ8254', 'NBJ8255', 'NBJ8256', 'NBJ8825', 'NBK1059', 'NBK1062', 'NBK1063',
                        'NBK1064', 'NBK220', 'NBK2409', 'NBK2411', 'NBK2412', 'NBK2413', 'NBK2414', 'NBK2415',
                        'NBK2416', 'NBK4306', 'NBK4307', 'NBK4308', 'NBK4309', 'NBK4311', 'NBK4313', 'NBL124',
                        'NBL1801', 'NBL1868', 'NBL1869', 'NBL4127', 'NBL4501', 'NBL4502', 'NBL4503', 'NBL4504',
                        'NBL4505', 'NBL4516', 'NBL4616', 'NBL8778', 'NBL8795', 'NBL8825', 'NBL9238', 'NBL9736',
                        'NBL9737', 'NBL9738', 'NBM2558', 'NBM1270', 'NBM1885', 'NBM2575', 'NBM4553', 'NBM5514',
                        'NBM5964', 'NBM8343', 'NBM8348', 'NBM9811', 'NBN1038', 'NBN5819', 'NBN9333', 'NBO4984',
                        'NBO2345', 'NBO3569', 'NBO3570', 'NBO3571', 'NBO3572', 'NBO3846', 'NBO4937', 'NBO4971',
                        'NBO4973', 'NBO5178', 'NBO5638', 'NBO6012', 'NBO7192', 'NBO7193', 'NBO8604', 'NBO8861',
                        'NBO8970', 'NBP3957', 'NBQ9582', 'NBQ1145', 'NBQ1150', 'NBQ2815', 'NBQ3519', 'NBQ4403',
                        'NBQ4423', 'NBQ7388', 'NBQ7389', 'NBQ7390', 'NBQ7391', 'NBQ7834', 'NBQ7977', 'NBQ9756',
                        'NBR9294', 'NBR2883', 'NBR4525', 'NBR4684', 'NBR4744', 'NBR6205', 'NBS3569', 'NBS3544',
                        'NBS3545', 'NBS3606', 'NBS5004', 'NBS6230', 'NBS6232', 'NBS6528', 'NBS7015', 'NBS7202',
                        'NBS7332', 'NBS7441', 'NBS7442', 'NBT1507', 'NBT1534', 'NBT3068', 'NBT4910', 'NBT5383',
                        'NBT6191', 'NBT6192', 'NBT6193', 'NBT6194', 'NBT6195', 'NBT8255', 'NBT9467', 'NBU8987',
                        'NBU8988', 'NBW5203', 'NBW5204', 'NBW5205', 'NBW5208', 'NBW7790', 'NBX2646', 'NBX7790',
                        'NBX9742', 'NBY2646', 'NCK8701', 'NCA8209', 'NCA8219', 'NCC1563', 'NCC9391', 'NCC9392',
                        'NCD1452', 'NCD1454', 'NCD4280', 'NCD4281', 'NCD6603', 'NCD8708', 'NCE1125', 'NCE7717',
                        'NCE7718', 'NCF3365', 'NCG1104', 'NCG2375', 'NCG3275', 'NCG3276', 'NCG3773', 'NCH6168',
                        'NCI9943', 'NCJ1383', 'NCJ1591', 'NCK6994', 'NCK1948', 'NCK2204', 'NCK2360', 'NCK2361',
                        'NCK2916', 'NCK2923', 'NCK3095', 'NCK3167', 'NCK4132', 'NCK4861', 'NCK5023', 'NCK5988',
                        'NCK6099', 'NCK6100', 'NCK6999', 'NCK7399', 'NCK7569', 'NCK7586', 'NCK7874', 'NCK7996',
                        'NCK8232', 'NCK8296', 'NCK8699', 'NCK9664', 'NCL3445', 'NCL3447',  'NCN9300',
                        'NCN9363', 'NCN9364', 'NCN9365', 'NCN9366', 'NCN9367', 'NCO3276', 'NCO7028', 'NCP1122',
                        'NCP1123', 'NCP1124', 'NCP1125', 'NCP1126', 'NCP9485', 'NCQ3984',
                        'NCQ3986', 'NCQ4595', 'NCQ4596', 'NCQ4597', 'NCQ4598', 'NCQ4599', 'NCQ4600', 'NCQ4601',
                        'NCQ5482', 'NCQ5771', 'NCQ5772', 'NCQ5773', 'NCQ5774', 'NCQ5775', 'NCQ5776', 'NCQ5777',
                        'NCQ5778', 'NCQ5779', 'NCQ9481', 'NCQ9482', 'NCQ9484', 'NCQ9485', 'NCQ9486', 'NCQ9487',
                        'NCQ9488', 'NCR7235', 'NCR7236', 'NCT2441', 'NCV7919', 'NCV7920', 'NCV7921', 'NCV7922',
                        'NCW6210', 'NCW6213', 'NCW6219', 'NCW6220', 'NCW6222', 'NCX3264', 'NCX4327', 'NCX5249',
                        'NCX5427', 'NCX5429', 'NCX8270', 'NCX8273', 'NCX8274', 'NCX8302', 'NCY1745', 'NCZ1425',
                        'NCZ3241', 'NCZ3242', 'NCZ3512', 'NCZ7947', 'NCZ8119', 'NCZ8620', 'NCZ8626',
                        'NDA1648', 'NDA1649', 'NDA6222', 'NDB1104', 'NDB2452', 'NDB2453', 'NDB6218', 'NDC4831',
                        'NDC6226', 'NDD2208', 'NDD2209', 'NDD6461', 'NDE4310', 'NDE4311', 'NDE5741', 'NDE5982',
                        'NDH2007', 'NDH2678', 'NDH2788', 'NDH2876', 'NDH2893', 'NDH2895', 'NDH3456', 'NDI5202',
                        'NDI5203', 'NDI5204', 'NDI6952', 'NDJ3475', 'NDJ3758', 'NDJ4687', 'NDJ4693', 'NDJ5002',
                        'NDJ6734', 'NDJ7303', 'NDK1951', 'NDK5102', 'NDK5257', 'NDK6651', 'NDK7104', 'NDK9169',
                        'NDK9170', 'NDK9171', 'NDK9172', 'NDK9173', 'NDL2099', 'NDL2256', 'NDL2259', 'NDL3071',
                        'NDL3072', 'NDL3073', 'NDL4437', 'NDL4438', 'NDL4494', 'NDL6524', 'NDL6952', 'NDL7647',
                        'NDL9486', 'NDM2752', 'NDM2798', 'NDM5773', 'NDM5908', 'NDM8723', 'NDM9251', 'NDN6493',
                        'NDN1287', 'NDN3832', 'NDN5818', 'NDN6961', 'NDN6962', 'NDN8496', 'NDN9333', 'NDN9854',
                        'NDN9944', 'NDN9945', 'NDO1139', 'NDO1193', 'NDO2209', 'NDO7874', 'NDP5680',
                        'NDP7593', 'NDP8213', 'NDQ1583', 'NDQ4827', 'NDQ6536', 'NDQ6537',
                        'NDS9111', 'NDS9112', 'NDS9113', 'NDS9114', 'NDS9115', 'NDS9116', 'NDX9663', 'NDY4182',
                        'NDZ1651', 'NED5231', 'NEF1803', 'NEF5752', 'NEG1415', 'NEG1436', 'NEG3488', 'NEG5383',
                        'NEG7684', 'NEG8900', 'NEG1717', 'NEG1846', 'NEG1942', 'NEG2162', 'NEG2163', 'NEG2173',
                        'NEG2462', 'NEG2463', 'NEG3635', 'NEG4462', 'NEG6023', 'NEG7317', 'NEG7340', 'NEG8539',
                        'NEG9660', 'NEI1048', 'NEI1053', 'NEJ2513', 'NEJ2514', 'NEJ3317', 'NEJ3345', 'NEJ4476',
                        'NEM3700', 'NEO7940', 'NEP4232', 'NEP4233', 'NEP7642', 'NER8755', 'NES2072', 'NES5532',
                        'NES5533', 'NES6316', 'NES6623', 'NES7291', 'NET3076', 'NET3078', 'NET3095', 'NET8956',
                        'NEV9593', 'NEV9870', 'NEW4347', 'NEW4373', 'NEW4374', 'NEW4963', 'NEW8243', 'NEX5158',
                        'NEX9322', 'NEX5157', 'NEX5761', 'NEX5786', 'NEX5789', 'NEX5800', 'NEX9323', 'NFC1659',
                        'NFC1660', 'NFC1661',
                        'NFG6432', 'NFG7684', 'NFH1420', 'NFH1421', 'NFK2443', 'NFT1147', 'NFT4334', 'NFT4344',
                        'NFT4967', 'NFT4968', 'NFT7757', 'NFT881', 'NFT8812', 'NFT8821', 'NFX5613', 'NFX5623',
                        'NFY1197', 'NFY1198', 'NFY5764', 'NFY5765', 'NFY8372', 'NFY9043', 'NFZ4269', 'NFZ1148',
                        'NFZ1483', 'NFZ6415', 'NFZ7066', 'NFZ9157', 'NGA1631', 'NGA6380', 'NGA6384', 'NGB9823',
                        'NGB4647', 'NGF1601', 'NGF6432', 'NGF6433', 'NGF6566', 'NGF6697', 'NGF6997', 'NGF7684',
                        'NGG2329', 'NGG4852', 'NGG4853', 'NGG5545', 'NGG6195', 'NGG8423', 'NGG9792', 'NGJ1116',
                        'NGJ1408', 'NGJ3654', 'NGK1319', 'NGK6335', 'NGL4122', 'NGL4139', 'NGL4395', 'NGM4433',
                        'NGP3055', 'NGP1457', 'NGP2598', 'NGP6160', 'NGP9725', 'NGR1030', 'NGR1031', 'NGR3521',
                        'NGR4640', 'NGS2856', 'NGS2848', 'NGS2850', 'NGS4151', 'NGS4402', 'NGS5687', 'NGS9642',
                        'NGT1057', 'NGT1052', 'NGT6298', 'NGT6299', 'NGU1908', 'NIE916', 'NIK804', 'NIK873', 'NIT167',
                        'NIW021', 'NIW023', 'NLQ4601', 'NOT462', 'NOT321', 'NQ0924', 'NQE857', 'NQH413', 'NQO924',
                        'NQS259', 'NQU235', 'NQY717', 'NZC3242', 'OM0237', 'OPA656', 'PCZ783', 'PDO230', 'PDO638',
                        'PEI247', 'PEI306', 'PEI487', 'PEI497', 'PEQ833', 'PEQ834', 'PIA313', 'PIA916', 'PIZ744',
                        'PIZ323', 'PKY232',  'PMZ556', 'PNQ249', 'PNQ829', 'POD642', 'POD662', 'POD699',
                        'POE983', 'POO669', 'POO679', 'POO689', 'POO699', 'POO709', 'PQA324', 'PQQ391', 'PQQ411',
                        'PQQ421', 'PQS529', 'PRD114', 'PRW677', 'PTO259', 'PVG132', 'PWJ611', 'PWU206', 'PXB424',
                        'PXR423', 'PXR539', 'PXR962', 'PXS151', 'PXW691', 'PXX386', 'PXY532', 'PYL363', 'PYL905',
                        'PYM321', 'PYM734', 'PYM744', 'PYU356', 'PYX256', 'CBN7690', 'QVO176', 'RAF670', 'RAJ985',
                        'RAK862', 'RAM993', 'RAW564', 'RBC314', 'RBK1991', 'RBP436', 'RCA156', 'RCB489', 'RCC967',
                        'RCD215', 'RCD617', 'RCR393', 'RCY913', 'RCZ783', 'RDS324', 'RDT841', 'RDT160', 'RDX193',
                        'RDZ364', 'RDZ724', 'REA671', 'REB971', 'REL696', 'REP689', 'REW264', 'REW273', 'REZ842',
                        'RFA675', 'RFB590', 'RFC643', 'RFD210', 'RFE861', 'RFF908', 'RFJ376', 'RFL929', 'RFM986',
                        'RFN673', 'RFP453', 'RFS238', 'RFY359', 'RFY679', 'RFY695', 'RGA118', 'RGC223', 'RGD496',
                        'RGF6566', 'RGG362', 'RGH214', 'RGH859', 'RGJ496', 'RGJ602', 'RGJ791', 'RGK863', 'RGK144',
                        'RGK920', 'RGK978', 'RGM362', 'RGM852',
                        'RGR363', 'RGS461', 'RGU771', 'RGV358', 'RGW700', 'RHA780', 'RHA928', 'RHB986', 'RHC124',
                        'RHC179', 'RHD659', 'RHD784', 'RHE124', 'RHE254', 'RHE797', 'RHF218', 'RHF269', 'RHF452',
                        'RHF546', 'RHG500', 'RHJ452', 'RHJ492', 'RHL780', 'RHM271', 'RHM272', 'RHM452', 'RHM482',
                        'RHP131', 'RHS308', 'RHU635', 'RHU783', 'RHU978', 'RHV296', 'RHV383', 'RHV603', 'RHV980',
                        'RHW126', 'RHW342', 'RHY634', 'RHY632', 'RHZ567', 'RJA486', 'RJA571', 'RJA694', 'RJC345',
                        'RJC676', 'RJD192', 'RJD539', 'RJG496', 'RJH492', 'RJH758', 'RJH781', 'RJK196', 'RJK403',
                        'RJK910', 'RJK930', 'RJL114', 'RJM207', 'RJM383', 'RJN358', 'RJN499', 'RJN703', 'RJR464',
                        'RJR623', 'RJS971', 'RJS991', 'RJT204', 'RJT825', 'RJV162', 'RJY311', 'RJY717', 'RJY737',
                        'RJY746', 'RJY747', 'RJY867', 'RJZ225', 'RJZ825', 'RKB425', 'RKC142', 'RKC298', 'RKC503',
                        'RKD487', 'RKD660', 'RKE607', 'RKG819', 'RKL115', 'RKL771', 'RKN261', 'RKP643', 'RKP245',
                        'RKP275', 'RKR725', 'RKT701', 'RKU717', 'RKU985', 'RKV582', 'RKV660', 'RKV763', 'RKX709',
                        'RKX813', 'RKY323', 'RKY491', 'RKZ595', 'RKZ713', 'RLA214', 'RLA813', 'RLC174', 'RLC353',
                        'RLC706',
                        'RLD852', 'RLE671', 'RLE282', 'RLG257', 'RLG596', 'RLH265', 'RLH607', 'RLI948', 'RLJ643',
                        'RLK458', 'RLK536', 'RLK989', 'RLL195', 'RLL485', 'RLL606', 'RLL876', 'RLL948', 'RLM280',
                        'RLM409', 'RLM563', 'RLN141', 'RLN316', 'RLN331', 'RLN658', 'RLP238', 'RLP279', 'RLP298',
                        'RLP308', 'RLP518', 'RLP660', 'RLP714', 'RLP803', 'RLR561', 'RLR879', 'RLR983', 'RLS524',
                        'RLU370', 'RLU541', 'RLU953', 'RLV283', 'RLV284', 'RLV542', 'RLV544', 'RLV841', 'RLX158',
                        'RLX265', 'RLX285', 'RLX701', 'RLX702', 'RLZ681', 'RMB156', 'RMB417', 'RMB437', 'RMB852',
                        'RMB957', 'RMD163', 'RMD335', 'RMD522', 'RME162', 'RME435', 'RME677', 'RME678', 'RME688',
                        'RME697', 'RMF154', 'RMF904', 'RMH517', 'RMH825', 'RMH887', 'RMH916', 'RMJ130', 'RMJ165',
                        'RMJ280', 'RMJ329', 'RMJ330', 'RMJ620', 'RMJ689', 'RMJ725', 'RMJ906', 'RMJ916', 'RMJ985',
                        'RMK119', 'RMK594', 'RMK746', 'RMK924', 'RML163', 'RML427', 'RMN113', 'RMN651', 'RMN861',
                        'RMP285', 'RMP113', 'RMP679', 'RMP708', 'RMR162', 'RMR254', 'RMR338', 'RMR396', 'RMR726',
                        'RMS681', 'RMS724', 'RMS744', 'RMS949', 'RMT410', 'RMT532', 'RMT572', 'RMT589', 'RMT848',
                        'RMT859',
                        'RMV801', 'RMW234', 'RMX701', 'RMX715', 'RMX892', 'RMX924', 'RMX983', 'RMZ742', 'RMZ763',
                        'RMZ793', 'RNA326', 'RNA489', 'RNA512', 'RNA543', 'RNA852', 'RNA872', 'RNA881', 'RNA892',
                        'RNB363', 'RNB651', 'RNC194', 'RNC571', 'RNC772', 'RND151', 'RND192', 'RND216', 'RND347',
                        'RND991', 'RNE147', 'RNE185', 'RNE244', 'RNE316', 'RNE575', 'RNE687', 'RNE905', 'RNE906',
                        'RNG254', 'RNG365', 'RNG873', 'RNG917', 'RNH386', 'RNH865', 'RNJ289', 'RNJ498', 'RNJ985',
                        'RNJ993', 'RNK185', 'RNK416', 'RNK526', 'RNK575', 'RNK599', 'RNK796', 'RNK924', 'RNK934',
                        'RNK949', 'RNK956', 'RNL480', 'RNL826', 'RNV938', 'RNW121', 'RNW262', 'RNW337', 'RNW564',
                        '003794', 'TBA470', 'TCI696', 'TCI814', 'TCI907', 'TCI917', 'TDH470', 'TDQ579', 'TDQ589',
                        'TDY1229', 'TEST001', 'TEST002', 'TEST003', 'TEST004', 'TEST005', 'TEST006', 'TFY507', 'TXE487',
                        'TGI731', 'TGV121', 'THQ214', 'THQ903', 'THQ967', 'THZ785', 'TIM343', 'TIU226', 'TIU236',
                        'TIU590', 'TIU674', 'TKQ831', 'TLZ163', 'TNR547', 'TNY964', 'TNZ768', 'TOP366', 'TPR561',
                        'TQA324', 'TQE877', 'TQO147', 'TQR764', 'TQW382', 'TQZ329', 'TQZ408', 'TQZ497', 'TQZ794',
                        'TRCK001', 'NEV9594', 'TRW677', 'TSZ785', 'TUF478', 'TVC817', 'TVC857', 'TVE950', 'TVG909',
                        'TVK258', 'TVL844', 'TVM267', 'TVM574', 'TVM584', 'TVM594', 'TVN436', 'TVP448', 'TVP533',
                        'TVT216', 'TVV119', 'TVX854', 'TVX944', 'TWA597', 'TWF574', 'TWF805', 'TWF872', 'TWF903',
                        'TWP190', 'TWP263', 'TWS690', 'TWU403', 'TWW955', 'TXA251', 'TXB321', 'TXB481', 'TXD945',
                        'TXE556', 'TXE586', 'TXE606', 'TXE608', 'TXE638', 'TXE929', 'TXF506', 'TXF557', 'TXH400',
                        'TXJ158', 'TXJ325', 'TXK447', 'TXL586', 'TXL676', 'TXL716', 'TXM959', 'TXN143', 'TXP736',
                        'TXR594', 'TXR303', 'TXR701', 'TXR724', 'TXR802', 'TXR979', 'TXR998', 'TXT748', 'TXT907',
                        'TXU152', 'TXU669', 'TXV938', 'TXW135', 'TXW143', 'TXW492', 'TXW930', 'TXW940', 'TXW950',
                        'TXX518', 'TXZ112', 'TXZ215', 'TXZ548', 'TYA551', 'TYD168', 'TYD472', 'TYD512', 'TYD634',
                        'TYD742', 'TYD981', 'TYD991', 'TYF507', 'TYG560', 'TYG190', 'TYG304', 'TYG462', 'TYG662',
                        'TYG958', 'TYJ207', 'TYJ217', 'TYJ404', 'TYJ416', 'TYJ493', 'TYJ812', 'TYK790', 'TYK931',
                        'TYP663', 'TYP932', 'TYP934', 'TYR103', 'TYR113', 'TYR361', 'TYU145', 'TYU207', 'TYU236',
                        'TYU971', 'TYV547', 'TYW809', 'TYX861', 'TYZ500', 'TYZ731', 'TYZ741', 'TYZ743', 'U0C061',
                        'U1J511', 'U1Y798', 'U2U854', 'U3B247', 'U3Y253', 'U3Y261', 'U3Y495', 'U6B309', 'U8J1267',
                        'U8J376', 'U8J378', 'U8J383', 'U8J390', 'U8J3933', 'U8J394', 'U8J399', 'U8J576', 'U8J930',
                        'U9Y627', 'UAI250', 'UAI586', 'UAI719', 'UAI906', 'UAI912', 'UAI940', 'UAQ651', 'UAQ684',
                        'UAQ733', 'UDS713', 'UDU854', 'UEW943', 'UGZ306', 'UGZ694', 'UIB829', 'UIB365', 'UIB536',
                        'UIB685', 'UIB794', 'UIB839', 'UIJ511', 'UIJ934', 'UIM887', 'UIQ228', 'UIV509', 'UIW972',
                        'UIW171', 'UIW509', 'UIW962', 'UIW952', 'UKI150', 'UKI550', 'UKN224', 'UKT820', 'ULQ830',
                        'UMA161', 'UMD485', 'UMD725', 'UMF145', 'UMW846', 'UNE611', 'UNE641', 'UNE661', 'UOC073',
                        'UOC094', 'UOK856', 'UOM142', 'UON546', 'UOR453', 'UOR460', 'UOS700', 'UOS772', 'UOX110',
                        'UOX649', 'UOX681', 'UOX725', 'UOX771', 'UOX859', 'UPE203', 'UPE486', 'UQA365', 'UQK705',
                        'UQK796', 'UQK836', 'UQK846', 'UQK856', 'UQK948', 'UQQ144', 'UQQ174', 'UQY240', 'UQY712',
                        'UQY713', 'URM861', 'URQ409', 'URQ419', 'URQ447', 'URQ497', 'URQ537', 'URS954', 'UST300',
                        'USZ400', 'UTI156', 'UTI296', 'UTI414', 'UTI839', 'UTI849', 'UUM147', 'UUM165', 'UUQ133',
                        'UUQ332', 'UUQ471', 'UUQ733', 'UUQ920', 'UVP565', 'UVA758', 'UVA855', 'UVB185', 'UVB336',
                        'UVB346', 'UVB426', 'UVB436', 'UVB780', 'UVB963', 'UVD910', 'UVD921', 'UVE426', 'UVE541',
                        'UVF318', 'UVF391', 'UVF896', 'UVG429', 'UVH685', 'UVL478', 'UVL938', 'UVM107', 'UVM411',
                        'UVM770', 'UVP391', 'UVP562', 'UVQ905', 'UVQ907', 'UVS313', 'UVS646', 'UVS697', 'UVS720',
                        'UVT391', 'UVT431', 'UVT448', 'UVT542', 'UVT560', 'UVT583', 'UVT800', 'UVU906', 'UVU114',
                        'UVU137', 'UVU144', 'UVU157', 'UVU295', 'UVU496', 'UVU697', 'UVU717', 'UVU847', 'UVU896',
                        'UVU916', 'UVU921', 'UVU986', 'UVX423', 'UVY314', 'UVY334', 'UVZ517', 'UVZ946', 'UWA102',
                        'UWA112', 'UWE637', 'UWE208', 'UWE521', 'UWE658', 'UWE717', 'UWE835', 'UWE845', 'UWI632',
                        'UWJ457', 'UWJ585', 'UWK217', 'UWK253', 'UWK275', 'UWK285', 'UWK363', 'UWK371', 'UWK427',
                        'UWK437', 'UWK504', 'UWK594', 'UWM756', 'UWM855', 'UWN567', 'UYA207', 'UYA267', 'UYA286',
                        'UYA312', 'UYA409', 'UYA461', 'UYB288', 'UYB327', 'UYB352', 'UYB366', 'UYC506', 'UYC508',
                        'UYC556', 'UYC651', 'UYC804', 'UYC805', 'UYE179', 'UYE272', 'UYE684', 'VEG850', 'WAP634',
                        'WAQ166', 'WAQ191', 'WAQ320', 'WAQ326', 'WAQ327', 'WAQ330', 'WBE9467', 'WBQ380', 'WCE209',
                        'WCJ645', 'WDD2209', 'WDP300', 'WEE696', 'WEI235', 'WEI249', 'WEI269', 'WEI325', 'WEI450',
                        'WEI576', 'WEI598', 'WEI676', 'WEI834', 'WEI844', 'WEQ241', 'WEV597', 'WFQ652', 'WGE285',
                        'WIE310', 'WIE518', 'WIE539', 'WIE852', 'WIE854', 'WIG269', 'WIG385', 'WIG563', 'WIG802',
                        'WIG812', 'WIS272', 'WIS273', 'WIS274', 'WIS293', 'WIS415', 'WIS648', 'WIU972', 'WIV264',
                        'WIV323', 'WIV543', 'WIV553', 'WIV897', 'WIW323', 'WIW320', 'WIW321', 'WIW347', 'WIW349',
                        'WIW350', 'WIW351', 'WIW366', 'WIW459', 'WIW461', 'WIW462', 'WJD459', 'WJM745', 'WJR969',
                        'WKC360', 'WKC987', 'WKQ600', 'WKQ762', 'WLM769', 'WLQ168', 'WLQ418', 'WNP422', 'WOD764',
                        'WOD768', 'WOE361', 'WOE365', 'WOE631', 'WOS772', 'WOV902', 'WOV922', 'WOW764', 'WOX947',
                        'WOY342', 'WOY824', 'WOZ301', 'WPU140', 'WQB708', 'WQB823', 'WQC106', 'WQJ880', 'WQK762',
                        'WQW125', 'WQW137', 'WQW154', 'WQW161', 'WQW183', 'WRI155', 'WSM902', 'WTQ733', 'WUI745',
                        'WUI774', 'WUQ901', 'WVO0221', 'WVO135', 'WVO176', 'WVO221', 'WVQ118', 'WVQ162', 'WVQ812',
                        'XCD638', 'XCN972', 'XDR673', 'XFN890', 'XFP966', 'XHB101', 'XHD615', 'XJC991', 'XJE588',
                        'XJE991', 'XJM504', 'XKG586', 'XKM271', 'XLB814', 'XLG425', 'XM1305', 'XM1310', 'XM1319',
                        'XM1320', 'XM1321', 'XM1322', 'XM1323', 'XM1327', 'XMA992', 'XNV494', 'ZCY882', 'ZEB700',
                        'ZES537', 'ZF5322', 'ZF5332', 'ZF5584', 'ZF6206', 'ZF6277', 'ZF6423', 'ZF6600', 'ZF7009',
                        'ZF7011', 'ZF7013', 'ZF7197', 'ZF8826', 'ZF8827', 'ZF9427', 'ZF9496', 'ZF9893', 'ZFF833',
                        'ZGE819', 'ZH0743', 'ZHR139', 'ZJT365', 'ZJV506', 'ZKM797', 'ZKT276', 'ZKT306', 'ZLB399',
                        'ZNA942', 'ZNJ404', 'ZNK153', 'ZPL170', 'ZPL210', 'ZPL260', 'ZPL270',
                        'ZRL942', 'ZTG771', 'ZTG625', 'ZTG711', 'ZTG815', 'ZTG835', 'ZTG956', 'ZTG994', 'ZTU312',
                        'ZTU943', 'ZTU973', 'ZW2528', 'ZW2529']

        closest_matching = get_close_matches(lp_num, lp_candidate, matching_case, threshold)
        if len(closest_matching) == 0:
            return lp_num
        return closest_matching[0]

    def vote_histogram(self, Truck):
        ### LP
        if len(Truck.lp_candidates) > 0:
            ### determine the final LP ID.
            LicensePlate, LP_Rate = Truck.vote_lp_histogram()
            license_id = ""
            for i in LicensePlate:
                if i is not None:
                    license_id += i

            # Matching start
            license_reconfirm_id = self.lpNumReconfirm(license_id)
            license_reconfirm_id2 = self.correctLP(license_reconfirm_id)
            print('LICENSE_ID: ', license_id, ' / ', 'RECONFIRM_ID: ', license_reconfirm_id2)

            Truck.LicenseID = [license_reconfirm_id2, LP_Rate]

        else:
            # Truck.LicenseID = ["-1", -1]
            Truck.LicenseID = ["RECOG_FAIL", 0]

        ### CID
        if len(Truck.cp_candidates) > 0 or len(Truck.cp_candidates2) > 0:
            if Truck.is_twin_truck is False:

                ### determine the final 1st container ID.
                CP_result, CP_Rate = Truck.vote_cp_histogram2(is_second=False)
                container_id = ""
                for i in CP_result:
                    if i is not None:
                        container_id += str(i)
                if len(container_id) == 0:
                    Truck.containerID = ['RECOG_FAIL', 0]
                    Truck.containerID2 = []
                else:
                    #Filtering CP number
                    container_id = self.correctCP(container_id)
                    Truck.containerID = [container_id, CP_Rate]
                    Truck.containerID2 = []

            else:
                ### determine the final 1st, 2nd container ID.
                CP_result1, CP_Rate1 = Truck.vote_cp_histogram2(is_second=False)
                container_id = ""
                if CP_result1 is not None:
                    for i in CP_result1:
                        if i is not None:
                            container_id += str(i)
                if len(container_id) == 0:
                    Truck.containerID = ['RECOG_FAIL', 0]
                else:
                    # Filtering CP number
                    container_id = self.correctCP(container_id)
                    Truck.containerID = [container_id, CP_Rate1]

                CP_result2, CP_Rate2 = Truck.vote_cp_histogram2(is_second=True)
                container_id2 = ""
                if CP_result2 is not None:
                    for i in CP_result2:
                        if i is not None:
                            container_id2 += str(i)
                if len(container_id) == 0:
                    Truck.containerID2 = ['RECOG_FAIL', 0]
                else:
                    # Filtering CP number
                    container_id2 = self.correctCP(container_id2)
                    Truck.containerID2 = [container_id2, CP_Rate2]

                self.isit_real_twin(Truck, container_id, container_id2)

    def save_info2file(self, Truck, output_file, delimeter):
        output_file.write(str(Truck.truck_cnt) + delimeter)

        ### LP
        for char in str(Truck.LicenseID[0]):
            if char is not None:
                output_file.write(char)
        output_file.write(delimeter)

        ### CID
        if len(Truck.cp_candidates) > 0 or len(Truck.cp_candidates2) > 0:
            if Truck.is_twin_truck is False:
                for char in str(Truck.containerID[0]):
                    output_file.write(char)
                output_file.write(delimeter)

            else:
                for char in str(Truck.containerID[0]):
                    output_file.write(char)
                output_file.write(delimeter)

                for char in str(Truck.containerID2[0]):
                    output_file.write(char)
                output_file.write(delimeter)


        else:
            output_file.write(delimeter)

        output_file.write(str(Truck.is_40ft_truck))
        output_file.write(delimeter)
        output_file.write(str(Truck.is_correct_DD))
        output_file.write(delimeter)
        output_file.write(str(Truck.Truck_Chassis_pos))
        output_file.write(delimeter)
        output_file.write('\n')

    def isit_real_twin(self, truck, cp1, cp2):
        if cp1 == cp2:
            truck.is_twin_truck = False
            truck.containerID2 = None

    ### make dict type data template using json data ###
    def load_config_json(self, dir):
        json_data = open(dir, encoding='utf-8').read()
        data_template = json.loads(json_data)

        TruckArrivedObjectItem_url = data_template['ServerIp'] + ":" + data_template['ServerPort'] + "/" + \
                                     data_template['ServerPrefixUrl'] + "/" + data_template['LaneNumber'] + "/" + \
                                     data_template['TruckArrivedSubAddr']
        AIToGosSendProperty_url = data_template['ServerIp'] + ":" + data_template['ServerPort'] + "/" + data_template[
            'ServerPrefixUrl'] + "/" + data_template['LaneNumber'] + "/" + data_template['RecognitionSubAddr']
        ImageStitched_url = data_template['ServerIp'] + ":" + data_template['ServerPort'] + "/" + data_template[
            'ServerPrefixUrl'] + "/" + data_template['LaneNumber'] + "/" + data_template['ImageStitchedAddr']
        DeviceStatus_url = data_template['ServerIp'] + ":" + data_template['ServerPort'] + "/" + data_template[
            'ServerPrefixUrl'] + "/" + data_template['LaneNumber'] + "/" + data_template['DeviceStatusAddr']

        laneType = data_template['LaneType']  # In-Lane
        lane = data_template['LaneNumber']  # L1

        FTP_server = data_template['FTP_Ip']
        FTP_user = data_template['FTP_user']
        FTP_password = data_template['FTP_pwd']
        FTP_port = int(data_template['FTP_port'])
        fileserver_dir = data_template['fileserver_dir']

        return TruckArrivedObjectItem_url, AIToGosSendProperty_url, ImageStitched_url, DeviceStatus_url, laneType, lane, FTP_server, FTP_user, FTP_password, FTP_port, fileserver_dir

    # Notify for hardware(GPU, CAM) status.
    def set_template_NotiDeviceStatusItem(self):
        NotiDeviceStatusItem = {
            "LaneID": "",
            "MessageID": "NotiDeviceStatus",
            "Body": {
                "DeviceType": "",
                "DeviceKey": "",
                "DeviceErrorCode": "",
                "DeviceStatus": ""

            }
        }
        return NotiDeviceStatusItem

    def set_template_TruckArrivedObjectItem(self):
        TruckArrivedObjectItem = {

            "LaneID": "",
            "MessageID": "TruckArrived",
            "Body": {}
        }
        return TruckArrivedObjectItem

    def set_template_BoomBarrierCloseObjectItem(self):
        BoomBarrierCloseObjectItem = {

            "LaneID": "",
            "MessageID": "BoomBarrierClose",
            "Body": {}
        }
        return BoomBarrierCloseObjectItem

    def set_template_RecognitionObjectItem(self):
        RecognitionObjectItem = {

            "LaneID": "",
            "MessageID": "LPDetected",

            "Body": {
                "LPNumber": "",
                "LPNumberPR": "",
                "LPImageUrl": "",
                "ChassisLength": "",
                "Containers": [
                    {
                        "ContainerNumber": "",
                        "ContainerNumberPR": "",
                        "ContainerISO": "",
                        "ContainerDoorDirection": "A",
                        "ContainerPosition": "",
                        "ContainerImageUrl": {
                            "Rear": "",
                            "Top": "",
                            "Left": "",
                            "Right": ""
                        }
                    },
                    {
                        "ContainerNumber": "",
                        "ContainerNumberPR": "",
                        "ContainerISO": "",
                        "ContainerDoorDirection": "A",
                        "ContainerPosition": "",
                        "ContainerImageUrl": {
                            "Rear": "",
                            "Top": "",
                            "Left": "",
                            "Right": ""
                        }
                    }
                ]
            }
        }
        return RecognitionObjectItem

    def set_template_StitchingObjectItem(self):
        StitchingObjectItem = {

            "LaneID": "",
            "MessageID": "ImageStitched",

            "Body": {
                "Containers": [
                    {
                        "ContainerNumber": "",
                        "StitchedImageUrl": {
                            "Rear": "",
                            "Top": "",
                            "Left": "",
                            "Right": ""
                        }
                    },
                    {
                        "ContainerNumber": "",
                        "StitchedImageUrl": {
                            "Rear": "",
                            "Top": "",
                            "Left": "",
                            "Right": ""
                        }
                    }
                ]
            }
        }
        return StitchingObjectItem

    def saveAsJSON_VAS_stitching(self, template_, truck, Date, Time):
        ### date, time ###
        template_['Date'] = Date
        template_['Time'] = Time
        template_['LaneType'] = self.lane_type  # In-Lane
        template_['LaneID'] = self.lane  # L1

        LPNumber = truck.LicenseID[0]
        if truck.is_twin_truck:  # twin
            template_['Body']['Containers'][0]['ContainerNumber'] = truck.containerID[0][:11]
            template_['Body']['Containers'][0]['StitchedImageUrl']['Rear'] = ""
            template_['Body']['Containers'][0]['StitchedImageUrl']['Top'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/12_Fore_Top.jpg"
            template_['Body']['Containers'][0]['StitchedImageUrl']['Left'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/13_Fore_Left.jpg"
            template_['Body']['Containers'][0]['StitchedImageUrl']['Right'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/14_Fore_Right.jpg"

            template_['Body']['Containers'][1]['ContainerNumber'] = truck.containerID2[0][:11]
            template_['Body']['Containers'][1]['StitchedImageUrl']['Rear'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/15_After_Rear.jpg"
            template_['Body']['Containers'][1]['StitchedImageUrl']['Top'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/16_After_Top.jpg"
            template_['Body']['Containers'][1]['StitchedImageUrl']['Left'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/17_After_Left.jpg"
            template_['Body']['Containers'][1]['StitchedImageUrl']['Right'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/18_After_Right.jpg"

        else:  # single
            del template_['Body']['Containers'][1]
            template_['Body']['Containers'][0]['ContainerNumber'] = truck.containerID[0][:11]
            template_['Body']['Containers'][0]['StitchedImageUrl']['Rear'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/11_Fore_Rear.jpg"
            template_['Body']['Containers'][0]['StitchedImageUrl']['Top'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/12_Fore_Top.jpg"
            template_['Body']['Containers'][0]['StitchedImageUrl']['Left'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/13_Fore_Left.jpg"
            template_['Body']['Containers'][0]['StitchedImageUrl']['Right'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/14_Fore_Right.jpg"

        return template_

    ### VAS detect and save data
    def saveAsJSON_VAS(self, template_, truck, Date, Time):

        ### Date ###
        template_['Date'] = Date
        template_['Time'] = Time
        template_['LaneType'] = self.lane_type  # In-Lane
        template_['LaneID'] = self.lane  # L1

        ### Oher info Chassis length ###
        if truck.is_40ft_truck:
            template_['Body']['ChassisLength'] = "40"
        elif truck.is_correct_DD == None:
            template_['Body']['ChassisLength'] = ""
        else:
            template_['Body']['ChassisLength'] = "20"

        ### LP ###
        LPNumber = truck.LicenseID[0]
        LPNumberPR = truck.LicenseID[1]
        if len(truck.LicenseID) != 0:
            template_['Body']['LPNumber'] = LPNumber
            template_['Body']['LPNumberPR'] = LPNumberPR
            template_['Body']['LPImageUrl'] = self.remote_path + "/" + template_[
                'Time'] + "_" + LPNumber + "/02_20ftTwin_LP.jpg"
        else:
            template_['Body']['LPNumber'] = ""
            template_['Body']['LPNumberPR'] = ""
            template_['Body']['LPImageUrl'] = self.remote_path + "/" + "02_20ftTwin_LP.jpg"

        ### CP ###
        if truck.Truck_Chassis_pos != "E":
            if truck.is_twin_truck:  # twin
                if len(truck.containerID) != 0:
                    print('Truck.containerID', truck.containerID)
                    template_['Body']['Containers'][0]['ContainerNumber'] = truck.containerID[0][:11]
                    template_['Body']['Containers'][0]['ContainerNumberPR'] = truck.containerID[1]
                    template_['Body']['Containers'][0]['ContainerISO'] = truck.containerID[0][11:]
                    template_['Body']['Containers'][0]['ContainerPosition'] = "F"

                    template_['Body']['Containers'][0]['ContainerImageUrl']['Rear'] = ""
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Top'] = self.remote_path + "/" + template_[
                        'Time'] + "_" + LPNumber + "/04_20ftFore_Top.jpg"
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Left'] = self.remote_path + "/" + \
                                                                                      template_[
                                                                                          'Time'] + "_" + LPNumber + "/05_20ftFore_Left.jpg"
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Right'] = self.remote_path + "/" + \
                                                                                       template_[
                                                                                           'Time'] + "_" + LPNumber + "/06_20ftFore_Right.jpg"
                    template_['Body']['Containers'][1]['ContainerImageUrl']['Rear'] = self.remote_path + "/" + \
                                                                                      template_[
                                                                                          'Time'] + "_" + LPNumber + "/07_20ftAfter_Rear.jpg"
                else:
                    template_['Body']['Containers'][0]['ContainerNumber'] = ""
                    template_['Body']['Containers'][0]['ContainerNumberPR'] = ""
                    template_['Body']['Containers'][0]['ContainerISO'] = ""
                    template_['Body']['Containers'][0]['ContainerPosition'] = ""

                    template_['Body']['Containers'][0]['ContainerImageUrl']['Rear'] = ""
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Top'] = ""
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Left'] = ""
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Right'] = ""
                    template_['Body']['Containers'][1]['ContainerImageUrl']['Rear'] = ""

                if len(truck.containerID2) != 0:
                    template_['Body']['Containers'][1]['ContainerNumber'] = truck.containerID2[0][:11]
                    template_['Body']['Containers'][1]['ContainerNumberPR'] = truck.containerID2[1]
                    template_['Body']['Containers'][1]['ContainerISO'] = truck.containerID2[0][11:]
                    template_['Body']['Containers'][1]['ContainerPosition'] = "A"

                    template_['Body']['Containers'][1]['ContainerImageUrl']['Top'] = self.remote_path + "/" + template_[
                        'Time'] + "_" + LPNumber + "/08_20ftAfter_Top.jpg"
                    template_['Body']['Containers'][1]['ContainerImageUrl']['Left'] = self.remote_path + "/" + \
                                                                                      template_[
                                                                                          'Time'] + "_" + LPNumber + "/09_20ftAfter_Left.jpg"
                    template_['Body']['Containers'][1]['ContainerImageUrl']['Right'] = self.remote_path + "/" + \
                                                                                       template_[
                                                                                           'Time'] + "_" + LPNumber + "/10_20ftAfter_Right.jpg"
                else:
                    template_['Body']['Containers'][1]['ContainerNumber'] = ""
                    template_['Body']['Containers'][1]['ContainerNumberPR'] = ""
                    template_['Body']['Containers'][1]['ContainerISO'] = ""
                    template_['Body']['Containers'][1]['ContainerPosition'] = ""

                    template_['Body']['Containers'][1]['ContainerImageUrl']['Top'] = ""
                    template_['Body']['Containers'][1]['ContainerImageUrl']['Left'] = ""
                    template_['Body']['Containers'][1]['ContainerImageUrl']['Right'] = ""

                # Door Direction
                if truck.is_correct_DD == True:
                    template_['Body']['Containers'][1]['ContainerDoorDirection'] = "A"
                elif truck.is_correct_DD == None:
                    template_['Body']['Containers'][1]['ContainerDoorDirection'] = "A"
                else:
                    template_['Body']['Containers'][1]['ContainerDoorDirection'] = "F"



            else:  # single
                del template_['Body']['Containers'][1]
                if len(truck.containerID) != 0:
                    template_['Body']['Containers'][0]['ContainerNumber'] = truck.containerID[0][:11]
                    template_['Body']['Containers'][0]['ContainerNumberPR'] = truck.containerID[1]
                    template_['Body']['Containers'][0]['ContainerISO'] = truck.containerID[0][11:]
                    template_['Body']['Containers'][0]['ContainerPosition'] = truck.Truck_Chassis_pos

                    template_['Body']['Containers'][0]['ContainerImageUrl']['Rear'] = self.remote_path + "/" + \
                                                                                      template_[
                                                                                          'Time'] + "_" + LPNumber + "/03_20ftAfter_Rear.jpg"
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Top'] = self.remote_path + "/" + template_[
                        'Time'] + "_" + LPNumber + "/04_20ftFore_Top.jpg"
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Left'] = self.remote_path + "/" + \
                                                                                      template_[
                                                                                          'Time'] + "_" + LPNumber + "/05_20ftFore_Left.jpg"
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Right'] = self.remote_path + "/" + \
                                                                                       template_[
                                                                                           'Time'] + "_" + LPNumber + "/06_20ftFore_Right.jpg"


                else:
                    template_['Body']['Containers'][0]['ContainerNumber'] = ""
                    template_['Body']['Containers'][0]['ContainerNumberPR'] = ""
                    template_['Body']['Containers'][0]['ContainerISO'] = ""
                    template_['Body']['Containers'][0]['ContainerPosition'] = ""

                    template_['Body']['Containers'][0]['ContainerImageUrl']['Rear'] = ""
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Top'] = ""
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Left'] = ""
                    template_['Body']['Containers'][0]['ContainerImageUrl']['Right'] = ""

                # Door Direction
                if truck.is_correct_DD == True:
                    template_['Body']['Containers'][0]['ContainerDoorDirection'] = "A"
                elif truck.is_correct_DD == None:
                    template_['Body']['Containers'][0]['ContainerDoorDirection'] = "A"
                else:
                    template_['Body']['Containers'][0]['ContainerDoorDirection'] = "F"

        else:
            del template_['Body']['Containers'][1]
            del template_['Body']['Containers'][0]

        return template_

    def folder_empty(self, fileserver_dir, start_folder, lane_idx):

        path_VAS = is_folder(fileserver_dir, start_folder)
        path_year = is_folder(path_VAS, str(datetime.now().year))
        path_month = is_folder(path_year, str('{0:02d}'.format(datetime.now().month)))
        path_day = is_folder(path_month, str('{0:02d}'.format(datetime.now().day)))
        path_lane = is_folder(path_day, str(lane_idx))
        final_path = path_lane

        return final_path


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def ftp_is_folder(path, name, ftp):
    ftp.cwd(path)
    if name in ftp.nlst():
        ftp.cwd(name)
    else:
        ftp.mkd(name)
        ftp.cwd(name)

    return path + "/" + name


def is_folder(path, name):
    directory = path + "/" + name
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def ftp_folder_empty(fileserver_dir, start_folder, lane_idx, recog_info, ftp):
    path_VAS = ftp_is_folder(fileserver_dir, start_folder, ftp)
    path_year = ftp_is_folder(path_VAS, str(datetime.now().year), ftp)
    path_month = ftp_is_folder(path_year, str('{0:02d}'.format(datetime.now().month)), ftp)
    path_day = ftp_is_folder(path_month, str('{0:02d}'.format(datetime.now().day)), ftp)
    path_lane = ftp_is_folder(path_day, str("L" + lane_idx), ftp)
    cardir = recog_info['Body']['LPNumber']
    path_car = ftp_is_folder(path_lane, cardir, ftp)

    return path_car


def write_frames(writer, frame_set):
    for idx in range(len(frame_set)):
        writer.write(frame_set[idx])
