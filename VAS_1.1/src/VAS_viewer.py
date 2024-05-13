import numpy as np
import cv2
import copy
import time

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

class VAS_viewer:

    def __new__(self, w,h):
        if not hasattr(self, 'instance'):
            self.instance = super(VAS_viewer, self).__new__(self)

        self.view_size = (w, h)
        self.F_img = np.zeros((h, w, 3), dtype = "uint8")
        self.Lu_img = np.zeros((h, w, 3), dtype = "uint8")
        self.Ld_img = np.zeros((h, w, 3), dtype = "uint8")
        self.Ru_img = np.zeros((h, w, 3), dtype = "uint8")
        self.Rd_img = np.zeros((h, w, 3), dtype = "uint8")
        self.Tr_img = np.zeros((h, w, 3), dtype = "uint8")

        return self.instance


    def set_view_position(self, w1, w2, w3, h1, h2):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.h1 = h1
        self.h2 = h2


    def set_frames(self, F_img, Ru_img, Lu_img, Tr_img):
        self.F_img = copy.deepcopy(F_img)
        self.Lu_img = copy.deepcopy(Lu_img)
        self.Ld_img = copy.deepcopy(Lu_img)
        self.Ru_img = copy.deepcopy(Ru_img)
        self.Rd_img = copy.deepcopy(Ru_img)
        self.Tr_img = copy.deepcopy(Tr_img)


    def show_windows(self, lane_number):

        F_resize = cv2.resize(self.F_img, dsize=self.view_size, interpolation=cv2.INTER_AREA)
        Ru_resize = cv2.resize(self.Ru_img, dsize=self.view_size, interpolation=cv2.INTER_AREA)
        Rd_resize = cv2.resize(self.Rd_img, dsize=self.view_size, interpolation=cv2.INTER_AREA)
        Lu_resize = cv2.resize(self.Lu_img, dsize=self.view_size, interpolation=cv2.INTER_AREA)
        Ld_resize = cv2.resize(self.Ld_img, dsize=self.view_size, interpolation=cv2.INTER_AREA)
        TR_resize = cv2.resize(self.Tr_img, dsize=self.view_size, interpolation=cv2.INTER_AREA)
    
        add_h1 = cv2.hconcat([F_resize, Ru_resize, Lu_resize])
        add_h2 = cv2.hconcat([TR_resize, Rd_resize, Ld_resize])
        add_v = cv2.vconcat([add_h1, add_h2])
        add_v_resize = cv2.resize(add_v, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("[Lane " + lane_number + " System view]", add_v_resize)
    



    def draw_vasSignal(self, state, truck_cnt):
        if state == True:
            cv2.putText(self.F_img, format("State : ON"), (100, self.F_img.shape[0]-150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(self.F_img, format("State : OFF"), (100, self.F_img.shape[0]-150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255,255), 3, cv2.LINE_AA)
            
        cv2.putText(self.F_img, format("Truck # : " + str(truck_cnt)), (100, self.F_img.shape[0]-200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255,255), 3, cv2.LINE_AA)



    def draw_frameIdx(self, frame_num, total_frame_num):
        cv2.putText(self.F_img, format("Frame # : " + str(frame_num+1) + " / " +  str(total_frame_num)), (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255,255), 3, cv2.LINE_AA)

    def draw_fps(self, fpsCheckTime, fpsCheck, avg_time, avg_fps, avg_frame_length):
        cv2.putText(self.F_img, format("(Total Time: " + str(int(round(avg_time * avg_frame_length / 1000))) + " [s]  /  AVG Speed : " + str(avg_time) + " [ms],  " + str(avg_fps) + " [fps])"), (100, self.F_img.shape[0]-50), cv2.FONT_HERSHEY_PLAIN, 3, (230, 230, 230), 3, cv2.LINE_AA)
        if fpsCheck >= 33:
            cv2.putText(self.F_img, format("Speed : " + str(fpsCheckTime) + " [ms],    " + str(fpsCheck) + " [fps]"), (100, self.F_img.shape[0]-100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255,255), 3, cv2.LINE_AA)
        elif fpsCheck >= 15 and fpsCheck < 33:
            cv2.putText(self.F_img, format("Speed : " + str(fpsCheckTime) + " [ms],    " + str(fpsCheck) + " [fps]"), (100, self.F_img.shape[0]-100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,125), 3, cv2.LINE_AA)
        elif fpsCheck < 15:
            cv2.putText(self.F_img, format("Speed : " + str(fpsCheckTime) + " [ms],    " + str(fpsCheck) + " [fps]"), (100, self.F_img.shape[0]-100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0,255), 3, cv2.LINE_AA)


    def draw_processRoi(self, state, LP_, CP_, LP_bottom_border):
        cv2.line(self.Tr_img, (0,CP_.end_from_line), (self.Tr_img.shape[1], CP_.end_from_line), (255,255,255), 5)
        cv2.line(self.Tr_img, (0,CP_.end_to_line), (self.Tr_img.shape[1], CP_.end_to_line), (255,255,255), 5)

        if state == True:
            cv2.rectangle(self.F_img, (LP_.start_rect_center[0] - LP_.rect_lx, LP_.start_rect_center[1] - LP_.rect_ly), (LP_.start_rect_center[0] + LP_.rect_lx, LP_.start_rect_center[1] + LP_.rect_ly), (0,255,255), 3)
            cv2.line(self.F_img, ((LP_.start_rect_center[0] - LP_.rect_lx),(LP_.start_rect_center[1] + LP_.rect_ly -LP_bottom_border)), ((LP_.start_rect_center[0] + LP_.rect_lx),(LP_.start_rect_center[1] + LP_.rect_ly -LP_bottom_border)), (0,200,200), 5)
            cv2.rectangle(self.Tr_img, ((CP_.end_rect_center[0] - CP_.rect_lx) , (CP_.end_rect_center[1] - CP_.rect_ly)), ((CP_.end_rect_center[0] + CP_.rect_lx) , (CP_.end_rect_center[1] + CP_.rect_ly)), (0,255,255), 5)
            cv2.line(self.Tr_img, ((CP_.end_rect_center[0] - CP_.rect_lx),CP_.end_rect_center[1]), ((CP_.end_rect_center[0] + CP_.rect_lx),CP_.end_rect_center[1]), (0,200,200), 5)
        else:
            cv2.rectangle(self.F_img, (LP_.start_rect_center[0] - LP_.rect_lx, LP_.start_rect_center[1] - LP_.rect_ly), (LP_.start_rect_center[0] + LP_.rect_lx, LP_.start_rect_center[1] + LP_.rect_ly), (255,255,255), 3)
            cv2.line(self.F_img, ((LP_.start_rect_center[0] - LP_.rect_lx),(LP_.start_rect_center[1] + LP_.rect_ly -LP_bottom_border)), ((LP_.start_rect_center[0] + LP_.rect_lx),(LP_.start_rect_center[1] + LP_.rect_ly -LP_bottom_border)), (255,255,255), 5)
            cv2.rectangle(self.Tr_img, ((CP_.end_rect_center[0] - CP_.rect_lx) , (CP_.end_rect_center[1] - CP_.rect_ly)), ((CP_.end_rect_center[0] + CP_.rect_lx) , (CP_.end_rect_center[1] + CP_.rect_ly)), (255,255,255), 5)
            cv2.line(self.Tr_img, ((CP_.end_rect_center[0] - CP_.rect_lx),CP_.end_rect_center[1]), ((CP_.end_rect_center[0] + CP_.rect_lx),CP_.end_rect_center[1]), (255,255,255), 5)
        


        
    def draw_boomBarrierState(self, state, CP_):
        if state == False: #open
            cv2.line(self.Tr_img, (0, CP_.boom_barrier_line), (self.Tr_img.shape[1], CP_.boom_barrier_line), (255,255,255), 5)
        else: # close
            cv2.line(self.Tr_img, (0, CP_.boom_barrier_line), (self.Tr_img.shape[1], CP_.boom_barrier_line), (0,255,255), 5)
