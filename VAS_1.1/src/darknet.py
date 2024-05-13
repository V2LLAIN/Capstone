#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"
- Set environment variable "DARKNET_PATH" to path darknet lib .so (for Linux)

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
from ctypes import *
import math
import random
import os
import cv2
import numpy as np


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)
    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))

def point_cvt(detections, w_scale, h_scale):
    pts = []

    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)

        left = int(w_scale * left)
        right = int(w_scale * right)
        top = int(h_scale * top)
        bottom = int(h_scale * bottom)

        pt = [left, top, right, bottom]
        pts.append(pt)

    return pts



def draw_boxes(detections, image, colors, pts, Key = True):

    idx = 0
    image_ = image.copy()
    for label, confidence, bbox in detections:

        left, top, right, bottom = pts[idx]

        if Key == True:
            cv2.rectangle(image_, (left, top), (right, bottom), colors, 3)
            cv2.putText(image_, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors, 2)
            cp_x = int( left +  (right - left) / 2 )
            cp_y = int( top + (bottom - top) / 2 )
            cv2.circle(image_, (cp_x, cp_y), 10, colors, -1)
            
        idx = idx + 1

    return image_

def draw_boxes_switches(detections, switchs, image, pts, Key = True):

    colors = (255, 255, 255)
    color_B = (255, 102, 51)
    color_G = (51, 255, 102)
    color_R = (51, 51, 255)


    idx = 0
    image_ = image.copy()
    for label, confidence, bbox in detections:

        left, top, right, bottom = pts[idx]

        if switchs[idx] == False: colors = color_G
        elif switchs[idx] == True: colors = color_R
            

        if Key == True:
            cv2.rectangle(image_, (left, top), (right, bottom), colors, 2)
            cv2.putText(image_, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors, 2)
            cp_x = int( left +  (right - left) / 2 )
            cp_y = int( top + (bottom - top) / 2 )
            # cv2.circle(image_, (cp_x, cp_y), 10, colors, -1)
            
        idx = idx + 1



    return image_



def isGap(detections):
    cont_back_flag = False
    cont_center_flag = False
    for det in detections:
        labelId = str(det[0])
        if( labelId == 'Cont_center' ):
            cont_center_flag = True
        elif ( labelId == 'Cont_back'):
            cont_back_flag = True

    if cont_center_flag == True and cont_back_flag == False:
        return True
    return False



def im_trim (img, box): 

    #x, y, w, h = box 
    l, t, r, b = box 
    if l < 0 or t < 0 or r < 0 or b < 0:
        return None
    img_trim = img[int(t):int(b), int(l):int(r)] 

    # cv2.imshow('img_trim', img_trim)
    # cv2.waitKey(0)
    return img_trim

def LP_rule(LP_class, LP_data):

    canditate_idxes = []
    remain_idxes = []
    remove_idxes = []
    for idx in range(len(LP_data)):
        max_idx, selected_LP_idx = select_LP_data(LP_data[idx], idx, LP_data)
        remain_idxes.append(max_idx)
        if len(selected_LP_idx) > 1:
            canditate_idxes.extend(selected_LP_idx)

    remove_idxes = list(set(canditate_idxes))
    remain_idxes = list(set(remain_idxes))
    remain_idxes.remove(-1)

    for idx in range(len(remain_idxes)):
        remove_idxes.remove(remain_idxes[idx])

    for idx_remove in range(len(remove_idxes)):
        del LP_data[remove_idxes[idx_remove]]

    recog_result = []
    ruled_LP_data = LP_data


    # print('#: ', len(ruled_LP_data), ', ', LP_class, ':')
    if LP_class == 'LP_1' and len(ruled_LP_data) >= 8:
        ruled_LP_data.sort(key=lambda ruled_LP_data : ruled_LP_data[2][1])
        

        up_LP = ruled_LP_data[:3]
        up_LP.sort(key=lambda up_LP : up_LP[2][0])
        for idx_l in range(len(up_LP)):
            recog_result.append(str(up_LP[idx_l][0]))
            # recog_result.append(str(up_LP[idx_l][0].decode('utf-8')))


        down_LP = ruled_LP_data[3:]
        down_LP.sort(key=lambda down_LP : down_LP[2][0])
        for idx_l in range(len(down_LP)):
            recog_result.append(str(down_LP[idx_l][0]))
            # recog_result.append(str(down_LP[idx_l][0].decode('utf-8')))

    elif LP_class == 'LP_2'and len(ruled_LP_data) >= 8:
        ruled_LP_data.sort(key=lambda ruled_LP_data : ruled_LP_data[2][0])
        for idx_l in range(len(ruled_LP_data)):
            recog_result.append(str(ruled_LP_data[idx_l][0]))
            # recog_result.append(str(ruled_LP_data[idx_l][0].decode('utf-8')))


    else:
        print('else, ', len(ruled_LP_data)) 

    return recog_result




def select_LP_data(cur_LPd, cur_idx, LP_data):

    max_idx = -1
    dist_thresh = 3
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


def detect_NN(img, net, meta, thres):

    width = network_width(net)
    height = network_height(net)

    # print(img.shape, " chan: ", len(img.shape))

    if len(img.shape) == 2:
        darknet_image = make_image(width, height, 1)
        frame_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    else:
        darknet_image = make_image(width, height, 3)
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    detections = detect_image(net, meta, darknet_image, thres)
    free_image(darknet_image)
    return detections, frame_resized


def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions


def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
        Returns a list with highest confidence class and their bbox
    """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])

lib = CDLL("/home/admin/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("C:\\Users\\DPW\\Desktop\\NN_python\\LP\\yolo_cpp_dll.dll", RTLD_GLOBAL)
#  lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
# if os.name == "nt":
#     cwd = os.path.dirname(__file__)
#     os.environ['PATH'] = cwd + ';' + os.environ['PATH']
#     winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
#     winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
#     envKeys = list()
#     for k, v in os.environ.items():
#         envKeys.append(k)
#     try:
#         try:
#             tmp = os.environ["FORCE_CPU"].lower()
#             if tmp in ["1", "true", "yes", "on"]:
#                 raise ValueError("ForceCPU")
#             else:
#                 print("Flag value {} not forcing CPU mode".format(tmp))
#         except KeyError:
#             # We never set the flag
#             if 'CUDA_VISIBLE_DEVICES' in envKeys:
#                 if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
#                     raise ValueError("ForceCPU")
#             try:
#                 global DARKNET_FORCE_CPU
#                 if DARKNET_FORCE_CPU:
#                     raise ValueError("ForceCPU")
#             except NameError as cpu_error:
#                 print(cpu_error)
#         if not os.path.exists(winGPUdll):
#             raise ValueError("NoDLL")
#         lib = CDLL(winGPUdll, RTLD_GLOBAL)
#     except (KeyError, ValueError):
#         hasGPU = False
#         if os.path.exists(winNoGPUdll):
#             lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
#             print("Notice: CPU-only mode")
#         else:
#             # Try the other way, in case no_gpu was compile but not renamed
#             lib = CDLL(winGPUdll, RTLD_GLOBAL)
#             print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
# else:
#     lib = CDLL(os.path.join(
#         os.environ.get('DARKNET_PATH', './'),
#         "libdarknet.so"), RTLD_GLOBAL)

altNames = None
def performDetect(img_opencv, network, metadata, thresh= 0.25, initOnly= False):
    """
    Convenience function to handle the detection and returns of objects.

    Displaying bounding boxes 

    Parameters
    ----------------
    imagePath: str
        Path to the image to evaluate. Raises ValueError if not found

    thresh: float (default= 0.25)
        The detection threshold

    initOnly: bool (default= False)
        Only initialize globals. Don't actually run a prediction.

    Returns
    ----------------------


    Output list of tuples with
        {
            "detections": as above
            "image": a numpy array representing an image, compatible with scikit-image
            "caption": an image caption
        }
    """
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"


    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if initOnly:
        print("Initialized detector")
        return None
  
    ### convert opencv2array
    img_width = img_opencv.shape[1]
    img_height = img_opencv.shape[0]
    
    #print(img_opencv.shape) 
    custom_image = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image,(lib.network_width(network), lib.network_height(network)), interpolation = cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)	

    ### Do the detection
    detections = detect(im, img_width, img_height, network, metadata, thresh)	# if is used cv2.imread(image)
    # print(detections)


    return detections

def detect(opencv_image, img_width, img_height, net, meta, thresh=.5, hier_thresh=.5, nms=.45, debug= False):

    ret = detect_image_p(img_width, img_height, net, meta, opencv_image, thresh, hier_thresh, nms, debug)
    return ret 

def detect_image_p(img_width, img_height, net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug= False):

    
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    letter_box = 0
    #predict_image_letterbox(net, im)
    #letter_box = 1

    ### Do the yolo~ using network
    if debug: print("did prediction")
    dets = get_network_boxes(net, img_width, img_height, thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV img

    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res

def convert_box_value(r): 
    """ 
    This method calculate box information of detected objects. 
    
    :input param: detected object's info.
    :return: boxes info. of detected object
    """

    boxes = [] 

    for k in range(len(r)):
        width = r[k][2][2] 
        height = r[k][2][3] 
        center_x = r[k][2][0] 
        center_y = r[k][2][1] 
        bottomLeft_x = center_x - (width / 2) 
        bottomLeft_y = center_y - (height / 2)
        if bottomLeft_x < 0 :
            bottomLeft_x = 0
        if bottomLeft_y < 0 :
            bottomLeft_y =0
        x, y, w, h = bottomLeft_x, bottomLeft_y, width, height 

        boxes.append((x, y, w, h)) 

    return boxes
    

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)

