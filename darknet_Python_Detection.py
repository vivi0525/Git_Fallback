from ast import Nonlocal
from ctypes import *
import os
from turtle import width
import cv2
import numpy as np
import time
import darknet
import threading

from typing import ByteString

def init():
	global preLeft,preRight,LCheck,RCheck
	Sspeed = [0, 0, 0]
	Croad,Cline = 0, 0
	l_Maxsp = 200
	global lock,first
	lock=threading.Lock()
	#print(cv2.__version__)

############################
def convertBack(x, y, w, h): 
	xmin = int(round(x - (w / 2)))
	xmax = int(round(x + (w / 2)))
	ymin = int(round(y - (h / 2)))
	ymax = int(round(y + (h / 2)))
	return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img): #畫YOLO框
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        
        #畫圖
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(img, (int(x),int(y)), 1, (255,0,0), 2)#
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(float(detection[1]), 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img
#############################

netMain = None
metaMain = None
altNames = None

def GMM():

    global metaMain, netMain, altNames,frame_resized,darknet_road_line,Fgaussian_range,kname,first
    configPath = "./cfg/yolov4-tiny-fall_come_left.cfg" #Your training cfg file
    weightPath = "./yolov4-tiny_parallel_low_best.weights" #weight file
    metaPath = "./cfg/fall_come_left.data" #Data file

#########看文件是否存在，否則return ValueError#############
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
###### 檢查 metaMain, NetMain 和 altNames. Loads it in script#####
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
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
#########看文件是否存在，否則return ValueError#############

    class_names = [metaMain.names[i].decode("ascii") for i in range(metaMain.classes)]
    class_colors = darknet.class_colors(class_names)

    cap = cv2.VideoCapture("bing_nohead_move_red.mov") #讀影片
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    cap.set(3, 640)
    cap.set(4, 480)

    print("Starting the YOLO loop...")
    darknet_image = darknet.make_image(darknet.network_width(netMain),darknet.network_height(netMain),3)
    
    fps = 0.0
    tic = time.time()

    _Keep_Run = True
    while _Keep_Run:#_Keep_Run  True
        ret, frame_read = cap.read()#this code repeat can turbo fps(加速影片播放速度，但疊太多層會太快)
        if(not ret):
            print("RTSP error! Please Check Internert")
            break
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,(darknet.network_width(netMain),darknet.network_height(netMain)),interpolation=cv2.INTER_LINEAR)
        toc = time.time()

        YOLO(frame_resized, darknet_image, class_names, fps)
    
    cap.release()
    cv2.destroyWindow
    print("yolo exist")

def YOLO(ResizeImg, DarkImg, ClassName, fps):

    global first, left_x_position, center, c, y_centers, left_zero, left_zero_check

    darknet.copy_image_from_bytes(DarkImg, ResizeImg.tobytes())
    detections = darknet.detect_image(netMain, ClassName, DarkImg, thresh=0.25)
    ##到這裡都是YOLO的固定程式內容##
    for detection in detections:
        #print(detection)
        print("Class_Name: " + detection[0])    
    image = cv2.cvtColor(ResizeImg, cv2.COLOR_BGR2RGB) #影片是BGR要轉成RGB
    cv2.imshow("Demo",image) #影片顯示時的標題名稱
    cv2.waitKey(3)

def catch_exit(): #按下鍵才會暫停
    global _Keep_Run
    #print(raw_input())
    _Keep_Run = False


if __name__ == "__main__":

    thread_list = []
    # init
    init()
    thread_list.append(threading.Thread(target=catch_exit))
    thread_list.append(threading.Thread(target=GMM))

    for th in thread_list:
        th.start()
    #
    for th in thread_list:
        th.join()
