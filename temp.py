import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

VIDEO_PATH = 'C:/Users/wb519128/GitHub/deep_sort_pytorch/data/2-sample-2020.mp4'

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)
        
        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
    
    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]
        
        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()
        
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            
            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")
            
            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))
            
            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))
        
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
    
    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
        # for ii in range(10):
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue
            
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            
            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            
            # TEST TO SEE IF THIS WORKS
            self.bbox_xywh, self.cls_conf, self.cls_ids = self.detector(im)
            
            # select person class
            person = cls_ids==0
            car = cls_ids==2
            mask = person + car
            
            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
            
            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
            
            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                
                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                
                results.append((idx_frame - 1, bbox_tlwh, identities))
            
            end = time.time()
            
            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)
            
            if self.args.save_path:
                self.writer.write(ori_im)
            
            # save results
            write_results(self.save_results_path, results, 'mot')
            
            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))

# Emulate parser behaviour so I can run in on interactive mode without making significant changes to the code

class mock_parser():
    def __init__(self, VIDEO_PATH):
        self.VIDEO_PATH = VIDEO_PATH
        self.config_detection = "./configs/yolov3.yaml"
        self.config_deepsort ="./configs/deep_sort.yaml"
        self.display ="store_true"
        self.frame_interval = 1
        self.display_width = 800
        self.display_height = 600
        self.save_path ="./output/"
        # self.use_cuda = True
        self.use_cuda = False
        self.cam =-1

args = mock_parser(VIDEO_PATH)

cfg = get_config()
cfg.merge_from_file(args.config_detection)
cfg.merge_from_file(args.config_deepsort)


#--------------------------------------------------------------------------------------------------

# vdo_trk = VideoTracker(cfg, args, video_path=args.VIDEO_PATH)
# vdo_trk.run()

# _, ori_im = vdo_trk.vdo.retrieve()
# im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()

vdo_trk.args.cam
vdo_trk.bbox_xywh
vdo_trk.cls_conf
vdo_trk.cls_ids


# cv2.destroyAllWindows()


