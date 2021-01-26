"""
                        CV traffic analysis
                          Video processing

This code creates the VideoTracker class and provides basic command line interface to
process video inputs.

"""


#-------------------------------------------------------------------------------------
# Settings
import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import pandas as pd

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results


#-------------------------------------------------------------------------------------

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
            assert os.path.isfile(self.video_path), "Error: path not found!"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()
        
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            
            # Paths of saved video and results
            def parse_file_name(path):
                in_file_name = os.path.basename(path)
                video_name = in_file_name.split('.')[0] + '.avi'
                results_name = in_file_name.split('.')[0] + '.csv'
                return video_name, results_name
            
            video_output_filename, results_filename = parse_file_name(self.video_path)
            
            self.save_video_path = os.path.join(self.args.save_path, video_output_filename)
            self.save_results_path = os.path.join(self.args.save_path, results_filename)
            
            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))
            
            # logging
            self.logger.info("Saving results to {}".format(self.save_results_path))
        
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
    
    def run(self):
        # Base variables
        idx_frame = 0
        
        # Create empty results array to be appended with frame data
        self.results_array = np.empty(shape = (0,7))
        
        # Loop over video frames
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue
            
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            
            # Detection on each frame
            detections_t = self.detector(im)
            bbox_xywh, cls_conf, cls_ids = detections_t[0], detections_t[1], detections_t[2]            
            
            # Filter detection classes to only relevant cases.
            # This is mostly to remove noise as it doesn't affect performance.
            keep_classes = [0, # person
                            1, # bicycle
                            2, # car 
                            3, # motorbyke
                            5, # bus
                            7] # truck
            
            mask = np.isin(cls_ids, keep_classes)
            
            # Process detections
            bbox_xywh = bbox_xywh[mask]
            #Bbox dilation just in case bbox too small
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]
            
            # Uodate tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)
            
            # # Make public attributes for debugging
            #self.im = im
            # self.outputs = outputs
            # self.detections = detections_t
            # self.cls_conf = cls_conf
            # self.cls_ids = cls_ids
            
            # Draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4].astype(int)
                identities = outputs[:, 4].astype(int)
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                
                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                
            
            end = time.time()
            
            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)
            
            if self.args.save_path:
                self.writer.write(ori_im)
            
            #------------------------------------------------------------------------
            # Exporting data processing
            
            # This processes each frame tracking data and appends it to the results
            # array that will be exported
            
            if len(outputs) > 0:
                # Tracking data for frame
                tracking_array_i = outputs
                
                # Add frame number to tracking array
                frame_num_array_i = np.full((tracking_array_i.shape[0], 1), idx_frame - 1)
                results_array_i = np.append(frame_num_array_i, tracking_array_i, 1)
                
                # Add frame data to results array
                self.results_array = np.append(self.results_array, results_array_i,0)
            
            #------------------------------------------------------------------------
            # Logging
            self.logger.info("frame: {},time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(idx_frame - 1, end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))\
            
            # Make it shorter for piloting
            # if idx_frame > 10:
            #     break
        #----------------------------------------------------------------------------
        # Export outputs
        # Turn to pandas and export csv
        pd.DataFrame(self.results_array, 
                    columns= ['frame', 'x_i', 'y_i', 'x_j', 'y_j','obj_id', 'class']).\
                to_csv(self.save_results_path, index = False)


#-------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="../output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    
    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
