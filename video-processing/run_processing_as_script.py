from video_processing import *

VIDEO_PATH = "../11 sample-short.mp4"

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
        self.save_path ="../output/"
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

vdo_trk.save_video_path

# # # vdo_trk.args.cam
# vdo_trk.cls_conf.shape
# # vdo_trk.cls_ids.shape
# vdo_trk.outputs
# vdo_trk.detections

# vdo_trk.results_array
#------------------------------------------------------------------------
# Exporting data

# final_mat = np.empty(shape = (0,8))

# # Create exporting elements per frame
# cls_ids_exp = np.expand_dims(vdo_trk.cls_ids, axis=0).transpose()
# cls_conf_exp = np.expand_dims(vdo_trk.cls_conf, axis=0).transpose()
# bbox_xyxy_exp = vdo_trk.outputs[:, :4] # Bbbox, first 4 elemeents of outputs
# ids_exp = vdo_trk.outputs[:, -1:] # Tracking ids, last element of outputs
# frame_mat = np.full((cls_ids_exp.shape[0],1), idx_frame)

# # Create exporting matrix. Meio gambiarra de numpy, mas e a vida.
# classes_mat = np.append(cls_ids_exp, cls_conf_exp, 1 )
# boxes_mat = np.append(bbox_xyxy_exp, classes_mat, 1 )
# ids_box_mat = np.append(ids_exp, boxes_mat, 1)

# final_mat_frame = np.append(frame_mat, ids_box_mat, 1)


# final_mat = np.append(final_mat, final_mat_frame, 0)

# # Turn to pandas and export csv
# pd.DataFrame(final_mat, 
#              columns= ['frame', 'obj_id', 'x_i', 'y_i', 'x_j', 'y_j', 'class', 'conf']).\
#     to_csv(self.save_results_path)


cv2.destroyAllWindows()


