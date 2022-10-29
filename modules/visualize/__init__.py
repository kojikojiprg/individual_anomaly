import os
from logging import Logger

from modules.pose import PoseDataHandler
from modules.individual import IndividualDataHandler
from modules.utils.video import Capture, Writer
from tqdm import tqdm

from . import pose as pose_vis


class Visualizer:
    def __init__(self, args, logger: Logger):
        self._logger = logger

        self._do_pose_estimation = True
        # self._do_individual = args.individual
        self._no_bg = args.video_no_background

    def visualise(self, video_path: str, data_dir: str):
        # load data
        if self._do_pose_estimation:
            pose_data_lst = PoseDataHandler.load(data_dir, self._logger)
        # if self._do_individual:
        #     ind_data_lst = IndividualDataHandler.load(data_dir, self._logger)

        # create video capture
        self._logger.info(f"=> loading video from {video_path}.")
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        tmp_frame = video_capture.read()[1]
        video_capture.set_pos_frame_count(0)

        out_paths = []
        video_num = os.path.basename(video_path).split(".")[0]
        # create video writer for pose estimation results
        if self._do_pose_estimation:
            if not self._no_bg:
                out_path = os.path.join(data_dir, f"{video_num}_pose.mp4")
            else:
                out_path = os.path.join(data_dir, f"{video_num}_pose_nobg.mp4")

            pose_video_writer = Writer(
                out_path, video_capture.fps, tmp_frame.shape[1::-1]
            )
            out_paths.append(out_path)

        # create video writer for individual results
        # if self._do_individual:
        #     out_path = os.path.join(data_dir, f"individual_{}.mp4")

        #     pose_video_writer = Writer(
        #         out_path, video_capture.fps, tmp_frame.shape[1::-1]
        #     )
        #     out_paths.append(out_path)

        self._logger.info(f"=> writing video into {out_paths}.")
        for frame_num in tqdm(range(video_capture.frame_count), ncols=100):
            frame_num += 1  # frame_num = (1, ...)
            ret, frame = video_capture.read()

            # write pose estimation video
            frame = pose_vis.write_frame(frame, pose_data_lst, frame_num, self._no_bg)
            if self._do_pose_estimation:
                pose_video_writer.write(frame)

        # release memory
        del video_capture
        if self._do_pose_estimation:
            del pose_video_writer
