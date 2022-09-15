import os
from logging import Logger

from modules.pose import DataHandler as pose_datahandler
from modules.utils.video import Capture, Writer
from tqdm import tqdm

from . import pose as pose_vis


class Visualizer:
    def __init__(self, logger: Logger):
        self._logger = logger

        self._do_keypoints = True

    def visualise(self, video_path: str, data_dir: str):
        pose_results = pose_datahandler.load(
            os.path.join(data_dir, ".json", "keypoints.json")
        )

        # create video capture
        self._logger.info(f"=> loading video from {video_path}.")
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        tmp_frame = video_capture.read()[1]
        video_capture.set_pos_frame_count(0)

        out_paths = []
        # create video writer for keypoints results
        if self._do_keypoints:
            out_path = os.path.join(data_dir, "keypoints.mp4")
            pose_video_writer = Writer(
                out_path, video_capture.fps, tmp_frame.shape[1::-1]
            )
            out_paths.append(out_path)

        self._logger.info(f"=> writing video into {out_paths}.")
        for frame_num in tqdm(range(video_capture.frame_count), ncols=100):
            frame_num += 1  # frame_num = (1, ...)
            ret, frame = video_capture.read()

            # write keypoint video
            frame = pose_vis.write_frame(frame, pose_results, frame_num)
            if self._do_keypoints:
                pose_video_writer.write(frame)

        # release memory
        del video_capture
        if self._do_keypoints:
            del pose_video_writer
