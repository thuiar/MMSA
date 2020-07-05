# coding: utf-8
import os
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class dataPre():
    def __init__(self, working_dir):
        self.working_dir = working_dir

    def FetchFrames(self, input_dir, output_dir):
        """
        fetch frames from raw videos using ffmpeg toolkits
        """
        print("Start Fetch Frames...")
        video_pathes = sorted(glob(os.path.join(self.working_dir, input_dir, '*/*.mp4')))
        output_dir = os.path.join(working_dir,  output_dir)
        for video_path in tqdm(video_pathes):
            video_id, clip_id = video_path.split('/')[-2:]
            clip_id = clip_id.split('.')[0]
            clip_id = '%04d' % (int(clip_id))
            cur_output_dir = os.path.join(output_dir, video_id, clip_id)
            if not os.path.exists(cur_output_dir):
                os.makedirs(cur_output_dir)
            cmd = "ffmpeg -i " + video_path + " -r 30 " + cur_output_dir + "/%04d.png -loglevel quiet"
            os.system(cmd)
   
    def AlignFaces(self, input_dir, output_dir):
        """
        fetch faces from frames using MTCNN
        """
        print("Start Align Faces...")
        mtcnn = MTCNN(image_size=224, margin=0)

        frames_pathes = sorted(glob(os.path.join(self.working_dir, input_dir, "*/*/*.png")))
        for frames_path in tqdm(frames_pathes):
            output_path = frames_path.replace(input_dir, output_dir)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            try:
                img = Image.open(frames_path)
                mtcnn(img, save_path=output_path)
            except Exception as e:
                print(e)
                continue

    def FetchAudios(self, input_dir, output_dir):
        """
        fetch audios from videos using ffmpeg toolkits
        """
        print("Start Fetch Audios...")
        video_pathes = sorted(glob(os.path.join(working_dir, input_dir, '*/*.mp4')))
        for video_path in tqdm(video_pathes):
            output_path = video_path.replace(input_dir, output_dir).replace('.mp4', '.wav')
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            # 调用ffmpeg执行音频提取功能
            cmd = 'ffmpeg -i ' + video_path + ' -f wav -vn ' + \
                    output_path + ' -loglevel quiet'
            os.system(cmd)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='path to CH-SIMS')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dp = dataPre(args.data_dir)

    # print(args.data_dir)

    # fetch frames from videos
    dp.FetchFrames('Raw', 'Processed/video/Frames')

    # align faces
    dp.AlignFaces('Processed/video/Frames', 'Processed/video/AlignedFaces')

    # fetch audio
    dp.FetchAudios('Raw', 'Processed/audio') # run in 3 down