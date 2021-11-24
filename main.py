import torch
from model import BasicVSRPlusPlus
import mmcv
import cv2
import os
import numpy as np
from utils import chop_forward_dim5
from copy import deepcopy

def read_video(video_path):
    video = mmcv.VideoReader(video_path)
    return video


def inference(video, model, device, input_frames = 10, output_dir='./result'):
    frame_count = len(video)
    h, w, c = video[0].shape

    step = 0
    frame_index = 0

    while(step < frame_count):
        if step + input_frames >= frame_count:
            sliding_window = frame_count - step
        else:
            sliding_window = input_frames
        
        lrs_ndarray = video[step : step+sliding_window]

        lrs_zero_to_one = [v.astype(np.float32) / 255. for v in lrs_ndarray]
        lrs_tensor = [torch.from_numpy(v).permute(2,0,1) for v in lrs_zero_to_one]
        lrs = torch.cat(lrs_tensor).view(-1, c, h, w).unsqueeze(0)

        # put inputs to GPU
        lrs = lrs.to(device)

        # GPU out of memory ... 
        output = chop_forward_dim5(lrs, model)

        output_ndarray = output.squeeze(0).detach().cpu()

        for i in range(sliding_window):
            frame = output_ndarray[i].permute(1,2,0).numpy()
            frame_name = os.path.join(output_dir, "{:08d}.png".format(frame_index))
            cv2.imwrite(frame_name, frame * 255.)
            frame_index += 1

        step += input_frames


def load_model(model, device, model_path):
    load_net = torch.load(model_path, map_location=lambda storage, loc:storage)
    load_net = load_net['state_dict']

    choose_key = 'generator'
    for key, value in deepcopy(load_net).items():
        key_list = key.split('.')

        if choose_key in key_list:
            tmp_key = ".".join(key_list[1:])
            load_net[tmp_key] = value
    
        load_net.pop(key)

    model.load_state_dict(load_net, strict=True)

    return model


if __name__ == "__main__":
    # read a video
    video_path = r'C:\Users\test.mp4'
    video = read_video(video_path)

    # set your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load your checkpoints
    model_path = r'E:\weight\basicvsr_pp_youku_iter_150000.pth'
    model = BasicVSRPlusPlus().to(device)
    model = load_model(model, device, model_path)

    # inference and save imgs
    inference(video, model, device)