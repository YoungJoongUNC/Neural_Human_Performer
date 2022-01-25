import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os
from termcolor import colored

class Visualizer:
    def __init__(self):
        data_dir = 'data/render/{}'.format(cfg.exp_name)
        print(colored('the results are saved at {}'.format(data_dir), 'yellow'))

    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        if cfg.white_bkgd:
            img_pred = np.ones((H, W, 3))
        else:
            img_pred = np.zeros((H, W, 3))

        img_pred[mask_at_box] = rgb_pred
        img_pred = img_pred[..., [2, 1, 0]]

        depth_pred = np.zeros((H, W))
        depth_pred[mask_at_box] = output['depth_map'][0].detach().cpu().numpy()

        img_root = 'data/render/{}/{}'.format(cfg.exp_name,
                                              batch['ind'].item())
        os.system('mkdir -p {}'.format(img_root))
        index = batch['index'].item()

        cv2.imwrite(os.path.join(img_root, '{}.jpg'.format(index)),
                    img_pred * 255)
