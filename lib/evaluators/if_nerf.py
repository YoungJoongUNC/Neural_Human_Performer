import numpy as np
from lib.config import cfg
from skimage.metrics import structural_similarity
import os
import cv2
import matplotlib.pyplot as plt
from termcolor import colored

class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []

        result_dir = os.path.join(cfg.result_dir,
                                  'epoch_' + str(cfg.test.epoch),
                                  cfg.exp_folder_name)
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

    def psnr_metric(self, img_pred, img_gt):

        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        if cfg.white_bkgd:
            img_pred = np.ones((H, W, 3))
        else:
            img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred

        if cfg.white_bkgd:
            img_gt = np.ones((H, W, 3))
        else:
            img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]

        result_dir = os.path.join(cfg.result_dir,
                                  'epoch_' + str(cfg.test.epoch),
                                  cfg.exp_folder_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        human_dir = os.path.join(result_dir, str(batch['human_idx'].item()))
        if not os.path.exists(human_dir):
            os.makedirs(human_dir)

        pred_dir = os.path.join(human_dir, 'pred')
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        gt_dir = os.path.join(human_dir, 'gt')
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)

        input_dir = os.path.join(human_dir, 'input')
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{}_view{}.png'.format(pred_dir, frame_index,
                                           view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{}_view{}_gt.png'.format(gt_dir, frame_index,
                                              view_index),
            (img_gt[..., [2, 1, 0]] * 255))

        for t in range(cfg.time_steps):
            for view in range(len(cfg.training_view)):
                tmp = batch['input_imgs'][t][0][view]

                tmp = tmp.data.detach().cpu().numpy().transpose(1, 2, 0)
                (tmp * 255).astype(np.uint8)
                plt.imsave(
                    '{}/frame{}_t_{}_view_{}.png'.format(input_dir, frame_index,
                                                         t, view), tmp)

        # compute the ssim
        ssim = structural_similarity(img_pred, img_gt, multichannel=True)

        return ssim

    def evaluate(self, output, batch):

        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        mse = np.mean((rgb_pred - rgb_gt) ** 2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

        mse_str = 'mse: {}'.format(np.mean(self.mse))
        psnr_str = 'psnr: {}'.format(np.mean(self.psnr))
        ssim_str = 'ssim: {}'.format(np.mean(self.ssim))

        print(mse_str)
        print(psnr_str)
        print(ssim_str)

    def summarize(self):

        result_root = os.path.join(cfg.result_dir,
                                   'epoch_' + str(cfg.test.epoch),
                                   cfg.exp_folder_name)

        mse_path = os.path.join(result_root, 'mse.npy')
        psnr_path = os.path.join(result_root, 'psnr.npy')
        ssim_path = os.path.join(result_root, 'ssim.npy')

        os.system('mkdir -p {}'.format(result_root))
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
        np.save(mse_path, self.mse)
        np.save(psnr_path, self.psnr)
        np.save(ssim_path, self.ssim)

        exp_str = 'experiment: {}'.format(cfg.exp_name)
        epoch_str = 'epoch: {}'.format(cfg.test.epoch)
        mse_str = 'mse: {}'.format(np.mean(self.mse))
        psnr_str = 'psnr: {}'.format(np.mean(self.psnr))
        ssim_str = 'ssim: {}'.format(np.mean(self.ssim))

        print(exp_str)
        print(epoch_str)
        print(mse_str)
        print(psnr_str)
        print(ssim_str)

        with open(os.path.join(result_root, 'summary.txt'), 'w') as out:
            out.writelines([exp_str, epoch_str, mse_str, psnr_str, ssim_str])
        self.mse = []
        self.psnr = []
        self.ssim = []

