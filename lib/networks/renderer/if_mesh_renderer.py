import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh
import matplotlib.pyplot as plt
import numpy as nps
import gc
import trimesh
import math


class Renderer:
    def __init__(self, net):
        self.net = net

    def paint_neural_human(self, batch, t, holder_feat_map, holder_feat_scale,
                           prev_weight=None, prev_holder=None):

        smpl_vertice = batch['smpl_vertice'][t]

        if cfg.rasterize:
            vizmap = batch['input_vizmaps'][t]

        image_shape = batch['input_imgs'][t].shape[-2:]

        input_R = batch['input_R']
        input_T = batch['input_T']
        input_K = batch['input_K']

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)

        if cfg.rasterize:
            result = vizmap[0]

        # uv
        vertice_rot = \
            torch.matmul(input_R[:, None], smpl_vertice.unsqueeze(-1))[..., 0]
        vertice = vertice_rot + input_T[:, None, :3, 0]
        vertice = torch.matmul(input_K[:, None], vertice.unsqueeze(-1))[..., 0]
        uv = vertice[:, :, :2] / vertice[:, :, 2:]

        latent = self.sample_from_feature_map(holder_feat_map,
                                              holder_feat_scale, image_shape,
                                              uv)

        latent = latent.permute(0, 2, 1)

        num_input = latent.shape[0]

        if cfg.use_viz_test:

            final_result = result
            big_holder = torch.zeros((latent.shape[0], latent.shape[1],
                                      cfg.embed_size)).cuda()  # .to(torch.cuda.current_device())
            big_holder[final_result == True, :] = latent[final_result == True,
                                                  :]

            if cfg.weight == 'cross_transformer':
                return final_result, big_holder  # [4, 6890], [4, 6890, 64]

        else:  # not using viz test

            holder = latent.sum(0)
            holder = holder / num_input
            return holder

    def sample_from_feature_map(self, feat_map, feat_scale, image_shape, uv):

        scale = feat_scale / image_shape
        scale = torch.tensor(scale).to(dtype=torch.float32).to(
            device=torch.cuda.current_device())

        uv = uv * scale - 1.0
        uv = uv.unsqueeze(2)

        samples = F.grid_sample(
            feat_map,
            uv,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )

        return samples[:, :, :, 0]

    def get_pixel_aligned_feature(self, batch, xyz, pixel_feat_map,
                                  pixel_feat_scale, batchify=False):



        image_shape = batch['input_imgs'][0].shape[-2:]
        input_R = batch['input_R']
        input_T = batch['input_T']
        input_K = batch['input_K']

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)


        if batchify == False:
            xyz = xyz.view(xyz.shape[0], -1, 3)
        xyz = repeat_interleave(xyz, input_R.shape[0])
        xyz_rot = torch.matmul(input_R[:, None], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + input_T[:, None, :3, 0]
        xyz = torch.matmul(input_K[:, None], xyz.unsqueeze(-1))[..., 0]
        uv = xyz[:, :, :2] / xyz[:, :, 2:]

        pixel_feat = self.sample_from_feature_map(pixel_feat_map,
                                                  pixel_feat_scale, image_shape,
                                                  uv)

        return pixel_feat

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def pts_to_can_pts(self, pts, batch):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch['Th'][:, None]
        pts = pts - Th
        R = batch['R']
        sh = pts.shape
        pts = torch.matmul(pts.view(sh[0], -1, sh[3]), R)
        pts = pts.view(*sh)
        return pts

    def transform_sampling_points(self, pts, batch):
        if not self.net.training:
            return pts
        center = batch['center'][:, None, None]
        pts = pts - center
        rot = batch['rot']
        pts_ = pts[..., [0, 2]].clone()
        sh = pts_.shape
        pts_ = torch.matmul(pts_.view(sh[0], -1, sh[3]), rot.permute(0, 2, 1))
        pts[..., [0, 2]] = pts_.view(*sh)
        pts = pts + center
        trans = batch['trans'][:, None, None]
        pts = pts + trans
        return pts

    def prepare_sp_input(self, batch):

        sp_input = {}

        # feature: [N, f_channels]
        sh = batch['feature'].shape
        sp_input['feature'] = batch['feature'].view(-1, sh[-1])

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        sp_input['i'] = batch['i']

        return sp_input

    def get_grid_coords(self, pts, sp_input, batch):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def batchify_rays(self,
                      sp_input,
                      grid_coords,
                      viewdir=None,
                      light_pts=None,
                      chunk=1024 * 32,
                      net_c=None,
                      batch=None,
                      xyz=None,
                      pixel_feat_map=None,
                      pixel_feat_scale=None,
                      norm_viewdir=None,
                      holder=None,
                      embed_xyz=None):
        """Render rays in smaller minibatches to avoid OOM.
        """

        all_ret = []

        for i in range(0, grid_coords.shape[1], chunk):

            if cfg.run_mode == 'test':
                gc.collect()
                torch.cuda.empty_cache()


            xyz_shape = xyz.shape
            xyz = xyz.reshape(xyz_shape[0], -1, 3)


            pixel_feat = self.get_pixel_aligned_feature(batch,
                                                        xyz[:, i:i + chunk],
                                                        pixel_feat_map,
                                                        pixel_feat_scale)
            ret = self.net(pixel_feat, sp_input,
                           grid_coords[:, i:i + chunk], holder=holder)

            all_ret.append(ret)
            if cfg.run_mode == 'test':
                gc.collect()
                torch.cuda.empty_cache()


        all_ret = torch.cat(all_ret, 1)

        return all_ret



    def render(self, batch):

        pts = batch['pts']
        sh = pts.shape
        light_pts = pts.clone()
        xyz = pts.clone()

        inside = batch['inside'][0].bool()

        pts = pts[0][inside][None]
        light_pts = light_pts[0][inside][None]
        xyz = xyz[0][inside][None]

        light_pts = embedder.xyz_embedder(light_pts)

        pts = pts.view(sh[0], -1, 1, 3)
        pts = self.pts_to_can_pts(pts, batch)

        sp_input = self.prepare_sp_input(batch)

        grid_coords = self.get_grid_coords(pts, sp_input, batch)
        grid_coords = grid_coords.view(sh[0], -1, 3)

        image_list = batch['input_imgs']

        weight = None
        holder = None

        temporal_holders = []
        temporal_weights = []

        for t in range(cfg.time_steps):

            images = image_list[t].reshape(-1, *image_list[t].shape[2:])

            if t == 0:
                holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = self.net.encoder(
                    images)
            else:
                holder_feat_map, holder_feat_scale, _, _ = self.net.encoder(
                    images)


            ### --- paint the holder

            weight, holder = self.paint_neural_human(batch, t,
                                                     holder_feat_map,
                                                     holder_feat_scale,
                                                     weight, holder)
            if cfg.weight == 'cross_transformer':
                if cfg.cross_att_mode == 'cross_att':
                    temporal_holders.append(holder)
                    temporal_weights.append(weight)

        if cfg.time_steps == 1:
            holder = temporal_holders[0]



        if grid_coords.size(1) < 1024 * 32:  # ray_o.shape = [batch, 1024, 3]

            pixel_feat = self.get_pixel_aligned_feature(batch, xyz,
                                                        pixel_feat_map,
                                                        pixel_feat_scale)
            alpha = self.net(pixel_feat, sp_input, grid_coords,
                             holder=holder)

        else:

            alpha = self.batchify_rays(sp_input, grid_coords, viewdir=None,
                                       light_pts=None,
                                       chunk=1024 * 32, net_c=None,
                                       batch=batch, xyz=xyz,
                                       pixel_feat_map=pixel_feat_map,
                                       pixel_feat_scale=pixel_feat_scale,
                                       holder=holder)

        alpha = alpha[0, :, 0].detach().cpu().numpy()
        cube = np.zeros(sh[1:-1])
        inside = inside.detach().cpu().numpy()
        cube[inside == 1] = alpha

        cube = np.pad(cube, 10, mode='constant')

        data_name = cfg.virt_data_root.split('/')[-1]

        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)

        mesh = trimesh.Trimesh(vertices, triangles)

        ret = {'cube': cube, 'mesh': mesh}

        return ret
