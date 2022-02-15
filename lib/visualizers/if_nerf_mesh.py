from lib.utils.if_nerf import voxels
import numpy as np
from lib.config import cfg
import os
from termcolor import colored

class Visualizer:
    def __init__(self):
        result_dir = 'data/mesh/{}/{}/{}'.format(cfg.exp_name,'epoch_'+str(cfg.test.epoch),cfg.exp_folder_name)
        print(colored('the results are saved at {}'.format(result_dir), 'yellow'))

    def visualize_voxel(self, output, batch):
        cube = output['cube']
        cube = cube[10:-10, 10:-10, 10:-10]
        cube[cube < cfg.mesh_th] = 0
        cube[cube > cfg.mesh_th] = 1

        sh = cube.shape
        square_cube = np.zeros((max(sh), ) * 3)
        square_cube[:sh[0], :sh[1], :sh[2]] = cube
        voxel_grid = voxels.VoxelGrid(square_cube)
        mesh = voxel_grid.to_mesh()
        mesh.show()

    def visualize(self, output, batch):
        mesh = output['mesh']
        result_dir = 'data/mesh/{}/{}/{}/{}'.format(cfg.exp_name,
                                                'epoch_' + str(cfg.test.epoch),
                                                cfg.exp_folder_name, str(batch['human_idx'].item()))
        os.system('mkdir -p {}'.format(result_dir))
        i = batch['frame_index'][0].item()
        result_path = os.path.join(result_dir, '{}.ply'.format(i))
        print(result_path)
        mesh.export(result_path)

