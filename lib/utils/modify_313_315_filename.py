import os

root_path_list = []
root_path_list.append(os.path.join('data','zju_mocap','CoreView_313'))
root_path_list.append(os.path.join('data','zju_mocap','CoreView_313', 'mask_cihp'))
root_path_list.append(os.path.join('data','zju_mocap','CoreView_313', 'mask'))
root_path_list.append(os.path.join('data','zju_mocap','CoreView_315'))
root_path_list.append(os.path.join('data','zju_mocap','CoreView_315', 'mask_cihp'))
root_path_list.append(os.path.join('data','zju_mocap','CoreView_315', 'mask'))

cam_lists = list(range(1,20))+list(range(22,24))

for root_path in root_path_list:
	for cam_idx in cam_lists:
		camera = 'Camera ({})'.format(cam_idx)
		img_folder=os.path.join(root_path,camera)

		files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder,f))]

		for file in files:

			old_filepath = os.path.join(img_folder,file)

			if os.path.basename(root_path) in ['mask_cihp']:
				new_filename = file.split('_')[4]+'.png'
			elif os.path.basename(root_path) in ['CoreView_313','CoreView_315']:
				new_filename = file.split('_')[4] + '.jpg'

			new_filepath = os.path.join(img_folder,new_filename)
			os.rename(old_filepath,new_filepath)