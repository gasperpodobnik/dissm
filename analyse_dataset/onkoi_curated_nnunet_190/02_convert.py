from os.path import join

# from gasperp_functions import convert_functions
import convert_functions

dataset = "onkoi_curated_nnunet_190"
organ_name = "Parotid_L"
dst_dir = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/{}".format(dataset)

convert_functions.convert_dir(join(dst_dir, organ_name))

# export PYTHONPATH=/media/medical/gasperp/projects/dissm:$PYTHONPATH
# conda activate itk && export PYTHONPATH=/media/medical/gasperp/projects/dissm/gasperp_functions:$PYTHONPATH && python /media/medical/gasperp/projects/dissm/analyse_dataset/onkoi_curated_nnunet_190/02_convert.py && conda deactivate
