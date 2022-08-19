from os.path import join

# from gasperp_functions import convert_functions
import convert_functions


organ_name = "Parotid_L"
dst_dir = r"/media/medical/projects/head_and_neck/onkoi_2019/dissm/pddca_raw"

convert_functions.convert_dir(join(dst_dir, organ_name))

# export PYTHONPATH=/media/medical/gasperp/projects/dissm:$PYTHONPATH
# conda activate itk && export PYTHONPATH=/media/medical/gasperp/projects/dissm/gasperp_functions:$PYTHONPATH && python /media/medical/gasperp/projects/dissm/analyse_dataset/pddca_raw/02_convert.py && conda deactivate
