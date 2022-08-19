from os.path import join
import sys

sys.path.append("/media/medical/gasperp/projects/dissm")
print(sys.path)
import gasperp_functions.convert_functions as convert_functions

organ_name = "Parotid_L"
dst_dir = r"/media/medical/projects/head_and_neck/onkoi_2019/dissm/pddca_pred_180"

convert_functions.convert_dir(join(dst_dir, organ_name))

# conda activate itk && python /media/medical/gasperp/projects/dissm/copy_data/convert_pddca_pred_180.py && conda deactivate
