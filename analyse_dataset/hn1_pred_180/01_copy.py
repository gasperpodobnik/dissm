import os
from os.path import join

import SimpleITK as sitk
from tqdm import tqdm
from utilities import utilities

dst_dir = r"/media/medical/projects/head_and_neck/onkoi_2019/dissm/hn1_pred_180"
nnunet_task_dir = (
    "/media/medical/projects/head_and_neck/nnUnet/Task180_onkoi-2019-CT-resampled"
)
organ_name = "Parotid_L"
results_dir = join(
    nnunet_task_dir,
    "results/FOLD-all_TRAINER-nnUNetTrainerV2_noMirroringAxis2_PLANS-nnUNetPlansv2.1_CHK-model_final_checkpoint_DATASET-193",
)


# load nnunet dataset dict
dataset_dict = utilities.read_dict_in_json(join(nnunet_task_dir, "dataset.json"))
# find label number that is used for the organ you want to extract
parotid_l_label = {j: int(i) for i, j in dataset_dict["labels"].items()}[organ_name]

os.makedirs(join(dst_dir, organ_name, "nifti"), exist_ok=True)
fps = utilities.list_all_files_in_dir_and_subdir(results_dir, suffix=".nii.gz")
for fp in tqdm(fps, total=len(fps)):
    fname = os.path.basename(fp)
    organ_sitk = utilities.get_binary_mask_from_multilabel_segmentation(
        sitk.ReadImage(fp), label_num=parotid_l_label
    )
    sitk.WriteImage(
        sitk.Cast(organ_sitk, sitk.sitkUInt8), join(dst_dir, organ_name, "nifti", fname)
    )
