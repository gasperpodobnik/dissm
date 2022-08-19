import os
import shutil
from os.path import join
from pathlib import Path
import SimpleITK as sitk
from utilities import utilities

label_num = 5
fps = [
    i
    for i in utilities.list_all_files_in_dir_and_subdir(
        "/media/medical/projects/head_and_neck/nnUnet/Task192_PDDCA",
        suffix=".nii.gz",
    )
    if "labelsT" in i
]

dst_dir = r"/media/medical/projects/head_and_neck/onkoi_2019/dissm/pddca_raw"
organ_name = "Parotid_L"

os.makedirs(join(dst_dir, organ_name, "nifti"), exist_ok=True)
for seg_pth in fps:
    fname = Path(seg_pth).name
    organ_sitk = utilities.get_binary_mask_from_multilabel_segmentation(
        sitk.ReadImage(seg_pth), label_num=label_num
    )
    if organ_sitk is None:
        continue
    sitk.WriteImage(
        sitk.Cast(organ_sitk, sitk.sitkUInt8),
        join(dst_dir, organ_name, "nifti", fname),
    )
