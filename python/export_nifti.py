# -*- coding: utf-8 -*-
"""导出 NIfTI: 重建体数据 + GT 体数据 → .nii.gz (供 3D Slicer / ITK-SNAP 等查看)。

用法:
    python export_nifti.py [时间戳]
输出:
    data/recon_<ts>.nii.gz   重建体数据(400³, 0.5mm 体素)
    data/gt_<ts>.nii.gz      GT 体数据(400³, 0.5mm 体素)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import nibabel as nib

from config import DATA_OUT, VOL_N


def save_nifti(path, vol, pitch=0.5):
    # 对角 affine: 体素间距 pitch mm, 原点在 (0,0,0)
    affine = np.diag([pitch, pitch, pitch, 1.0])
    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    nib.save(img, path)


def main(ts):
    vol = np.fromfile(f'{DATA_OUT}/volume_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    gt = np.fromfile(f'{DATA_OUT}/gt_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    rp = f'{DATA_OUT}/recon_{ts}.nii.gz'
    gp = f'{DATA_OUT}/gt_{ts}.nii.gz'
    save_nifti(rp, vol)
    save_nifti(gp, gt)
    print(f'[write] {rp}')
    print(f'[write] {gp}')
    print(f'  体素 0.5mm, 尺寸 {VOL_N}^3, 可用 3D Slicer 打开对比')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '1094408')
