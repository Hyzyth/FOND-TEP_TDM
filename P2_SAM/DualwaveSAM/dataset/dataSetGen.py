import numpy as np
import utils
import nibabel as nib
import os  # 导入 os 用于操作文件路径

def read_data(path_to_nifti, return_numpy=True):
    """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
    if return_numpy:
        return nib.load(str(path_to_nifti)).get_fdata()
    return nib.load(str(path_to_nifti))



dataname = "Hecktor"   # CHEN/Hecktor


# 获取病人文件夹的路径
if dataname == "CHEN":
    paths = utils.get_paths_to_patient_files('/data/code/med_test_case/CA_Seg_161_crop')
else:
    paths = utils.get_paths_to_patient_files('/data/code/H-process/Hecktor')

# 输出文件夹数量
print(f"共发现 {len(paths)} 个病例")

# 遍历每个病例
for i in range(len(paths)):
    print(f"正在处理第 {i} 个病例")
    
    # 读取CT、PET和Mask
    ct = read_data(paths[i][0])
    print(f"CT shape: {ct.shape}")
    pt = read_data(paths[i][1])
    print(f"PET shape: {pt.shape}")
    mask = read_data(paths[i][2])
    print(f"Mask shape: {mask.shape}")
    
    # 将CT和PET数据堆叠在一起
    input_data = np.stack([ct, pt], axis=-1)
    
    # 获取病人的id，这里假设路径中的病人文件夹名字即为病人ID
    patient_id = os.path.basename(os.path.dirname(paths[i][0]))  # 获取路径中父文件夹的名称作为病人ID
    
    # 创建保存路径
    save_path = f"{dataname}/{patient_id}.npz"
    
    # 保存每个病人的数据
    np.savez(save_path, input=input_data, target=mask)
    print(f"保存病人 {patient_id} 的数据到 {save_path}")
