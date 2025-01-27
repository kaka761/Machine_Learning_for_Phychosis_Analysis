import os
import SimpleITK as sitk
import shutil

def createNewMask(file_path,dst_path):
    file_base_name = os.path.basename(file_path)
    sitkImage = sitk.ReadImage(file_path)              
    npImage = sitk.GetArrayFromImage(sitkImage)      
    npImage[npImage > 0 ] =1                          
    outImage = sitk.GetImageFromArray(npImage)         
    outImage.SetSpacing(sitkImage.GetSpacing())        
    outImage.SetDirection(sitkImage.GetDirection())    
    outImage.SetOrigin(sitkImage.GetOrigin())          
    sitk.WriteImage(outImage,os.path.join(dst_path,file_base_name))

if not os.path.exists('./Projects/MRI/ML/MyDataRaw'):
    os.makedirs('./Projects/MRI/ML/MyDataRaw')

source_data_path = './Projects/MRI/raw'
dst_data_path = './Projects/MRI/ML/MyDataRaw'
kinds = ['ARMS_NT','ARMS_T', 'FEP', 'C']
index = 0
for kind in kinds:
    kind_path = os.path.join(source_data_path, kind)
    for folder in os.listdir(kind_path):
        file_name =  os.path.join(kind_path,folder )
        dst_path =  os.path.join(dst_data_path, kind)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        if 'ttt' in file_name:
            dst_file_path = os.path.join(dst_path, file_name.split('/')[-1])
            shutil.copy(file_name, dst_file_path)
            index += 1
        elif '.nii' in file_name:
            createNewMask(file_name, dst_path)
print("Done")