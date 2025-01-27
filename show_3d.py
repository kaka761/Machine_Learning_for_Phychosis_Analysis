import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

f = nib.load('./Projects/MRI/seg/ARMS_NT/mwp1ARMS006.nii')
OrthoSlicer3D(f.dataobj[:,:,:]).show()
