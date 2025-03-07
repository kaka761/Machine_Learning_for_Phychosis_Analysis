import SimpleITK as sitk
import six
import os
import numpy as np
import matplotlib.pyplot as plt

import time
import tqdm



class FeatureMapVisualizition:
    def __init__(self):
        self.original_image_path = ''
        self.original_roi_path = ''
        self.feature_map_path = ''
        self.feature_name = ''

    def LoadData(self,original_image_path, original_roi_path, feature_map_path):
        self.original_image_path = original_image_path
        self.original_roi_path = original_roi_path
        self.feature_map_path = feature_map_path
        self.feature_name = (os.path.split(self.feature_map_path)[-1]).split('.')[0]
        _, _, self.original_image_array = self.LoadNiiData(self.original_image_path, is_show_info=False)
        _, _, self.original_roi_array = self.LoadNiiData(self.original_roi_path, is_show_info=False)
        self._max_roi_index = np.argmax(np.sum(self.original_roi_array, axis=(0,1)))
        _, _, self.original_feature_map_array = self.LoadNiiData(self.feature_map_path, is_show_info=False)
        if os.path.splitext(self.feature_map_path)[-1] == '.nrrd':
            self.original_feature_map_array = self.GenerarionFeatureROI()
        else:
            _, _, self.original_feature_map_array = self.LoadNiiData(self.feature_map_path, is_show_info=False)


    def GetIndexRangeInROI(self,roi_mask, target_value=1):
        if np.ndim(roi_mask) == 2:
            x, y = np.where(roi_mask == target_value)
            x = np.unique(x)
            y = np.unique(y)
            return x, y
        elif np.ndim(roi_mask) == 3:
            x, y, z = np.where(roi_mask == target_value)
            x = np.unique(x)
            y = np.unique(y)
            z = np.unique(z)
            return x, y, z

    def LoadNiiData(self, file_path, dtype=np.float32, is_show_info=False, is_flip=True, flip_log=[0, 0, 0]):
        image = sitk.ReadImage(file_path)
        data = np.asarray(sitk.GetArrayFromImage(image), dtype=dtype)

        show_data = np.transpose(data)
        show_data = np.swapaxes(show_data, 0, 1)

        # To process the flip cases
        if is_flip:
            direction = image.GetDirection()
            for direct_index in range(3):
                direct_vector = [direction[0 + direct_index * 3], direction[1 + direct_index * 3], direction[2 + direct_index * 3]]
                abs_max_point = np.argmax(np.abs(direct_vector))
                if direct_vector[abs_max_point] < 0:
                    show_data = np.flip(show_data, axis=direct_index)
                    flip_log[direct_index] = 1

        if is_show_info:
            print('Image size is: ', image.GetSize())
            print('Image resolution is: ', image.GetSpacing())
            print('Image direction is: ', image.GetDirection())
            print('Image Origion is: ', image.GetOrigin())

        return image, data, show_data

    def GenerarionFeatureROI(self):
      # print(sorted(self.original_feature_map_array.flatten(), reverse=True))
      x, y, z = self.GetIndexRangeInROI(self.original_roi_array)
      feature_roi = np.zeros_like(self.original_roi_array, dtype=np.float32)
      feature_roi[np.min(x):np.min(x) + self.original_feature_map_array.shape[0],
      np.min(y):np.min(y) + self.original_feature_map_array.shape[1],
      :self.original_feature_map_array.shape[2]]= self.original_feature_map_array
      return feature_roi

    def Normalize01(self,data, clip=0.0):
        new_data = np.asarray(data, dtype=np.float32)
        if clip > 1e-6:
            data_list = data.flatten().tolist()
            data_list.sort()
            new_data.clip(data_list[int(clip * len(data_list))], data_list[int((1 - clip) * len(data_list))])

        new_data = new_data - np.min(new_data)
        new_data = new_data / np.max(new_data)
        return new_data

    def ShowTransforedImage(self,store_path):
        feature_map_show_array = self.original_feature_map_array[:, :, self._max_roi_index]
        plt.title(self.feature_name)
        plt.imshow(feature_map_show_array, cmap='gray')
        plt.contour(self.original_roi_array[:, :, self._max_roi_index], colors='r')
        plt.axis('off')
        if store_path:
            plt.savefig(store_path+'_transformed_image.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.savefig(store_path+'_transformed_image.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0)

        plt.show()

    def ShowColorByROI(self, background_array, fore_array, roi_array, color_map, feature_name, threshold_value=1e-6, size=0, store_path='',
                       is_show=True):

        roi_center = [int(np.mean(np.where(roi_array == 1)[0])), int(np.mean(np.where(roi_array == 1)[1]))]

        if size:
            roi_array = roi_array[roi_center[0] - size:roi_center[0] + size,
                            roi_center[1] - size:roi_center[1] + size]

            background_array = background_array[roi_center[0] - size:roi_center[0] + size,
                            roi_center[1] - size:roi_center[1] + size]

            fore_array = fore_array[roi_center[0] - size:roi_center[0] + size,
                               roi_center[1] - size:roi_center[1] + size]


        if background_array.shape != roi_array.shape:
            print('Array and ROI must have same shape')
            return

        background_array = self.Normalize01(background_array)
        fore_array = self.Normalize01(fore_array)
        cmap = plt.get_cmap(color_map)
        rgba_array = cmap(fore_array)
        rgb_array = np.delete(rgba_array, 3, 2)

        index_roi_x, index_roi_y = np.where(roi_array < threshold_value)

        start_time = time.time()
        index_list = range(len(index_roi_x))
        for position_index in tqdm.tqdm(index_list):
            index_x, index_y = index_roi_x[position_index], index_roi_y[position_index]

            rgb_array[index_x, index_y, :] = background_array[index_x, index_y]
        print(time.time()-start_time)
        plt.imshow(rgb_array, cmap=color_map)
        plt.colorbar()
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.title(feature_name)

        if store_path:
            plt.savefig(store_path+'.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
            #plt.savefig(store_path+'.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0)

        if is_show:
            plt.show()

        plt.close()
        plt.clf()
        return rgb_array

    def Show(self, index='', color_map='rainbow', store_path=''):
        if not index:
            
            ### Coronal ###
            background_array_c = self.original_image_array[self._max_roi_index, :, :]
            fore_array_c = self.original_feature_map_array[self._max_roi_index, :, :]
            roi_c = self.original_roi_array[self._max_roi_index, :, :]
            if not os.path.exists(store_path + 'Coronal/'):
                os.makedirs(store_path + 'Coronal/')
            store_path_c = store_path + 'Coronal/' + (os.path.split(self.feature_map_path)[-1]).split('.')[0]
            
            
            ### Sagittal ###
            background_array_s = self.original_image_array[:, self._max_roi_index, :]
            fore_array_s = self.original_feature_map_array[:, self._max_roi_index, :]
            roi_s = self.original_roi_array[:, self._max_roi_index, :]
            if not os.path.exists(store_path + 'Sagittal/'):
                os.makedirs(store_path + 'Sagittal/')
            store_path_s = store_path + 'Sagittal/' + (os.path.split(self.feature_map_path)[-1]).split('.')[0]
            
            ### Axial ###
            background_array_a = self.original_image_array[:, :, self._max_roi_index]
            fore_array_a = self.original_feature_map_array[:, :, self._max_roi_index]
            roi_a = self.original_roi_array[:,:, self._max_roi_index]
            if not os.path.exists(store_path + 'Axial/' ):
                os.makedirs(store_path + 'Axial/' )
            store_path_a = store_path + 'Axial/' + (os.path.split(self.feature_map_path)[-1]).split('.')[0]
            

        else:
            background_array = self.original_image_array[:, :, index]
            fore_array = self.original_feature_map_array[:, :, index]
            roi = self.original_roi_array[:, :, index]

        self.ShowColorByROI(background_array_c, fore_array_c, roi_c, color_map=color_map, feature_name=self.feature_name,store_path=store_path_c, is_show=False)
        self.ShowColorByROI(background_array_s, fore_array_s, roi_s, color_map=color_map, feature_name=self.feature_name,store_path=store_path_s, is_show=False)
        self.ShowColorByROI(background_array_a, fore_array_a, roi_a, color_map=color_map, feature_name=self.feature_name,store_path=store_path_a, is_show=False)

def main():
    kinds = ['ARMS_NT','ARMS_T', 'FEP', 'C'] ## 
    for kind in kinds:
        path = './Projects/MRI/seg/' + kind
        path_roi= './Projects/MRI/ML/MyDataSeg/' + kind
        path_nrrd = './Projects/MRI/ML/Vis/seg_Maximum/seg_fea/' + kind
        store_path = r'./Projects/MRI/ML/Vis/seg_Maximum/seg_fea_map/'
        for index, f in enumerate( os.listdir(path)):
            image_path = os.path.join(path, f) 
            roi_path = os.path.join(path_roi, f)
            num = f.split('.')[0][-3:]
            f_nrrd = 'seg_' + kind + '_' + num + '_original_firstorder_Maximum.nrrd'
            feature_map_path = os.path.join(path_nrrd, f_nrrd)
            featuremapvisualization = FeatureMapVisualizition()
            featuremapvisualization.LoadData(image_path, roi_path, feature_map_path)
            dst_path = store_path + kind + '/'
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            featuremapvisualization.Show(color_map='rainbow', store_path=dst_path) # color_map='seismic'

    

if __name__ == '__main__':
    main()


