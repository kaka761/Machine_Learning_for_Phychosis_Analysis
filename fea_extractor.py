import os
import pandas as pd
import radiomics  #这个库专门用来提取特征
from radiomics import featureextractor

# generate feature files and save as FEP.csv & C.csv

#kinds = ['ARMS_NT','ARMS_T', 'FEP']#, 'C']
kinds = ['ARMS_T']
# para_path = './Projects/MRI/ML/MR_1mm.yaml'
### for test ###
para_path = './Projects/MRI/ML/Fea_test/MR_1mm_FEP_C.yaml' # feature extraction configuration file
extractor = featureextractor.RadiomicsFeatureExtractor(para_path)
for kind in kinds:
    print("{}:start feature extracting".format(kind))
    features_dict = dict()
    df = pd.DataFrame()
    path = './Projects/MRI/MRI_FD/results_25_30/' + kind
    path_lab = './Projects/MRI/ML/MyDataFD/FD_25_30/' + kind
    # initialize feature extractor
    for index, f in enumerate( os.listdir(path)):
        #for f in os.listdir(os.path.join(path, folder)):
        ori_path = os.path.join(path, f)
        lab_path = os.path.join(path_lab, f)
        print(ori_path)
        print(lab_path)
        features = extractor.execute(ori_path, lab_path)  # extract feature
        for key, value in features.items():  # output feature
            if 'diagnostics_Versions' in  key or 'diagnostics_Configuration' in key: # remove commen features
                continue
            features_dict[key] = value
        df = df.append(pd.DataFrame.from_dict(features_dict.values()).T,ignore_index=True)
        print(index)
    df.columns = features_dict.keys()
    df.to_csv('{}.csv'.format(kind),index=0)
print("Done")