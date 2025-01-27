import csv

inp = './Projects/MRI/ML/Fea_FD/FD25_30/ARMS_T.csv'
out = './Projects/MRI/ML/Fea_test/FEP_HC/ARMS_T_FD25_30_2.csv'


# for FEP vs CHR_NT
# FD15_50
choice = ['diagnostics_Image-interpolated_Minimum',
'diagnostics_Image-interpolated_Maximum',
'diagnostics_Mask-interpolated_Maximum',
'original_shape_MinorAxisLength',
'original_shape_SurfaceArea',
'original_firstorder_Maximum',
'original_glcm_MaximumProbability',
'original_glrlm_RunPercentage',
'original_glszm_LargeAreaEmphasis',
'original_glszm_LargeAreaLowGrayLevelEmphasis',
'original_glszm_ZonePercentage',
'original_gldm_GrayLevelVariance',
'log-sigma-3-0-mm-3D_glrlm_GrayLevelNonUniformity',
'log-sigma-3-0-mm-3D_glszm_GrayLevelVariance',
'log-sigma-3-0-mm-3D_gldm_LargeDependenceLowGrayLevelEmphasis',
'wavelet-LLH_firstorder_10Percentile',
'wavelet-LLH_firstorder_Skewness',
'wavelet-LLH_glszm_LowGrayLevelZoneEmphasis',
'wavelet-LHH_glcm_Imc1',
'wavelet-LHH_glcm_MaximumProbability',
'wavelet-LHH_glszm_ZonePercentage',
'wavelet-HLL_firstorder_RobustMeanAbsoluteDeviation',
'wavelet-HLL_glcm_Idmn',
'wavelet-HLL_glszm_HighGrayLevelZoneEmphasis',
'wavelet-HLL_glszm_SmallAreaHighGrayLevelEmphasis',
'wavelet-HLH_glcm_InverseVariance',
'wavelet-HHL_firstorder_RootMeanSquared',
'wavelet-HHH_glrlm_RunLengthNonUniformity',
'wavelet-LLL_firstorder_RobustMeanAbsoluteDeviation',
'wavelet-LLL_glszm_LargeAreaLowGrayLevelEmphasis',
'wavelet-LLL_gldm_DependenceNonUniformityNormalized',
'wavelet-LLL_gldm_LargeDependenceLowGrayLevelEmphasis',
'wavelet-LLL_gldm_SmallDependenceEmphasis']

idx = []

with open(inp, 'r', newline ='') as fi:
    with open(out, 'a', newline = '') as fo:
        reader = csv.reader(fi)
        writer = csv.writer(fo)
        header = next(reader, None)
        for i in range(len(header)):
            if header[i] in choice:
                idx.append(i)
        for row in reader:
            data = []
            h = []
            for k in idx:
                data.append(row[k])
            writer.writerow(data)
            