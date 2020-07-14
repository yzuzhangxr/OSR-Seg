import dicom2nifti
original_dicom_directory = 'D:/testdata/images/image_0001'
output_file = 'result.nii'
dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=True)
