# 3D_unet_nuk

## General information

* Trained weights are saved in "saved_models", prob maps und segmentation masks in "predictions".
* Set multiple_thresholds to "True" to use different thresholds for the prob maps for inference
* Only used on 64x64x32 dataset (first create a cropped dataset with repo "dataset_functions")
* Make sure to have dirs "GT_masks" and "tumor_ROIs" in your dataset (see step 2 in deepMedic_nuk repo)
* tumor_ROIs are used to compute the dice scores only within the confinement (to be able to compare results with SegmentationApplicationValidation)
* To use predefined splits, set settings["mode"] to "predefined_splits" in config.py and make sure to have the .txt-files defining the splits in dir "splits" in your data_path (copy the contents of e.g. the dir "heterogen" (dirs "seed_1" to "seed_5") found in the dir "make_cv_splits" from the databox "Conventional_methods_and_make_cv_splits" into an empty dir named "splits" in your data_path).
* So far, CV was only used in mode "predefined_splits"
  

## Steps

1. Set data_path in config.py, e.g. to "D:\datasets_segmentation\FET_20_40_TBR_348_homogen_v1_646432", which contains simulated images (e.g. "t_001_b_358_simulated.nii")
   
1. Choose a mode ("standard" or "predefined_splits") in config.py.<br>
In standard mode, set you n_train and a random seed for splitting dataset into training and test set.

1. Run "trainference.py" (ignore "predefined_splits" and "fold" when using "standard" mode)

1. To use 5-fold CV, set mode to "predefined splits" and set "predefined_split_seed" and "fold_cv" in "trainference_cv.py"<br>
   (fold_cv = 5 for 5-fold CV). Find the results in a .txt file (e.g. "cv_5_fold_predef_seed_1_1.txt", last number is just a dummy counter)
   
1. 5-repeated 5-fold CV: After performing 5 times "trainference_cv.py" (predefined_seed=1, ..., predefined_seed=5), run "repeated_cv_single_images.py" and in each filedialog, choose a CV results .txt-file (see step 4).
This outputs e.g. "5_repeated_5_fold.txt"
   
test
   
