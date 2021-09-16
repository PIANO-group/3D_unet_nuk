import os
import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import time
import random
from config import data_path


def create_tf_dataset(n_train, seed, save=False, save_dir="", all=True, n_samples=""):
    os.chdir(data_path)
    ds_x_filepaths = glob.glob("*.nii")
    ds_y_filepaths = glob.glob(os.path.join("GT_masks", "*.nii"))

    random.seed(seed)
    z = list(zip(ds_x_filepaths, ds_y_filepaths))
    random.shuffle(z)
    ds_x_filepaths, ds_y_filepaths = zip(*z)
    ds_x_filepaths = list(ds_x_filepaths)
    ds_y_filepaths = list(ds_y_filepaths)

    if not all:
        assert type(n_samples) == int, "Please specify 'n_samples' in 'create_tf_dataset'."
        ds_x_filepaths = ds_x_filepaths[:n_samples]
        ds_y_filepaths = ds_y_filepaths[:n_samples]


    ds_x_train_niftis = [sitk.ReadImage(image_filepath) for image_filepath in ds_x_filepaths[:n_train+1]]
    ds_x_train_np = np.array([sitk.GetArrayFromImage(image_nifti) for image_nifti in ds_x_train_niftis])
    ds_x_train_np = np.transpose(ds_x_train_np, (0, 3, 2, 1))

    ds_y_train_niftis = [sitk.ReadImage(image_filepath) for image_filepath in ds_y_filepaths[:n_train+1]]
    ds_y_train_np = np.array([sitk.GetArrayFromImage(image_nifti) for image_nifti in ds_y_train_niftis])
    ds_y_train_np = np.transpose(ds_y_train_np, (0, 3, 2, 1))

    test_roi_paths = [os.path.join("tumor_ROIs", "tumor_ROI_" + name.split("_")[1] + ".nii") for name in ds_x_filepaths[n_train+1:]]
    test_rois_niftis = [sitk.ReadImage(roi_filepath) for roi_filepath in test_roi_paths]
    test_rois_np = np.array([sitk.GetArrayFromImage(roi_nifti) for roi_nifti in test_rois_niftis])
    test_rois_np = np.transpose(test_rois_np, (0, 3, 2, 1))
    ds_x_test_niftis = [sitk.ReadImage(image_filepath) for image_filepath in ds_x_filepaths[n_train+1:]]
    ds_x_test_np = np.array([sitk.GetArrayFromImage(image_nifti) for image_nifti in ds_x_test_niftis])
    # ds_x_test_np = np.multiply(ds_x_test_np, test_rois_np)
    ds_x_test_np = np.transpose(ds_x_test_np, (0, 3, 2, 1))

    # sitk.Show(sitk.GetImageFromArray(ds_x_test_np[0]))
    ds_y_test_niftis = [sitk.ReadImage(image_filepath) for image_filepath in ds_y_filepaths[n_train+1:]]
    ds_y_test_np = np.array([sitk.GetArrayFromImage(image_nifti) for image_nifti in ds_y_test_niftis])
    ds_y_test_np = np.transpose(ds_y_test_np, (0, 3, 2, 1))

    ds_y_train_names = [filepath.split(os.path.sep)[-1] for filepath in ds_y_filepaths[:n_train+1]]
    ds_y_test_names = [filepath.split(os.path.sep)[-1] for filepath in ds_y_filepaths[n_train+1:]]


    # tfds_train_niftis = tf.data.Dataset.from_tensor_slices(ds_train_np)
    tfds_train_np = {"train": tf.data.Dataset.from_tensor_slices((ds_x_train_np, ds_y_train_np)),
                     "test": tf.data.Dataset.from_tensor_slices((ds_x_test_np, ds_y_test_np)),
                     "x_train_names": ds_x_filepaths[:n_train+1],
                     "y_train_names": ds_y_train_names,
                     "x_test_names": ds_x_filepaths[n_train+1:],
                     "y_test_names": ds_y_test_names,
                     "test_rois_np": test_rois_np}

    # print(tfds_train_niftis)
    if save:
        assert save_dir != "", "Specify tf dataset save directory!"
        tf.data.experimental.save(tfds_train_np, save_dir)


    return tfds_train_np


def create_tf_dataset_predefined_splits(predefined_split_seed, fold, normalize=False, save=False, save_dir="", n_samples=""):
    with open(os.path.join(data_path, "splits", "seed_" + str(predefined_split_seed), "train_" + str(fold) + ".txt")) as file:
        ds_x_train_filepaths = file.read().splitlines()
    with open(os.path.join(data_path, "splits", "seed_" + str(predefined_split_seed), "test_" + str(fold) + ".txt")) as file:
        ds_x_test_filepaths = file.read().splitlines()

    ds_y_train_filepaths = [
        os.path.join(data_path, "GT_masks", "GT_mask_" + name.split("/")[-1].split("_")[1] + ".nii") for
        name in ds_x_train_filepaths]
    ds_y_test_filepaths = [
        os.path.join(data_path, "GT_masks", "GT_mask_" + name.split("/")[-1].split("_")[1] + ".nii") for
        name in ds_x_test_filepaths]

    ds_x_train_niftis = [sitk.ReadImage(image_filepath) for image_filepath in ds_x_train_filepaths]
    ds_x_train_np = np.array([sitk.GetArrayFromImage(image_nifti) for image_nifti in ds_x_train_niftis])
    ds_x_train_np = np.transpose(ds_x_train_np, (0, 3, 2, 1))

    ds_y_train_niftis = [sitk.ReadImage(GT_mask_filepath) for GT_mask_filepath in ds_y_train_filepaths]
    ds_y_train_np = np.array([sitk.GetArrayFromImage(GT_mask_nifti) for GT_mask_nifti in ds_y_train_niftis])
    ds_y_train_np = np.transpose(ds_y_train_np, (0, 3, 2, 1))

    test_roi_paths = [os.path.join(data_path, "tumor_ROIs", "tumor_ROI_" + name.split("/")[-1].split("_")[1] + ".nii") for name in ds_x_test_filepaths]
    test_rois_niftis = [sitk.ReadImage(roi_filepath) for roi_filepath in test_roi_paths]
    test_rois_np = np.array([sitk.GetArrayFromImage(roi_nifti) for roi_nifti in test_rois_niftis])
    test_rois_np = np.transpose(test_rois_np, (0, 3, 2, 1))

    ds_x_test_niftis = [sitk.ReadImage(image_filepath) for image_filepath in ds_x_test_filepaths]
    ds_x_test_np = np.array([sitk.GetArrayFromImage(image_nifti) for image_nifti in ds_x_test_niftis])
    # ds_x_test_np = np.multiply(ds_x_test_np, test_rois_np)
    ds_x_test_np = np.transpose(ds_x_test_np, (0, 3, 2, 1))

    # sitk.Show(sitk.GetImageFromArray(ds_x_test_np[0]))
    ds_y_test_niftis = [sitk.ReadImage(GT_mask_filepath) for GT_mask_filepath in ds_y_test_filepaths]
    ds_y_test_np = np.array([sitk.GetArrayFromImage(image_nifti) for image_nifti in ds_y_test_niftis])
    ds_y_test_np = np.transpose(ds_y_test_np, (0, 3, 2, 1))

    if normalize:
        ds_x_train_np = [(img-np.mean(img))/np.std(img) for img in ds_x_train_np]
        ds_x_test_np = [(img - np.mean(img)) / np.std(img) for img in ds_x_test_np]

    ds_y_train_names = [filepath.split(os.path.sep)[-1] for filepath in ds_y_train_filepaths]
    ds_y_test_names = [filepath.split(os.path.sep)[-1] for filepath in ds_y_test_filepaths]

    # tfds_train_niftis = tf.data.Dataset.from_tensor_slices(ds_train_np)
    tfds_train_np = {"train": tf.data.Dataset.from_tensor_slices((ds_x_train_np, ds_y_train_np)),
                     "test": tf.data.Dataset.from_tensor_slices((ds_x_test_np, ds_y_test_np)),
                     "x_train_names": ds_x_train_filepaths,
                     "y_train_names": ds_y_train_names,
                     "x_test_names": ds_x_test_filepaths,
                     "y_test_names": ds_y_test_names,
                     "test_rois_np": test_rois_np}

    # print(tfds_train_niftis)
    if save:
        assert save_dir != "", "Specify tf dataset save directory!"
        tf.data.experimental.save(tfds_train_np, save_dir)

    return tfds_train_np


def load_tf_dataset(name, overwrite=False, all=True, n_samples=""):
    os.chdir(data_path)
    dataset_path = os.path.join("tensorflow_dataset", name)

    if not os.path.isdir(dataset_path) or overwrite:
        if overwrite:
            print(f"Overwriting existing tf dataset in: {dataset_path}...")
        else:
            print(f"Creating new tf dataset and saving in: {dataset_path}...")
            os.makedirs(dataset_path)
        _ = create_tf_dataset(save=True, save_dir=dataset_path, all=all, n_samples=n_samples)
        print("...created.")

    print(f"Loading tf dataset from: {dataset_path}...")
    tf_dataset = tf.data.experimental.load(dataset_path)
    print(f"...done.")
    return tf_dataset


if __name__ == "__main__":
    print("hi")

    tic = time.time()
    tf_dataset = create_tf_dataset(n_train=188, normalize=True, all=False, n_samples=314)
    # tf_dataset = load_tf_dataset("original_10", all=False, n_samples=10)
    toc = time.time()

    print(f"Lasted {str(toc-tic)} seconds.")

    # print(tf_dataset)
    i = 0
    # for element in tf_dataset["train"].as_numpy_iterator():
    #     if i == 0:
    #         print(type(element))
    #         # print(element["x_train"].shape)
    #         # print(element["x_train_names"].decode())
    #         # print(element["y_train_names"].decode())
    #         sitk.Show(sitk.GetImageFromArray(np.transpose(element[0], (2, 1, 0))))
    #         sitk.Show(sitk.GetImageFromArray(np.transpose(element[1], (2, 1, 0))))
    #     i += 1
    # print(f"Number of training images: {str(i)}")

    i=0
    # for element in tf_dataset["validation"].as_numpy_iterator():
    #     if i == 0:
    #         print(type(element))
    #         # print(element["x_train"].shape)
    #         # print(element["x_train_names"].decode())
    #         # print(element["y_train_names"].decode())
    #         sitk.Show(sitk.GetImageFromArray(np.transpose(element[0], (2, 1, 0))))
    #         sitk.Show(sitk.GetImageFromArray(np.transpose(element[1], (2, 1, 0))))
    #     i += 1

    # print(f"Number of validation images: {str(i)}")

    print(tf_dataset["y_train_names"])
    for element in tf_dataset["y_train_names"].take(1):
        if i == 0:
            print(element)
            # print(element["x_train"].shape)
            # print(element["x_train_names"].decode())
            # print(element["y_train_names"].decode())
            # sitk.Show(sitk.GetImageFromArray(np.transpose(element[0], (2, 1, 0))))
            # sitk.Show(sitk.GetImageFromArray(np.transpose(element[1], (2, 1, 0))))
        i += 1