import os
import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import tkinter
import sys
import copy
import statistics
from tkinter import filedialog
from tqdm import tqdm

from config import data_path, sessions_path, settings
from preprocessing import create_tf_dataset, create_tf_dataset_predefined_splits
from losses import soft_dice_loss
from metrics import dice, dice_single
from unet3D import unet3D


def inference(mode, model_filepath, predefined_split_seed=1, fold=1, multiple_thresholds=True):
    os.chdir(sessions_path)
    # model_path = model_filepath.rsplit("/", 2)[0]
    # model_filename = model_filepath.split("/")[-1].split(".")[0]

    model_path = os.path.dirname(os.path.dirname(model_filepath))
    model_filename = os.path.basename(model_filepath).split(".")[0]

    predictions_dir = "predictions"
    # session_name = model_path.split("/")[-1]
    session_name = os.path.basename(model_path)

    pred_session_name = session_name + "_" + model_filename

    predictions_path = os.path.join(sessions_path, predictions_dir, pred_session_name)
    if not os.path.isdir(predictions_path):
        os.makedirs(predictions_path)

    prob_maps_path = os.path.join(predictions_path, "prob_maps")
    if not os.path.isdir(prob_maps_path):
        os.makedirs(prob_maps_path)

    model = unet3D()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00003),
                  loss=soft_dice_loss,
                  metrics=['accuracy',
                           tf.keras.metrics.MeanIoU(num_classes=2),
                           dice])

    model.summary()

    print(f"Loading weights from {model_filepath}")
    model.load_weights(os.path.join(model_path, "weights", "weights.h5"))

    BATCH_SIZE = 4
    BUFFER_SIZE = 500
    if mode == "standard":
        tf_dataset = create_tf_dataset(n_train=settings["n_train"], seed=settings["seed"])
    elif mode == "predefined_splits":
        tf_dataset = create_tf_dataset_predefined_splits(predefined_split_seed=predefined_split_seed, fold=fold)
    test_ds = tf_dataset["test"]
    test_names_path = tf_dataset["x_test_names"]
    test_names = [os.path.basename(filepath) for filepath in test_names_path]
    test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.cache().batch(BATCH_SIZE)
    test_rois_np = tf_dataset["test_rois_np"]

    gt_masks = []
    for i, (image, gt) in enumerate(test_ds.as_numpy_iterator()):
                if i == 0:
                    prob_masks = model.predict(image, verbose=1)
                    gt_masks = gt

                else:
                    prob_masks = np.append(prob_masks, model.predict(image, verbose=1), axis=0)
                    gt_masks = np.append(gt_masks, gt, axis=0)

    prob_masks = prob_masks[:, :, :, :, 0]
    print(type(prob_masks))
    # reference_img = sitk.ReadImage("t_001_b_009_simulated.nii")
    # reference_img = sitk.ReadImage(os.path.join(data_path, "t_001_b_358_simulated.nii"))

    niftis_prob = [sitk.GetImageFromArray(np.transpose(image_np, (2, 1, 0))) for image_np in prob_masks]
    print("Saving probability masks...")
    for i in tqdm(range(len(niftis_prob))):
        img = sitk.ReadImage(os.path.join(data_path, test_names[i]))
        niftis_prob[i].CopyInformation(img)
        prob_map_name = "pred_" + test_names[i]
        sitk.WriteImage(niftis_prob[i], os.path.join(prob_maps_path, prob_map_name))
    print("...done.")

    scores_dict = {}

    mean_dice = 0
    if multiple_thresholds:
        thresholds = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.95, 0.92, 0.97]
    else:
        thresholds = [0.5]
    for threshold in thresholds:
        print(f"Applying threshold {threshold}...")
        pred_masks = copy.deepcopy(prob_masks)
        pred_masks[pred_masks > threshold] = 1
        pred_masks[pred_masks <= threshold] = 0
        pred_masks = np.multiply(pred_masks, test_rois_np)
        # sitk.Show(sitk.GetImageFromArray(pred_masks[0]))

        dice_scores = []
        for i in range(len(pred_masks)):
            dice_scores.append(dice_single(gt_masks[i], pred_masks[i]))

        print(len(dice_scores))
        mean_dice = statistics.mean(dice_scores)
        std_p = statistics.pstdev(dice_scores)
        threshold_dir = "threshold_" + str(threshold) + "_dice_" + str(int(mean_dice*100))
        threshold_path = os.path.join(predictions_path, threshold_dir)
        if not os.path.isdir(threshold_path):
            os.makedirs(threshold_path)

        prediction_names = []
        niftis_pred = [sitk.GetImageFromArray(np.transpose(image_np, (2, 1, 0))) for image_np in pred_masks]
        print("Saving predictions...")
        for i in tqdm(range(len(niftis_pred))):
            img = sitk.ReadImage(os.path.join(data_path, test_names[i]))
            niftis_pred[i].CopyInformation(img)
            prediction_name = "pred_" + test_names[i].split(".")[0] + "_dice_" + str(int(dice_scores[i]*100)) + ".nii"
            sitk.WriteImage(niftis_pred[i], os.path.join(threshold_path, prediction_name))
            prediction_names.append(prediction_name)
        print("...done.")

        dice_scores_sorted, prediction_names_sorted = zip(*sorted(zip(dice_scores, prediction_names)))
        dice_scores_sorted = [round(dice_score, 3) for dice_score in dice_scores_sorted]

        content = ["Mean: \t\t\t" + str(mean_dice) + "\n"]
        content.append("STD: \t\t\t" + str(std_p) + "\n")

        for i in range(len(prediction_names_sorted)):
            content.append(prediction_names_sorted[i] + "\t\t" + str(dice_scores_sorted[i]) + "\n")
        print(content)

        with open(os.path.join(threshold_path, "test.txt"), "w") as file:
            file.writelines(content)

        if threshold == 0.5:
            dice_scores_sorted_half = dice_scores_sorted
            prediction_names_sorted_half = prediction_names_sorted
            mean_dice_half = mean_dice
            std_half = std_p

    scores_dict["dice_scores"] = dice_scores_sorted_half
    scores_dict["prediction_names"] = prediction_names_sorted_half
    scores_dict["mean_dice"] = mean_dice_half
    scores_dict["std"] = std_half

    return scores_dict

if __name__ == "__main__":
    os.chdir(sessions_path)
    root = tkinter.Tk()
    root.withdraw()
    model_filepath = filedialog.askopenfilename(parent=root, initialdir="saved_models", title="Choose saved model file")
    if model_filepath == "":
        print("No file selected.")
        print("Aborting...")
        sys.exit()
    inference(settings["mode"], model_filepath, predefined_split_seed=1, fold=1, multiple_thresholds=True)