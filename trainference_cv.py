import os
import statistics

from config import data_path, sessions_path, settings
from train import train
from inference import inference


def trainference_cv(predefined_split_seed, fold_cv, epochs=settings["epochs"]):
    os.chdir(sessions_path)

    session_names = []
    dice_scores = []
    prediction_names = []
    mean_dice_scores_foldwise = []
    stds_foldwise = []

    for i in range(1,fold_cv+1):

        # Run training and get session name
        session_name = train(mode=settings["mode"], predefined_split_seed=predefined_split_seed, fold=i, epochs=epochs)
        session_names.append(session_name)

        # Reconstruct model filepath from threshold 0.5
        model_filepath = os.path.join(sessions_path, "saved_models", session_name, "weights", "weights.h5")
        scores_dict = inference(mode=settings["mode"], model_filepath=model_filepath, predefined_split_seed=predefined_split_seed, fold=i)

        dice_scores.extend(scores_dict["dice_scores"])
        prediction_names.extend(scores_dict["prediction_names"])
        mean_dice_scores_foldwise.append(scores_dict["mean_dice"])
        stds_foldwise.append(scores_dict["std"])

    dice_scores_sorted, prediction_names_sorted = zip(*sorted(zip(dice_scores, prediction_names)))

    overall_dice = statistics.mean(dice_scores_sorted)
    overall_std = statistics.pstdev(dice_scores_sorted)

    output_filename = "cv_" + str(fold_cv) + "_fold_predef_seed_" + str(predefined_split_seed) + ".txt"
    j = 1
    while (os.path.isfile(output_filename)):
        if j == 1:
            output_filename = output_filename.split(".")[0] + "_" + str(j) + ".txt"
        else:
            output_filename = output_filename.rsplit("_", 1)[0] + "_" + str(j) + ".txt"
        j += 1

    content = [str(fold_cv) + " fold cv on sessions:\n"]

    for i in range(len(session_names)):
        content.append(session_names[i] + "\t\t" + "mean_dice: " + str(mean_dice_scores_foldwise[i]) + "\t\t" + "std: " + str(stds_foldwise[i]) + "\n")
    content.append("\n")

    content.append("Over all images of every test set: \n")
    content.append("dice = \t" + str(overall_dice) + "\n")
    content.append("std = \t" + str(overall_std) + "\n")
    content.append("\n")

    for i in range(len(prediction_names_sorted)):
        content.append(prediction_names_sorted[i] + "\t\t" + str(dice_scores_sorted[i]) + "\n")

    with open(output_filename, "w") as file:
        file.writelines(content)


if __name__ == "__main__":
    predefined_split_seed = 2
    fold_cv = 5

    trainference_cv(predefined_split_seed=predefined_split_seed, fold_cv=fold_cv)





