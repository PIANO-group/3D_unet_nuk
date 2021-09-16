import os

from config import data_path, sessions_path, settings
from train import train
from inference import inference

predefined_split_seed = 1
fold = 1

# Run training and get session name
session_name = train(mode=settings["mode"], predefined_split_seed=predefined_split_seed, fold=fold, epochs=settings["epochs"])

# Reconstruct model filepath from threshold 0.5
model_filepath = os.path.join(sessions_path, "saved_models", session_name, "weights", "weights.h5")

inference(settings["mode"], model_filepath, predefined_split_seed=predefined_split_seed, fold=fold, multiple_thresholds=True)