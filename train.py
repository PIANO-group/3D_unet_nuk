import os
import tensorflow as tf
from datetime import datetime
import pandas as pd

from config import sessions_path, settings
from preprocessing import create_tf_dataset, create_tf_dataset_predefined_splits
from unet3D import unet3D, dropout, batch_norm_before_act, l2_reg, regularizer
from losses import soft_dice_loss
from metrics import dice


def train(mode, predefined_split_seed=1, fold=1, epochs=settings["epochs"]):
    # "predefined_split_seed" and "fold" are only used for mode "predefined_splits"


    if not os.path.isdir(sessions_path):
        os.makedirs(sessions_path)
    os.chdir(sessions_path)

    print(os.getcwd())

    if mode == "standard":
        tf_dataset = create_tf_dataset(n_train=settings["n_train"], seed=settings["seed"])
    elif mode == "predefined_splits":
        tf_dataset = create_tf_dataset_predefined_splits(predefined_split_seed, fold)

    train_ds = tf_dataset["train"]
    test_ds = tf_dataset["test"]
    print(train_ds.element_spec)
    # x_train_names = tf_dataset["x_train_names"]
    # y_train_names = tf_dataset["y_train_names"]
    # x_test_names = tf_dataset["x_test_names"]
    # y_test_names = tf_dataset["y_test_names"]

    # model = unet()
    # model.summary

    model3D = unet3D()
    model3D.summary(line_length=150)

    # model3D_small = unet3D_small()
    # model3D_small.summary(line_length=150)

    # 314
    # tf.keras.optimizers.Adam(learning_rate=0.0001) -> 0.9 dice after 40 epochs
    # tf.keras.optimizers.RMSprop(learning_rate=0.0001) -> same

    # 188
    # tf.keras.optimizers.Adam(learning_rate=0.00001) -> 0.78 train_dice and 0.61 validation_dice after 40 epochs
    # tf.keras.optimizers.Adam(learning_rate=0.00003) -> 0.96 train_dice and 0.85 validation_dice after 200 epochs, check

    # learning_rate = 0.00003
    learning_rate = 0.0001
    optimizer = "rmsprop"
    loss_str = "soft_dice_loss"
    # loss_str = "cross-entropy"
    # loss_str = "iou_loss"

    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = soft_dice_loss
    # loss = iou_loss

    model3D.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                    loss=loss,
                    metrics=['accuracy',
                             tf.keras.metrics.MeanIoU(num_classes=2),
                             dice])

    BUFFER_SIZE = 500
    BATCH_SIZE = 4
    train_ds.batch(BATCH_SIZE)
    train_dataset = train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    now = datetime.now()
    if mode == "standard":
        saved_model_dir = now.strftime("%y%m%d") + "_" + now.strftime("%H%M")
    elif mode == "predefined_splits":
        saved_model_dir = now.strftime("%y%m%d") + "_" + now.strftime("%H%M") + "_" + "predef_seed_" + str(predefined_split_seed) + "_" + str(fold)
    saved_model_path = os.path.join(sessions_path, "saved_models", saved_model_dir)
    # if not os.path.isdir(saved_model_path):
    #     os.makedirs(saved_model_path)

    saved_weights_dir = "weights"
    saved_weights_path = os.path.join(saved_model_path, saved_weights_dir)
    print(saved_weights_path)
    if not os.path.isdir(saved_weights_path):
        os.makedirs(saved_weights_path)


    content = ["Learning rate: \t" + str(learning_rate) + "\n"]
    content.append("Optimizer: \t" + str(optimizer) + "\n")
    content.append("Loss: \t" + str(loss_str) + "\n")
    content.append("Buffer size: \t" + str(BUFFER_SIZE) + "\n")
    content.append("Batch size: \t" + str(BATCH_SIZE) + "\n\n")
    content.append("Batch norm: \t" + str(batch_norm_before_act) + "\n\n")
    if regularizer != None:
        content.append("L2 regularization: \t" + str(l2_reg) + "\n\n")
    else:
        content.append("L2 regularization: \t" + str(regularizer) + "\n\n")
    content.append("Dropout: \t" + str(dropout) + "\n\n")

    content.append("Mode: \t" + mode + "\n")
    if mode == "standard":
        content.append("Number of training images: \t" + str(settings["n_train"]) + "\n")
        content.append("seed: \t" + str(settings["seed"]) + "\n")
    elif mode == "predefined_splits":
        content.append("Predefined split seed: \t" + str(predefined_split_seed) + "\n")
        content.append("Fold: \t\t\t" + str(fold) + "\n\n")


    with open(os.path.join(saved_model_path, "config.txt"), "w") as file:
        file.writelines(content)

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=os.path.join(saved_weights_path, "cp.h5"),
    #     verbose=1,
    #     save_weights_only=True,
    #     save_freq=47*10)

        # filepath=os.path.join(saved_weights_path, "cp-{epoch:02d}.h5"),
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(saved_weights_path, "best_checkpoint.h5"),
        verbose=1,
        monitor="val_dice",
        mode="max",
        save_best_only = True,
        save_weights_only=True)

    model_history = model3D.fit(train_dataset,
                                epochs=epochs,
                                validation_data=test_ds,
                                callbacks=[cp_callback])


    print(model_history.history.keys())

    df = pd.DataFrame(model_history.history)
    df.to_excel(os.path.join(saved_model_path, "history.xlsx"))

    model3D.save_weights(os.path.join(saved_weights_path, "weights.h5"), save_format="h5")


    return saved_model_dir


if __name__ == "__main__":
    # "predefined_split_seed" and "fold" are only used for mode "predefined_splits"
    train(mode=settings["mode"], predefined_split_seed=1, fold=1, epochs=settings["epochs"])