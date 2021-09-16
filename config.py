import os


# data_path = r"C:\datasets_segmentation\FET_20_40_TBR_314_heterogen_v1_test"
# data_path = r"C:\datasets_segmentation\FET_20_40_TBR_348_heterogen_v1_646432_U-net"
# data_path = r"C:\datasets_segmentation\FET_20_40_TBR_348_heterogen_v1_646432"
# data_path = r"D:\datasets_segmentation\FET_20_40_TBR_348_heterogen_v1_646432"
data_path = r"D:\datasets_segmentation\FET_20_40_TBR_348_homogen_v1_646432"


sessions_dir = "unet3d"
sessions_path = os.path.join(data_path, sessions_dir)

# set mode to "predefined_splits" for using splits defined by .txt-files found in "splits" dir in data_path,
# otherwise use "standard
settings = {"mode": "predefined_splits",
            "epochs": 2}

if settings["mode"] == "standard":
    settings["n_train"] = 278
    settings["seed"] = 1