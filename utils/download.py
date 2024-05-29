import logging
import os
import shutil


def Download_Checkpoint():
    if not os.path.exists("./checkpoints/"): os.mkdir("./checkpoints/")
    if not os.path.exists(os.path.join("./checkpoints/", "vallex-checkpoint.pt")):
        import wget
        try:
            print(
                "Downloading model from https://huggingface.co/datasets/yuossfalaa/Vall-E_X_Training_Data/blob/main/latest-checkpoint.pt ...")
            # download from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt to
            # ./checkpoints/vallex-checkpoint.pt
            wget.download("https://huggingface.co/datasets/yuossfalaa/Vall-E_X_Training_Data/blob/main/latest-checkpoint.pt",
                          out="./checkpoints/vallex-checkpoint.pt", bar=wget.bar_thermometer)
        except Exception as e:
            logging.info(e)
            raise Exception(
                "\n Model weights download failed, please go to "
                "'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'"
                "\n manually download model weights and put it to {} .".format(os.getcwd() + "\checkpoints"))


def Download_AR_G2P_Checkpoint():
    if not os.path.exists("./utils/g2p/Arabic_G2P_Model/checkpoints/"): os.mkdir(
        "./utils/g2p/Arabic_G2P_Model/checkpoints/")
    if not os.path.exists("./utils/g2p/Arabic_G2P_Model/checkpoints/best_model.pt"):
        import gdown
        try:
            print("Downloading AR G2P model")
            url = "https://drive.google.com/file/d/1ZXnh9_CznwRL_2-vwvvNUUJxePCGK2H8/view?usp=sharing"
            output = "best_model.pt"
            gdown.download(url=url, output=output,fuzzy=True)
            shutil.move("./best_model.pt", "./utils/g2p/Arabic_G2P_Model/checkpoints/best_model.pt")
        except Exception as e:
            logging.info(e)
            raise Exception(
                "\n Model weights download failed, please go to "
                "' '"
                "\n manually download model weights and put it to {} .".format(
                    os.getcwd() + "./utils/g2p/Arabic_G2P_Model/checkpoints/"))


def DownloadAllNeededResources():
    Download_Checkpoint()
    Download_AR_G2P_Checkpoint()

