import ml_collections
import copy
import os

CONFIG = ml_collections.ConfigDict({
    "cnn_model_save_path": "model/MCNN/model_",
    "gcn_model_save_path": "model/AGCN/model_",
    # "model_save_path": "model/model_",
    "loss_save_path": "model/loss/loss_",
    "test_result_path": "model/test/test_",
})

def get_config():
    config = copy.deepcopy(CONFIG)
    return config
