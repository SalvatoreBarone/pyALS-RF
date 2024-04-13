from src.Model.Regressor import Regressor
from src.Model.Classifier import Classifier
import pandas as pd
import numpy as np
from src.HDLGenerators.HDLGenerator import HDLGenerator

def load_errors(test_set, predictions_path: str = "", lag_size:int = 250):
    try:
        with open(predictions_path, 'r') as file:
            lines = file.read()
            time_series_listed = [float(line) for line in lines.split()]
    except FileNotFoundError:
        print(f"The errors file {predictions_path} was not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
    ts_list = test_set[test_set.columns[0]].tolist()
    ts_list = ts_list[lag_size:]
    # print(f"Predictions {len(time_series_listed)} , Timestamps {len(ts_list)} Originals {len(test_set[test_set.columns[0]].tolist())}")
    return pd.DataFrame({test_set.columns[0]: ts_list,test_set.columns[1]:time_series_listed })

if __name__ == "__main__":
    job_path = "../../shared/c1_xgb.joblib"
    test_ds_pth = "../../shared/lags/C-1.csv"
    xgb_original_preds_pth = "../../shared/C-1-XGB.txt"
    regressor = Regressor(ncpus = 4, use_espresso = True, learning_rate = 0.1)
    model_features = [{"name" : f"f{i}", "type":"double"}  for i in range(0,250)]
    regressor.parse(model_source = job_path, dataset_description = model_features)
    exact_luts_dbs, exact_luts_bns, exact_ffs_dbs = HDLGenerator.get_resource_usage(classifier = regressor)
    print(f" DB Luts {exact_luts_dbs} DB FF {exact_ffs_dbs} BN Luts {exact_luts_bns}")
    # print("********************************")
    # # Get the dump
    # regressor.trees[0].dump()
    # # Extract the boolean networks
    # bn = regressor.trees[0].boolean_networks
    # for b in bn:
    #     print(f"SP {b['sop']} CN {b['class']}")
    # dbs = HDLGenerator.get_dbs(regressor.trees[0])
    # print(dbs)