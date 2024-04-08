from src.Model.Regressor import Regressor
from src.Model.Classifier import Classifier
import pandas as pd
import numpy as np

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
    # Test classifier 
    # pmml_path = "./example/statlog_segment/rf5/rf_5.pmml"
    # test_ds_pth = "./example/statlog_segment/rf5/test_dataset_4_pyALS-rf.csv"
    # df = pd.read_csv(test_ds_pth, sep =";")
    # outcomes = df["Outcome"].to_numpy()
    # df.drop("Outcome", axis=1, inplace=True)
    # test_ds = df.values
    # classifier = Classifier(ncpus = 1, use_espresso = False)
    # classifier.parse(model_source = pmml_path)
    # # print(f"Shape {np.shape(test_ds)} Len {len(np.shape(test_ds))}")
    # # print(f"Type {type(test_ds)}")
    # # for t in test_ds:
    # #     print(classifier.trees[0].visit(t))
    # # print(f"Number of classes {len(classifier.model_classes)}")
    # # exit(1)
    # # x = classifier.predict(x_test = test_ds)
    # # s = [np.argmax(n) for n in x]
    # # assert len(s) == len(outcomes)
    # # for pred, real in zip(s, outcomes):
    # #     print(f" Pred {pred} and real {real}")

    #    Test regressor
    pmml_path = "../../shared/c1_rf.pmml"
    test_ds_pth = "../../shared/lags/C-1.csv"
    rf_original_preds_pth = "../../shared/C-1-RF.txt"
    with open(rf_original_preds_pth, 'r') as file:
            lines = file.read()
            rf_original_preds = [float(line) for line in lines.split()]
    df = pd.read_csv(test_ds_pth, sep =",")
    outcomes = df["Outcome"].to_numpy()
    df.drop("Outcome", axis=1, inplace=True)
    test_ds = df.values
    regressor = Regressor(ncpus = 4, use_espresso=False)
    regressor.parse(model_source = pmml_path)
    out_values = []
    for x in test_ds:
        trees_outs = []
        for t in regressor.trees:
            trees_outs.append(t.visit(x))
        out_values.append(np.mean(np.array(trees_outs)))  
    # assert len(outcomes) == len(out_values) 
    rf_preds = regressor.predict(test_ds)
    #for ind, o in enumerate(out_values):
    #    print(f"idx {ind} Out {o} Out pred {rf_original_preds[ind]} pred predict {rf_preds[ind]}")
    #exit(1)
    # # Test XGB
    #pmml_path = "../../shared/NASA_PMML_XGB/C-1.pmml"
    job_path = "../../shared/c1_xgb.joblib"
    test_ds_pth = "../../shared/lags/C-1.csv"
    xgb_original_preds_pth = "../../shared/C-1-XGB.txt"
    with open(xgb_original_preds_pth, 'r') as file:
        lines = file.read()
        xgb_original_preds = [float(line) for line in lines.split()]
    df = pd.read_csv(test_ds_pth, sep =",")
    outcomes = df["Outcome"].to_numpy()
    df.drop("Outcome", axis=1, inplace=True)
    test_ds = df.values
    regressor = Regressor(ncpus = 4, use_espresso = False, learning_rate = 0.1)
    model_features = [{"name" : f"f{i}", "type":"double"}  for i in range(0,250)]
    regressor.parse(model_source = job_path, dataset_description = model_features)
    # Test the evaluation
    xgb_preds = regressor.predict(test_ds)
    for idx, (rf_p,xgb_p) in enumerate(zip(rf_preds, xgb_preds)):
        print(f" Idx {idx} RF_P: {rf_p} RF_OR {rf_original_preds[idx]} XGB_P: {xgb_p} XGB_OR: {xgb_original_preds[idx]}")

