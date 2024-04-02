from src.Model.Regressor import Regressor
from src.Model.Classifier import Classifier
import pandas as pd
import numpy as np
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
    # pmml_path = "../../shared/pmml_models/C-1.pmml"
    # test_ds_pth = "../../shared/lags/C-1.csv"
    # df = pd.read_csv(test_ds_pth, sep =",")
    # outcomes = df["Outcome"].to_numpy()
    # df.drop("Outcome", axis=1, inplace=True)
    # test_ds = df.values
    # regressor = Regressor(ncpus = 4, use_espresso=False)
    # regressor.parse(model_source = pmml_path)
    # out_values = []
    # for x in test_ds:
    #     trees_outs = []
    #     for t in regressor.trees:
    #         trees_outs.append(t.visit(x))
    #     out_values.append(np.mean(np.array(trees_outs)))  
    # assert len(outcomes) == len(out_values) 
    # preds = regressor.predict(test_ds)
    # for ind, o in enumerate(out_values):
    #     print(f"idx {ind} Out {o} Out pred {preds[ind]}")

    # # Test XGB
    #pmml_path = "../../shared/NASA_PMML_XGB/C-1.pmml"
    job_path = "../../shared/NASA_JOB/C-1.joblib"
    test_ds_pth = "../../shared/lags/C-1.csv"
    df = pd.read_csv(test_ds_pth, sep =",")
    outcomes = df["Outcome"].to_numpy()
    df.drop("Outcome", axis=1, inplace=True)
    test_ds = df.values
    regressor = Regressor(ncpus = 4, use_espresso=False, learning_rate=0.2)
    regressor.parse(model_source = job_path)
    # out_values = []
    # for x in test_ds:
    #     trees_outs = []
    #     for t in regressor.trees:
    #         trees_outs.append(t.visit(x))
    #     out_values.append(np.mean(np.array(trees_outs)))  
    # assert len(outcomes) == len(out_values) 
    # preds = regressor.predict(test_ds)
    # for ind, o in enumerate(out_values):
    #     print(f"idx {ind} Out {o} Out pred {preds[ind]}")
