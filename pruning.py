import argparse
import pandas as pd
import joblib
import numpy as np
import os
from src.Flows.GREPSK.LossBasedGREPSK import LossBasedGREPSK
from src.Flows.GREPSK.ResiliencyBasedGREPSK import ResiliencyBasedGREPSK
import time
import logging

def run_loss_based_grep(model, cost_criterion,  X, y, pruning_fraction, validation_fraction, max_loss, pruning_path, report_path):
    trimmer = LossBasedGREPSK(classifier = model, pruning_set_fraction = pruning_fraction, max_loss = max_loss)
    trimmer.split_pruning_validation_set(X,y, validation_fraction)
    trimmer.trim_fixed(cost_criterion)
    trimmer.store_prunign_conf(pruning_path)
    trimmer.dump_report(report_path)

def run_redundancy_based_grep(model, cost_criterion,  X, y, pruning_fraction, validation_fraction, min_eta, pruning_path, report_path):
    trimmer = ResiliencyBasedGREPSK(classifier = model, pruning_set_fraction = pruning_fraction, min_eta = min_eta)
    trimmer.split_pruning_validation_set(X,y, validation_fraction)
    trimmer.trim(cost_criterion)
    trimmer.store_prunign_conf(pruning_path)
    trimmer.dump_report(report_path)


if __name__ == "__main__":
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum log level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("GrepSK.log"),  # Log to a file
            logging.StreamHandler()  # Log to console
        ]
    )
    parser = argparse.ArgumentParser(description="Accept a list of strings.")
    # Declare the arguments 
    parser.add_argument("--algo", type = int, help = "0 for loss based 1 for redundancy based", default = 0)
    parser.add_argument("--model_input", type = str, help = "Path of the joblib model.", default = None)
    parser.add_argument("--test_set", type = str, help = "Path of the testing set", default = None)
    parser.add_argument("--cost_criterion", type = int, help = "Type of used cost critetion for selecting the best leaf. 1 for Depth, 2 activity 3 combined.", default = 0)
    parser.add_argument("-f", "--fraction", type=float, help="Fraction of the pruning_set to use", default=0.5)
    parser.add_argument("-v", "--validation_fraction", type=float, help="Fraction of the pruning set used for validation. In redundancy based algo it is not used.", default=0.5)
    parser.add_argument("-l", "--max_loss", type=float, help="Maximum loss in loss based algorithm. In redundancy based algo it is not used.", default=1.0)
    parser.add_argument("-e", "--min_eta", type=int, help="Minimum difference between leaf redundancy and new redundancy. In loss based algo it is not used.", default=1.0)
    parser.add_argument("-c", "--pruning_path", type=str, help="Path in which the pruning cfg is saved", default=None)
    parser.add_argument("-r", "--report_path", type=str, help="Path in which the report is saved", default=None)

    args = parser.parse_args()
    if args.algo != 0 and args.algo != 1:
        assert 1 == 0, "Provide --algo 1 for Redundancy based and 0 for loss based"
    
    cost_criterion = args.cost_criterion
    if cost_criterion != 1 and cost_criterion != 2 and cost_criterion != 3:
        assert 1 == 0, "Provide a valid cost criterion, check help for more infos. "

    if not os.path.exists(args.model_input):
        assert 1 == 0, "Provide a valid model"
    model = joblib.load(args.model_input)

    if not os.path.exists(args.test_set):
        assert 1 == 0, "Provide a valid testing set path"
    df = pd.read_csv(args.test_set, sep = ";")
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    pruning_fraction = args.fraction
    if pruning_fraction <= 0.0 or pruning_fraction >= 1.0:
        assert 1 == 0, "Error in pruning fraction, provide a float in range [0.0, 1.0]"
     
    validation_fraction = args.validation_fraction
    if validation_fraction <= 0.0 or validation_fraction >= 1.0:
        assert 1 == 0, "Error in pruning fraction, provide a float in range [0.0, 1.0]"
    
    max_loss = args.max_loss
    if max_loss < 0.0 or max_loss > 100.0:
        assert 1 == 0, "Error in pruning fraction, provide a float in range [0.0, 100.0]"
    
    
    min_eta = args.min_eta
    if min_eta < 0:
        assert 1 == 0, "Error in pruning fraction, provide a positive minimum eta."

    pruning_path = args.pruning_path
    if not os.path.exists(pruning_path):
        assert 1 == 0, "Provide a valid path to store the pruning cfg"
    
    report_path = args.report_path
    if not os.path.exists(report_path):
        assert 1 == 0, "Provide a valid path to store the report of pruning"

    if args.algo == 0:
        run_loss_based_grep(model, cost_criterion, X, y, pruning_fraction, validation_fraction, max_loss, pruning_path, report_path)
    else:
        run_redundancy_based_grep(model, cost_criterion, X, y, pruning_fraction, validation_fraction, min_eta, pruning_path, report_path)
