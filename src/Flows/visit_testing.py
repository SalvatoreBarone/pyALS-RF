import logging, joblib, numpy as np
from distutils.dir_util import mkpath
from itertools import combinations, product
from tqdm import tqdm
from ..ctx_factory import load_configuration_ps, create_classifier, store_flow
from ..ConfigParsers.PsConfigParser import *
from ..Model.Classifier import Classifier
from .TMR.tmr import TMR
import os 
from pyalslib import double_to_hex, apply_mask_to_double, apply_mask_to_int, double_to_bin
from ..Model.FaultCollection import FaultCollection

def visit_test(ctx, ps_dir, val_path, working_mode = 0, error_margin = 0.01, confidence_level = 0.95, individual_prob = 0.5, out_dir = "./", ncpus = 1):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)   

    classifier = ctx.obj["classifier"]
    if ps_dir is not None:
        ps_dir_cfg = ps_dir
    else:
        ps_dir_cfg = ctx.obj["configuration"].outdir    
    
    with open(ps_dir_cfg, "r") as file:
        raw_content = file.read()
    corrected_content = raw_content.replace("{x:", '{"x":').replace(", f:", ', "f":').replace(", g:", ', "g":')
    
    # Convertire la stringa corretta in un dizionario Python
    try:
        data = json.loads(corrected_content)
    except json.JSONDecodeError as e:
        assert 1 == 0, "Unable to correct the errors in json5" 
    
    if val_path != None:
        validation_indexes = np.loadtxt(val_path, dtype = int)
        validation_samples_x = [classifier.x_test[i] for i in validation_indexes]
        validation_samples_y = [classifier.y_test[i] for i in validation_indexes]
        logger.info(f"Validating accuracy on new samples")
        base_acc =  classifier.evaluate_accuracy(validation_samples_x, validation_samples_y, disable_tqdm = False)
        logger.info(f"Baseline accuracy {base_acc}")
        acc_lossess = []
        nabs_array = []
        savings = []
        # Order each configuration by accuracy loss.
        for solution in tqdm(data, desc = "Evaluating accuracy on the test set"):
            nabs = solution["x"]
            savings.append(int(solution["f"][1]))
            nabs_array.append(nabs)
            nabs_dict = {f["name"]: n for f, n in zip(classifier.model_features, nabs[:len(classifier.model_features)])}
            classifier.set_nabs(nabs_dict)
            accuracy = classifier.evaluate_accuracy(validation_samples_x, validation_samples_y, disable_tqdm = True)
            loss = base_acc - accuracy
            logger.info(f"Accuracy for new solution {accuracy} Loss: {loss}")
            acc_lossess.append(loss)
            classifier.reset_nabs_configuration()
        sorted_idx = np.argsort(acc_lossess)
        min_loss_idx = np.argmin(acc_lossess)
        #loss_ranges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ranges_values = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]
        range_full    = [False for r in range(len(ranges_values))]
        selected_lossess = 0
        break_outer = False
        loss_array_sel = []
        for loss in sorted_idx:
            for range_id in range(0,len(ranges_values)):
                if acc_lossess[loss] >= ranges_values[range_id][0] and  acc_lossess[loss] < ranges_values[range_id][1] and not range_full[range_id] :
                    range_full[range_id] = True
                    selected_lossess += 1
                    loss_array_sel.append(loss)
                if selected_lossess == len(ranges_values):
                    break_outer = True
                    break
            if break_outer :
                break    
        found_lossess = [acc_lossess[sel] for sel in loss_array_sel]
        found_nabs = [nabs_array[sel] for sel in loss_array_sel] 
        found_savings = [savings[sel] for sel in loss_array_sel]

    else:
        # Simply load the first term.
        found_lossess = []
        found_nabs = []
        found_savings = []
        for solution  in tqdm(data, desc = "Evaluating accuracy on the test set"):
            found_lossess.append(int(solution["f"][0]))
            found_savings.append(int(solution["f"][1]))
            found_nabs.append(solution["x"])
    # Generate an index containing the configuration of each approximated variant
    index_list = []
    cfg_paths = []
    index_path = os.path.join(out_dir, "index.json5")
    for l,n, sav in zip(found_lossess, found_nabs, found_savings):
        # The out dir of each configuration is in the same folder of the index file
        cpath = os.path.join(out_dir, f"cfg_{l:.2f}")
        if not os.path.exists(cpath):
            os.makedirs(cpath)
        cfg_paths.append(cpath)
        index_list.append({ "Loss": l, "Nab" : n, "Out:": cpath , "Savings" : sav})
    with open(index_path, "w") as f:
        json5.dump(index_list, f ,indent = 2)
    # Generate the configuration for each NAB
    for l,n,cp in tqdm(zip(found_lossess, found_nabs, cfg_paths), desc = "Sampling and Saving Fault Configurations"):
        fc = FaultCollection(classifier, n)
        if working_mode == 0: # Sample from the entire fault universe
            fc.sample_faults(type_of_faults = 0, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        elif working_mode == 1: # Sample from different fault universes
            fc.sample_faults(type_of_faults = 1, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
            fc.sample_faults(type_of_faults = 2, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
            fc.sample_faults(type_of_faults = 3, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        elif working_mode >= 2 and working_mode < 5:
            fc.sample_faults(type_of_faults = working_mode - 1, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        else:
            assert 1 == 0, "Invalid configuration of the fault parameter"
        fc.faults_to_json5_list(classifier = classifier,  out_path = cp)
        