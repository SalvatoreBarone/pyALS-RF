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
import pandas as pd

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

def test_classifier_from_indexes(ctx, quantization_type, indexes_path, ncpus, outpath):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)   
    classifier = ctx.obj["classifier"]
    # Alter decision boxes comparison logic
    if quantization_type != None:
        classifier.set_thds_type(quantization_type)

    if indexes_path is not None:
        validation_indexes = np.loadtxt(indexes_path, dtype = int)
        test_samples    = classifier.x_test[validation_indexes]
        test_labels     = classifier.y_test[validation_indexes]
    else:
        test_samples    = classifier.x_test
        test_labels     = classifier.y_test
    #sample_classes = classifier.predict(test_samples, disable_tqdm = False)
    leaves_per_tree = classifier.compute_leaves_idx(test_samples, disable_tqdm = False)
    class_per_tree  = classifier.transform_leaves_into_classess(leaves_per_tree)
    
    # Transpose class_per_tree
    classes_per_sample = [[] for _ in range(len(test_samples))]
    for i in range(len(test_samples)):        
        for j in range(len(classifier.trees)):      
            classes_per_sample[i].append(class_per_tree[j][i])
    
    votes_vector = classifier.get_votes_vectors_by_leaves_idx(leaves_per_tree, classifier.y_test)
    leaves_costs = classifier.get_leaves_costs_by_leaves_idx(leaves_per_tree)

    num_classes  = np.unique(test_labels.ravel()).size
    corr_class = [ 0 for label in range(num_classes)]
    total_count_per_class = [0 for label in range(num_classes)]
    for sId, y in enumerate(test_labels):
        correct_label = int(y)
        total_count_per_class[correct_label] += 1
        if correct_label == int(np.argmax(votes_vector[sId])):
            corr_class[correct_label] += 1
    for i in range(num_classes):
        corr_class[i] = corr_class[i] / total_count_per_class[i] * 100
    if os.path.exists(outpath):
        out_vv = os.path.join(outpath, "votes_vector.txt")
        out_cacc = os.path.join(outpath, "per_class_acc.txt")
        lCostPath = os.path.join(outpath, "leaves_costs.txt")
        leavesLabelsPath = os.path.join(outpath, "leaves_labels.txt")
        np.savetxt(out_vv, votes_vector, fmt = "%d")
        np.savetxt(out_cacc, corr_class, fmt ="%.4f")
        np.savetxt(lCostPath, leaves_costs, fmt = "%d")
        np.savetxt(leavesLabelsPath, classes_per_sample, fmt ="%d")


def perclass_margin(ctx, quantization_type, indexes_path, ncpus, outpath):
    def per_sample_margin(pred_vectors, yprun):
        """Compute one margin value per sample in the test set."""
        ntrees = np.sum(pred_vectors[0])
        margins = []
        for p, y in zip(pred_vectors, yprun):
            y = int(y)
            votes_correct = p[y]
            sorted_votes = np.sort(p)
            # second most voted class
            if np.argmax(p) == y:
                second_class_votes = sorted_votes[-2]
            else:
                second_class_votes = sorted_votes[-1]
            margins.append((votes_correct - second_class_votes) / ntrees)
        return margins

    logger = logging.getLogger("pyALS-RF")
    logger.info("[MR-HEU-FLOW] Running per-class margin evaluation...")

    # === Load classifier and configuration ===
    load_configuration_ps(ctx)
    create_classifier(ctx)
    classifier = ctx.obj["classifier"]

    if quantization_type is not None:
        classifier.set_thds_type(quantization_type)

    # === Load test subset if provided ===
    if indexes_path is not None:
        validation_indexes = np.loadtxt(indexes_path, dtype=int)
        test_samples = classifier.x_test[validation_indexes]
        test_labels = classifier.y_test[validation_indexes]
    else:
        test_samples = classifier.x_test
        test_labels = classifier.y_test

    # === Compute leaves and per-tree class predictions ===
    leaves_per_tree = classifier.compute_leaves_idx(test_samples, disable_tqdm=False)
    class_per_tree = classifier.transform_leaves_into_classess(leaves_per_tree)

    # === Compute vote vectors and margins (test set only) ===
    votes_vector = classifier.get_votes_vectors_by_leaves_idx(leaves_per_tree, classifier.y_test)
    margins = per_sample_margin(pred_vectors=votes_vector, yprun=test_labels)

    # === Build per-sample DataFrame ===
    os.makedirs(outpath, exist_ok=True)
    # print(margins)
    # print(len(test_labels))
    # print(len(margins))
    # print(test_labels)
    # exit(1)
    per_sample_df = pd.DataFrame({
        "SampleIndex": np.arange(len(test_labels)),
        "Label": test_labels.ravel(),
        "Margin": margins
    })

    per_sample_csv = os.path.join(outpath, "per_sample_margins.csv")
    per_sample_df.to_csv(per_sample_csv, index=False)
    logger.info(f"[MR-HEU-FLOW] Saved per-sample margins to: {per_sample_csv}")

    # === Compute per-class margin statistics (test set) ===
    class_margin_stats = (
        per_sample_df
        .groupby("Label")
        .agg(
            Test_Samples=("Margin", "count"),
            Average_Margin=("Margin", "mean"),
            Std_Margin=("Margin", "std")
        )
        .reset_index()
    )

    # === Compute total (train + test) samples per class ===
    train_labels = classifier.y_train.ravel()
    test_labels_all = classifier.y_test.ravel()
    all_labels = np.concatenate((train_labels, test_labels_all))
    unique_labels, total_counts = np.unique(all_labels, return_counts=True)
    class_total_counts = pd.DataFrame({
        "Label": unique_labels,
        "Total_Samples_TrainTest": total_counts
    })

    # === Merge per-class stats with total counts ===
    class_summary = pd.merge(class_margin_stats, class_total_counts, on="Label", how="outer").fillna(0)

    # === Save final per-class summary ===
    per_class_csv = os.path.join(outpath, "per_class_margin_summary.csv")
    class_summary.to_csv(per_class_csv, index=False)
    logger.info(f"[MR-HEU-FLOW] Saved per-class margin summary to: {per_class_csv}")

    logger.info("[MR-HEU-FLOW] Per-class margin evaluation completed successfully.")