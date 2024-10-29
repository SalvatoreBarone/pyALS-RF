"""
Copyright 2021-2024 Salvatore Barone <salvatore.barone@unina.it>
                    Antonio Emmanuele <antonio.emmanuele@unina.it>
This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
pyALS-RF; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""

import logging, joblib, numpy as np
from distutils.dir_util import mkpath
from itertools import combinations, product
from tqdm import tqdm
from ..ctx_factory import load_configuration_ps, create_classifier, store_flow
from ..ConfigParsers.PsConfigParser import *
from ..Model.Classifier import Classifier
from ..Model.FaultCollection import FaultCollection
import os 
import json5
import copy
import numpy as np
import struct

""" Begin bit string manipulation functions. **** """
# Convert numpy.float64 to a bitstring
def float64_to_bitstring(value):
    # Use struct to interpret the float as a binary sequence
    packed = struct.pack('>d', value)  # '>d' for big-endian double-precision float
    # Convert the binary sequence into an integer and then to binary string
    bitstring = ''.join(f'{byte:08b}' for byte in packed)
    return bitstring

# Modify specific bits (for example, setting bit 5 and 10 to 1)
def modify_bitstring(bitstring, positions, values):
    bit_list = list(bitstring)
    for pos, val in zip(positions, values):
        bit_list[pos] = str(val)
    return ''.join(bit_list)

# Convert the modified bitstring back to numpy.float64
def bitstring_to_float64(bitstring):
    # Convert bitstring back to an integer
    int_value = int(bitstring, 2)
    # Pack this integer back into a binary sequence and interpret it as a float
    packed = int_value.to_bytes(8, byteorder='big')
    return struct.unpack('>d', packed)[0]

""" Inject a fault into a feature.
    feature:        The feature that will be injected with faults.
    bit_positions:  Positions of the bits that will be altered.
    fixed_values:   The values (0/1 for each bit_position) that the final result will have.
    Example:
        feature : in binary 101111111
        bit_positions: [6,7,8]
        values : [0,0,0]
        Output bitstring ( which will be reconverted) : 101111000
"""
def inject_fault_feature(feature, bit_positions, fixed_values):
    bitstring = float64_to_bitstring(value = feature)
    injected_bitstring = modify_bitstring(bitstring = bitstring, positions = bit_positions, values = fixed_values)
    reconverted_value = bitstring_to_float64(bitstring = injected_bitstring)
    return reconverted_value

def inject_fault_input(classifier: Classifier, faults, x_test):
    # For each fault {feat_idx : {bit_flipped: value}}
    for f in faults:
        feature_idx = list(f.keys())[0]
        bit_to_flip = list(f[feature_idx].keys())[0]
        fixed_value = f[feature_idx][bit_to_flip]
        # Reconvert in integers
        feature_idx = int(feature_idx)
        bit_to_flip = int(bit_to_flip)
        fixed_value = int(fixed_value)
        # For each possible input alter the specific feature of the fault.
        for idx, ipt in enumerate(x_test):
#            old_feat = x_test[idx][feature_idx]
            x_test[idx][feature_idx] = inject_fault_feature(feature = ipt[feature_idx], bit_positions = [bit_to_flip], fixed_values = [fixed_value])
            # if old_feat != x_test[idx][feature_idx]:
            #     print("ALTERED ")
            #     print(old_feat)
            #     print(x_test[idx][feature_idx])
            #     print(float64_to_bitstring(old_feat))
            #     print(float64_to_bitstring(x_test[idx][feature_idx]))
            #     print(f)
            #     exit(1)
            # else:
            #     print("NOT ALTERED ")
            #     print(f)
            #     print(old_feat)
            #     print(x_test[idx][feature_idx])
            #     print(float64_to_bitstring(old_feat))
            #     print(float64_to_bitstring(x_test[idx][feature_idx]))
            #print(x_test[idx][feature_idx])

""" End bit string manipulation functions. **** """
# Module interface function used to generate a fault collection
def gen_fault_collection(ctx, working_mode = 0, error_margin = 0.01, confidence_level = 0.95, individual_prob = 0.5, out_dir = "./", ncpus = 1):
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    # Generate the fault collection
    f = FaultCollection(classifier)
    # Distinct the two working modes.
    if working_mode == 0: # Sample from the entire fault universe
        f.sample_faults(type_of_faults = 0, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
    elif working_mode == 1: # Sample from different fault universes
        f.sample_faults(type_of_faults = 1, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        f.sample_faults(type_of_faults = 2, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        f.sample_faults(type_of_faults = 3, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
    f.faults_to_json5_list(classifier = classifier,  out_path = out_dir)

# Execute a faulted visit for each different fault in the folder input_faults.
# Classess probability vectors are saved into the output dir specified by the 
# "output" parameter in input. Each file (1 for each category of fault so 3 different)
# files, containts a list, for each fault for type in input fault, of the probabilities vector.
# i.e. for 10 feature faults and 50 input samples the feat out file contains 50 vector for each different
# fault (i.e. 10 different features.)
def fault_visit(ctx, output, input_faults, ncpus, num_samples = 50):
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    x_test = copy.deepcopy(classifier.x_test[0 : num_samples])
    y_test = copy.deepcopy(classifier.y_test[0 : num_samples])
    """ ************************************************ """
    # Feature Faults
    feat_path   = os.path.join(input_faults, "feat_faults.json5") 
    out_path_vectors    = os.path.join(output, "feat_faults_vectors.json5")
    # Load the feat JSON5 file
    with open(feat_path, "r") as file:
        loaded_faults = json5.load(file)
    x_test_temp = x_test
    fault_vect_list  = []
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with feature faults"):
        # Inject faults into the inputs
        inject_fault_input(classifier = classifier, faults = loaded_faults, x_test = x_test_temp)
        # Visit
        faulted_vec = classifier.predict(x_test_temp)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # # Restore, x_test_temps mantaints at each cycle always the pointer to the original array.
        x_test_temp = copy.deepcopy(x_test)
    # Save to json5 the list of vectors
    with open(out_path_vectors, "w") as f:
        json5.dump(fault_vect_list, f, indent = 2)
    """ ************************************************ """
    # For DBs faults.
    dbs_path   = os.path.join(input_faults, "dbs_faults.json5") 
    out_path_vectors    = os.path.join(output, "dbs_faults_vectors.json5")
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(dbs_path, "r") as file:
        loaded_faults = json5.load(file)
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with feature faults"):
        # Inject faults into the inputs
        old_dbs = classifier.inject_tree_boxes_faults_fb(f)
        # Visit
        faulted_vec = classifier.predict(x_test_temp)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # Restore
        classifier.restore_dbs(f)
    # Save to json5 the list of vectors
    with open(out_path_vectors, "w") as f:
        json5.dump(fault_vect_list, f, indent = 2)
    
    """ ************************************************ """
    # For BNs faults
    bns_path   = os.path.join(input_faults, "bns_faults.json5") 
    out_path_vectors    = os.path.join(output, "bns_faults_vectors.json5")
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(bns_path, "r") as file:
        loaded_faults = json5.load(file)
    old_bns = classifier.store_bns()
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with feature faults"):
        # Inject faults into the inputs
        classifier.inject_bns_faults(f)
        # Visit
        faulted_vec = classifier.predict(x_test_temp)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # Restore
        classifier.restore_bns(old_bns)
    # Save to json5 the list of vectors
    with open(out_path_vectors, "w") as f:
        json5.dump(fault_vect_list, f, indent = 2)

# Function used to dump class probabilities vectors without any fault. 
def dump_unfaulted_class_vector(ctx, output, ncpus, num_samples = 50):
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    x_test = copy.deepcopy(classifier.x_test[0 : num_samples])
    vectors = classifier.predict(x_test)
    vectors = vectors.tolist()
    # Save to json5 the list of vectors
    with open(os.path.join(output, "class_vec_no_faults.json5"), "w") as file:
        json5.dump(vectors, file, indent = 2)
""" Test function used during development. """
# def fault_injection(ctx, output, ncpus):
#     # Initialize the logger 
#     logger = logging.getLogger("pyALS-RF")
#     logger.info("Runing the TMR flow.")
#     load_configuration_ps(ctx)
#     create_classifier(ctx)    
#     classifier = ctx.obj["classifier"]
#     #boxes_fault_universe = get_decision_boxes_fault_sites(classifier = classifier)
#     #bn_fault_sites = get_bn_fault_sites(classifier = classifier)
#     #input_fault_sites = get_input_fault_sites(classifier = classifier)
    
#     # Test Injection Assertions
#     #print(classifier.trees[0].get_boolean_net("0", False))
#     #print(classifier.trees[0].get_boolean_net("1", False))
#     #print(classifier.trees[1].get_boolean_net("0", False))
#     #classifier.inject_bns_faults({ "0" : {"0": {'(not Node_0 and not Node_1 and Node_2 and Node_22 and Node_24)': "False"}, "1" : {'(Node_0 and not Node_28 and Node_29)': "True"}}, "1": {"0": {'(not Node_0 and not Node_1 and not Node_2 and Node_3)': "False"} }})
#     #print(classifier.trees[0].boolean_networks[0])
#     #print(classifier.trees[0].boolean_networks[1])
#     #print(classifier.trees[1].boolean_networks[0])
#     #print("Ciao Ciao")

#     # # Test altered visiting phase with restoring
    
#     # scores = classifier.predict(classifier.x_test[0:10])
#     # old_bns = classifier.store_bns()
#     # # Store the previously sampled classifier output.
    
#     # classifier.inject_bns_faults({ "0" : {"0": {'(not Node_0 and not Node_1 and Node_2 and Node_22 and Node_24)': "False"}, "1" : {'(Node_0 and not Node_28 and Node_29)': "True"}}, "1": {"0": {'(not Node_0 and not Node_1 and not Node_2 and Node_3)': "False"} }})
#     # old_dbs = classifier.inject_tree_boxes_faults_fb({"0": {"Node_1" : True, "Node_2" : False}, "1": {"Node_3" : False, "Node_4": True}})
#     # scores_altered = classifier.predict(classifier.x_test[0:10])
#     # ctr_original = 0
#     # ctr_n_altered = 0
#     # for score, altered_score in zip(scores,scores_altered):
#     #     ctr_original += 1
#     #     if not np.array_equal(score, altered_score):
#     #         print(f"Original {score} Altered {altered_score}")
#     #         ctr_n_altered += 1
#     #         ctr_original  -= 1
#     # print(ctr_original)
#     # print(ctr_n_altered)
#     # classifier.restore_bns(old_bns)
#     # classifier.restore_dbs(old_dbs)
#     # restored_scores = classifier.predict(classifier.x_test[0:10])
#     # for score, restored_score in zip(scores, restored_scores):
#     #     if not np.array_equal(score,restored_score):
#     #         assert 1 == 0,"RESTORING NOT WORING"
#     # print("ALL OK !")


#     # Test modified assertion visiting.
#     #    v = " False or Y"
#     #    print(int(eval(v, {"X": False, "N" : False, "Y" : False})))
#     #    print("Hello")
    
#     # Test boolean networks injection
#     #print(classifier.trees[0].correct_boxes)
#     #print(classifier.trees[0].faulted_boxes)
    
#     #print(classifier.trees[0].fix_assertion_fns())
#     #print(classifier.trees[1].correct_boxes)
#     #print(classifier.trees[1].faulted_boxes)
    
#     # faults = sample_faults(list(range(100)), 5)
#     # print(faults)

#     # Test fault collection, change type_of_faults for different fault types.
#     # f = FaultCollection(classifier)
#     # f.print_fault_sites()
#     # f.sample_faults(type_of_faults = 3, nro_faults = 100)
#     # f.print_faults()

#     # Test the fault collection json generation

#     # f = FaultCollection(classifier)
#     # f.sample_faults(type_of_faults = 1, nro_faults = 10)
#     # f.sample_faults(type_of_faults = 2, nro_faults = 10)
#     # f.sample_faults(type_of_faults = 3, nro_faults = 10)

#     # f.print_faults()
#     # tree_names = [tree.name for tree in classifier.trees]
#     # classes_names = [class_name for class_name in classifier.classes_name]
#     # list_features = [feat["name"] for feat in classifier.model_features]
#     # f.faults_to_json5(list_feature_names = list_features,list_tree_names = tree_names, list_class_names = classes_names,  out_path = "./")
#     # exit(1)

#     # print("Fault universe size")
#     f = FaultCollection(classifier)
#     f.sample_faults(type_of_faults = 0, error_margin = 0.01, confidence_level = 0.95, individual_prob = 0.5)
#     f.faults_to_json5_list(classifier = classifier,  out_path = "./")
