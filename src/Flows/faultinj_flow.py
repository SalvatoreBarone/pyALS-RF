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
import os 



import numpy as np
import struct

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
    print(bitstring)
    injected_bitstring = modify_bitstring(bitstring = bitstring, positions = bit_positions, values = fixed_values)
    print(injected_bitstring)
    reconverted_value = bitstring_to_float64(bitstring = injected_bitstring)
    return reconverted_value

# Get all the decision boxes.
# The fault universe of decision boxes is the set of decision boxes per each different tree.
def get_decision_boxes_fault_sites(classifier: Classifier):
    fault_per_tree = {}
    for tree in classifier.trees:
        fault_per_tree.update({tree.name: [box["name"] for box in tree.decision_boxes]})
    return fault_per_tree
        
# Get all the fault universe of the boolean network.
# Implying the set of assertion functions for each possible 
# boolean network.
def get_bn_fault_sites(classifier: Classifier):
    fault_per_tree = {}
    # Boolean network is a list of dictionaries where each item is a dictionary 
    # containing all the informations for a class.

    # For each tree
    for tree in classifier.trees:
        fault_per_class = {}
        # Save the minterms for each class.
        for bn in tree.boolean_networks:
            fault_per_class.update({bn["class"] : bn["minterms"]})
        fault_per_tree.update({tree.name : fault_per_class})
    return fault_per_tree

def get_input_fault_sites(classifier: Classifier):
    input_fault_sites = {}
    # print(classifier.x_test[0])
    for f in classifier.trees[0].model_features:
        input_fault_sites.update({f["name"] : 64})
    return input_fault_sites

def test_boxes_output(classifier: Classifier):
    for x in classifier.x_test:
        for tree in classifier.trees:
            for a in tree.boolean_networks:
                print(a["type"])
            exit(1)
            # Outputs of decision boxes are boolean values indexed with 'Node_Idx', idx ranges 
            # from 0 to Number of Nodes. 
            print(tree.get_boxes_output(x))
            for feature in x :
                print(type(x))
            exit(1)

""" Perform a visit of the classifier by injecting faults during the visiting phase.  
    classifier:     Classifier that will perform the tree visiting phase.
    input_fault:    For each input (i.e. for each element of the test set) the list containing for each input feature, the set of bit positions 
                    altered and the set of values (0 or 1) injected.
    boxes_faults:   For each tree in the classifier contains the set of nodes containing a list of fixed values per each Node (DB).
    bn_faults:      For each tree in the classifier contains the set of assertion functions with a fixed True or False value. 

"""        
def classifier_visit_injected(classifier: Classifier, input_faults, boxes_faults, bn_faults):
    # Inject into the input sample.
    x_test_inj = np.copy(classifier.x_test)
    
    # DECOMMENTALO DOPO.
    # # For each sample
    # for i, input_sample in enumerate(x_test_inj):
    #     # For each feature, inject faults.
    #     for j, input_feature in enumerate(input_sample):
    #         x_test_inj[i][j] = inject_fault_feature(input_feature, input_faults[i][j][0], input_faults[i][j][1])
            
    # Get the Boxes out using the injected test set.
    #inject_fault_boxes(classifier, boxes)
    #for tree, tree_box_faults in zip(classifier.trees, boxes_faults):
    for tree in classifier.trees:
        print(type(tree.name))
        tree.fix_boxes_outs({"Node_1" : True, "Node_17" : False})
        outs = tree.get_boxes_out_faults(attributes = classifier.x_test[0])
        #tree.fix_boxes_outs({"Node_1" : True, "Node_17" : False}) 
        #print(outs)
        exit(1)
    # Inject into the boxes
    # Inject into assertions
    # Compute outupt
    # Return prediction 
    return 0

def fault_injection(ctx, output, ncpus):
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    boxes_fault_universe = get_decision_boxes_fault_sites(classifier = classifier)
    bn_fault_sites = get_bn_fault_sites(classifier = classifier)
    input_fault_sites = get_input_fault_sites(classifier = classifier)
    classifier.inject_tree_boxes_faults({"0": {"Node_1" : True, "Node_2" : False}, "1": {"Node_3" : False, "Node_4": True}})
    #classifier_visit_injected(classifier = classifier, input_faults = [[[[6,7,8],[0,0,0]]]], boxes_faults = 0, bn_faults = 0)
    #print(classifier.trees[0].boolean_networks)
    v = " True or Y or N"
    print(int(eval(v, {"X": False, "N" : False, "Y" : False})))
    print("HEllo")
    #print(classifier.trees[0].correct_boxes)
    #print(classifier.trees[0].faulted_boxes)

    #print(classifier.trees[1].correct_boxes)
    #print(classifier.trees[1].faulted_boxes)

    #print(input_fault_sites)
    exit(1)
    # Obtaining the fault sites for each input.
    #print(bn_fault_sites)