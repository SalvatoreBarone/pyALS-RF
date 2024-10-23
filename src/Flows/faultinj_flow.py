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
        
def tree_visiting_injected(classifier: Classifier, input_sample, input_faults, boxes_faults, bn_faults):
    # Inject into the input sample.
    # Get the Boxes out
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
    print(input_fault_sites)
    exit(1)
    # Obtaining the fault sites for each input.
    #print(bn_fault_sites)