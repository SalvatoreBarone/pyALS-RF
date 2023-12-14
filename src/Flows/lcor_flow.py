"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""

import logging, joblib, numpy as np
from distutils.dir_util import mkpath
from itertools import combinations, product
from tqdm import tqdm
from ..ctx_factory import load_configuration_ps, create_classifier
from ..ConfigParsers.PsConfigParser import *
from ..Model.Classifier import Classifier


# def compute_leaves_correlation(classifier : Classifier):
#     for cls in classifier.model_classes:
#         print(cls)
#         for comb in combinations(classifier.trees, 2):
#             db_a = comb[0].decision_boxes 
#             db_b = comb[1].decision_boxes 
            
#             leaves_a = [ l for l in comb[0].leaves if l["class"] == cls]
#             leaves_b = [ l for l in comb[1].leaves if l["class"] == cls]
            
#             for la in leaves_a:
#                 nodes_a = la["sop"].replace("not ", "").replace("(", "").replace(")", "").split(" and ")
#                 not_nodes_a = [ "<=" if "not" in n else ">" for n in la["sop"].replace("(", "").replace(")", "").split(" and ")  ]
#                 dbs_a = [ db for db in db_a if db["name"] in nodes_a ]
#                 for db, out in zip(dbs_a, not_nodes_a):
#                     print(db["box"].feature_name, out, db["box"].threshold)
                
#                 print("")
#                 for lb in leaves_b:
#                     nodes_b = lb["sop"].replace("not ", "").replace("(", "").replace(")", "").split(" and ")
#                     not_nodes_b = [ "<=" if "not" in n else ">" for n in lb["sop"].replace("(", "").replace(")", "").split(" and ")  ]
#                     dbs_b = [ db for db in db_b if db["name"] in nodes_b ]
#                     for db, out in zip(dbs_b, not_nodes_b):
#                         print(db["box"].feature_name, out, db["box"].threshold)
#       exit()

# Compute the number of samples in each leaf
# The func returns a data structure indexed by [leaf_class,tree,leaf].
# Each element [i,j,z] contains (test_input,value) that falls into the leaf z.
def samples_per_leaves(classifier : Classifier):
    # samples_per_leaves_dict = { c : {t.name : {  l["sop"] : [] for l in t.leaves if l["class"] == c } for t in classifier.trees } for c in classifier.model_classes}
    samples_per_leaves_dict = { c : {t.name : {  l["sop"] : set() for l in t.leaves if l["class"] == c } for t in classifier.trees } for c in classifier.model_classes} 
    number_of_leaves = 0
    for x, y in  tqdm(zip(classifier.x_test, classifier.y_test), total=len(classifier.x_test), desc="Computing dataset partitions", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        for tree in classifier.trees:
            boxes_output = tree.get_boxes_output(x) # Get decision boxes
            for class_name, assertions in tree.class_assertions.items():
                for leaf in assertions:
                    number_of_leaves += 1
                    if eval(leaf, boxes_output): # if the Assertion is true then save (x,y) in the DS
                        # print(f'Leaf name {leaf}')
                        # print(f'Sample, class {(tuple(frozenset(x)),y[0])}')
                        # samples_per_leaves_dict[class_name][tree.name][leaf].add((x, y))
                        samples_per_leaves_dict[class_name][tree.name][leaf].add((tuple(frozenset(x)),y[0]))
    print(f'N.ro leaves : {number_of_leaves}')
    return samples_per_leaves_dict

# Compute the correlation between each couple of leaves.
# returns a structure indexed by [(tree_A,leaf_A),(tree_B,leaf_B)] such that:
# DS[(tree_A,leaf_A),(tree_B,leaf_B)] =  correlation(leaf_a,leaf_b)
def compute_leaves_correlation(samples_per_leaves_dict):
    corr_per_leaf = {}
    conc = 0
    disc = 0
    for cls, cls_data in samples_per_leaves_dict.items():
        for tree,tree_data in cls_data.items():
            for leaf,leaf_data in tree_data.items():
                print(leaf_data)
                # if len(leaf_data) == 0:
                #     disc += 1
                # else :
                #     conc += 1

    for cls, cls_data in samples_per_leaves_dict.items(): # for each class
        for comb in combinations(cls_data.keys(), 2): # for each couple of tree.
            tree_A = comb[0] # take the pair of tree 
            tree_B = comb[1]
            #for leaves in samples_per_leaves_dict[cls][tree_A], samples_per_leaves_dict[cls][tree_B]:
            # counter = 0 
            for leaves in product(list(samples_per_leaves_dict[cls][tree_A].keys()),list(samples_per_leaves_dict[cls][tree_B].keys())):
                N11, N00, N1X, NX1, N0X, NX0 = 0, 0, 0, 0, 0, 0
                Q_stat = 0
                #print(leaves)
                leaf_A = leaves[0]
                leaf_B = leaves[1]

                # print(f'Leaf A: {leaf_A}')
                # print(f'Leaf B: {leaf_B}')
                # if leaf_A in samples_per_leaves_dict[cls][tree_A].keys():
                #     print("OOOOK ")
                # else:
                #     print(" FUCK ")
                # if leaf_B in samples_per_leaves_dict[cls][tree_A].keys():
                #     print("OOOOK ")
                # else:
                #     print(" FUCK ")
                # print(f'Samples in leaf A: {samples_per_leaves_dict[cls][tree_A][leaf_A]}')
                # print(f'Samples in leaf B: {samples_per_leaves_dict[cls][tree_B][leaf_B]}')
                # if(len(samples_per_leaves_dict[cls][tree_A][leaf_A]) > 0 and len(samples_per_leaves_dict[cls][tree_B][leaf_B]) > 0 ):
                #     conc += 1
                # elif  :
                #     disc += 1
                
                #exit()
                
                # print(f'{type(samples_per_leaves_dict[cls][tree_A][leaf_A])}')
                # if(len(samples_per_leaves_dict[cls][tree_A][leaf_A]) > 0 and len(samples_per_leaves_dict[cls][tree_B][leaf_B]) > 0 ) :
                #     print(type(samples_per_leaves_dict[cls][tree_A][leaf_A][0])) 
                #     print(type(samples_per_leaves_dict[cls][tree_A][leaf_A][0][0])) 
                #     print(samples_per_leaves_dict[cls][tree_A][leaf_A][0][0]) 
                #     print(type(samples_per_leaves_dict[cls][tree_A][leaf_A][0][1]))
                #     print(samples_per_leaves_dict[cls][tree_A][leaf_A][0][1]) 

                #     print(type(samples_per_leaves_dict[cls][tree_B][leaf_B][0])) 
                #     print(type(samples_per_leaves_dict[cls][tree_B][leaf_B][0][0])) 
                #     print(samples_per_leaves_dict[cls][tree_B][leaf_B][0][0]) 
                #     print(type(samples_per_leaves_dict[cls][tree_B][leaf_B][0][1]))
                #     print(samples_per_leaves_dict[cls][tree_B][leaf_B][0][1]) 

                #     counter += 1 
                #     if counter == 10:
                #         exit()
                
                # for xy in samples_per_leaves_dict[cls][tree_A][leaf_A] :
                #     is_present = False
                #     # check if the class is on the other vector
                #     for xy1 in sample   

                      
                # The set here is the set of (x,y) present in at least leaf_A or leaf_B without repetition.
                #for xy in samples_per_leaves_dict[cls][tree_A][leaf_A] + samples_per_leaves_dict[cls][tree_B][leaf_B]:   
                # for xy in np.union1d(samples_per_leaves_dict[cls][tree_A][leaf_A], samples_per_leaves_dict[cls][tree_B][leaf_B]):
                #    y = xy[1]
                #     if xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy in samples_per_leaves_dict[cls][tree_B][leaf_B] and y == cls:
                #         # Case 1: x,y is in both leaves and y == leaf_label i.e. prediction is correct 
                #         N11 += 1
                #     elif xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy in samples_per_leaves_dict[cls][tree_B][leaf_B] and y != cls:
                #         # Case 2: x,y is in both leaves and y != leaf_label i.e. prection is incorrect.
                #         N00 += 1
                #     elif xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy not in samples_per_leaves_dict[cls][tree_B][leaf_B] and y == cls:
                #         # Case 3: x,y is only in leaf_A and y == leaf_label.
                #         N1X += 1
                #     elif xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy not in samples_per_leaves_dict[cls][tree_B][leaf_B] and y != cls:
                #         # Case 4: x,y is only in leaf_A and y != leaf_label.
                #         N0X += 1
                #     elif xy not in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy  in samples_per_leaves_dict[cls][tree_B][leaf_B] and y == cls:
                #         # Case 5: x,y is only in leaf_B and y == leaf_label.
                #         NX1 += 1
                #     elif xy not in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy in samples_per_leaves_dict[cls][tree_B][leaf_B] and y != cls:
                #         # Case 6: x,y is only in leaf_B and y != leaf_label.
                #         NX0 += 1
                # n_conc = N11*N00
                # n_disc = N1X*NX1*N0X*NX0
                # #if len(samples_per_leaves_dict[cls][tree_A][leaf_A]) == 0 and len(samples_per_leaves_dict[cls][tree_B][leaf_B]) == 0 :
                # if n_conc == 0 or n_disc == 0:
                #    Q_stat = 0
                # else :
                #     Q_stat = (n_conc - n_disc) / (n_conc + n_disc)
                # #print(Q_stat)
                #exit()
                # corr_per_leaf[(tree_A,leaf_A)] = {(tree_B,leaf_B) : Q_stat}
    print(f'conc {conc} disc {disc}')
    return corr_per_leaf
                        
            
# Algorithm(classifier, sample_per_leaf,corr_per_leaf,T',max_loss):
# 
# score_per_leaf = compute_score(corr_per_leaf, classifier)
# base_accuracy =  (classifier,T') -> già la tengo
# actual_accuracy = base_accuracy 
# scores_idx = 0
# while base_accuracy - actual_accuracy > max_loss and scores_idx < len(score_per_leaf) :
#   best_score = score_per_leaf[0]
#   helper_classifier = prune(best_score, classifier)
#   actual_score = ComputeAccuracy(classifier,T')
#   if base_accuracy - actual_accuracy >= max_loss:
#       classifier = helper_classifier
#       Update_Correlation(corr_per_leaf)
#       Update_Scores(score_per_leaf)
#       scores_idx = 0 
#   else 
#       scores_idx ++
# Problemi: indice scores_idx -> aggiornamento delle correlazioni mi porta a dover ricalcolare gli score, indice va resettato?

def leaves_correlation_flow(ctx, output):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    
    create_classifier(ctx)
    samples_per_leaves_dict = samples_per_leaves(ctx.obj["classifier"])
    # for c, class_data in samples_per_leaves_dict.items():
    #     print(c)
    #     for tree, tree_data in class_data.items():
    #         print(f"\t{tree}")
    #         for leaf, samples in tree_data.items():
    #             print(f"\t\t{leaf}: {len(samples)}")
    corr_per_leaf = compute_leaves_correlation(samples_per_leaves_dict)
    # for leaf,leaf_data in corr_per_leaf.items():
    #     print(f'**** Leaf {leaf} is correlated with : \n')
    #     for leaf_data,corr_value in leaf_data.items():
    #        print(f'Leaf: {leaf}  Corr Value: {corr_value}  ')
    #     print(f"\n")