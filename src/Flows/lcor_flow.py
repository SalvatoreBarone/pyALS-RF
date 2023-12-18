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
    #total_set = set()
    #s = 0
    #classified = [False for i in range(0,len(classifier.x_test))]
    for x, y in  tqdm(zip(classifier.x_test, classifier.y_test), total=len(classifier.x_test), desc="Computing dataset partitions", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        for tree in classifier.trees:
            boxes_output = tree.get_boxes_output(x) # Get decision boxes
            for class_name, assertions in tree.class_assertions.items():
                for leaf in assertions:
                    if eval(leaf, boxes_output): # if the Assertion is true then save (x,y) in the DS
                        # print(f'Leaf name {leaf}')
                        # print(f'Sample, class {(tuple(frozenset(x)),y[0])}')
                        # samples_per_leaves_dict[class_name][tree.name][leaf].add((x, y))
                        #classified[s] = True
                        samples_per_leaves_dict[class_name][tree.name][leaf].add((tuple(frozenset(x)),y[0]))
                        #total_set.add(tuple(frozenset(x)))
        #s += 1
    # # To delete...
    # number_of_leaves = 0
    # for tree in classifier.trees:
    #     boxes_output = tree.get_boxes_output(x) # Get decision boxes
    #     for class_name, assertions in tree.class_assertions.items():
    #         for leaf in assertions:
    #             number_of_leaves += 1
    # print(f'N.ro leaves : {number_of_leaves} total number of classified samples {len(total_set)}')
    # print(f'number of samples {len(classifier.x_test)}') 
    # print(f'Counted samples {s}')
    # counted_classified = 0
    # for c in classified:
    #     if c == True :
    #         counted_classified += 1
    # print(f'Counted classified {counted_classified}')
    # reps = 0
    # for i in range(0,len(classifier.x_test)):
    #     for j in range(0,len(classifier.x_test)):
    #         if np.array_equal(classifier.x_test[i],classifier.x_test[j]) and i != j:
    #             reps += 1
    # print(f' Repetitions {reps} Total: {reps+len(total_set)}')
    
    # counter = 0
    # counter_wrong = 0
    # for cls, cls_data in samples_per_leaves_dict.items():
    #     for tree,tree_data in cls_data.items():
    #         for leaf,leaf_data in tree_data.items():
    #             for sample in leaf_data:
    #                 # print(sample)
    #                 # print(sample[1])
    #                 if sample[1] != int(cls):
    #                     counter_wrong += 1
    #                     print(f'Leaf {cls} Correct {sample[1]}')
    #                     print(f' Type CLS : {type(cls)} Type sample : {type(sample[1])}')
    #                 counter += 1
    # print(f'Total {counter} Wrong {counter_wrong}')
    return samples_per_leaves_dict

# Compute the correlation between each couple of leaves.
# returns a structure indexed by [(tree_A,leaf_A),(tree_B,leaf_B)] such that:
# DS[(tree_A,leaf_A),(tree_B,leaf_B)] =  correlation(leaf_a,leaf_b)
def compute_leaves_correlation(samples_per_leaves_dict):
    corr_per_leaf = {}
    corr_per_leaf_cleaned = {}
    for cls, cls_data in samples_per_leaves_dict.items(): # for each class
        for comb in combinations(cls_data.keys(), 2): # for each couple of tree.
            tree_A = comb[0] # take the pair of tree 
            tree_B = comb[1]
            for leaves in product(list(samples_per_leaves_dict[cls][tree_A].keys()),list(samples_per_leaves_dict[cls][tree_B].keys())):
                #N11, N00, N1X, NX1, N0X, NX0 = 0, 0, 0, 0, 0, 0
                conc, disc = 0, 0
                Q_stat = 0
                leaf_A = leaves[0]
                leaf_B = leaves[1]

                # if len(samples_per_leaves_dict[cls][tree_A][leaf_A]) > 0 and len(samples_per_leaves_dict[cls][tree_B][leaf_B]) > 0 :
                #     # The set here is the set of (x,y) present in at least leaf_A or leaf_B without repetition.
                #     for xy in samples_per_leaves_dict[cls][tree_A][leaf_A] | samples_per_leaves_dict[cls][tree_B][leaf_B]:   
                #         y = xy[1]
                #         if xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy in samples_per_leaves_dict[cls][tree_B][leaf_B] and y == cls:
                #             # Case 1: x,y is in both leaves and y == leaf_label i.e. prediction is correct 
                #             N11 += 1
                #         elif xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy in samples_per_leaves_dict[cls][tree_B][leaf_B] and y != cls:
                #             # Case 2: x,y is in both leaves and y != leaf_label i.e. prection is incorrect.
                #             N00 += 1
                #         elif xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy not in samples_per_leaves_dict[cls][tree_B][leaf_B] and y == cls:
                #             # Case 3: x,y is only in leaf_A and y == leaf_label.
                #             N1X += 1
                #         elif xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy not in samples_per_leaves_dict[cls][tree_B][leaf_B] and y != cls:
                #             # Case 4: x,y is only in leaf_A and y != leaf_label.
                #             N0X += 1
                #         elif xy not in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy  in samples_per_leaves_dict[cls][tree_B][leaf_B] and y == cls:
                #             # Case 5: x,y is only in leaf_B and y == leaf_label.
                #             NX1 += 1
                #         elif xy not in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy in samples_per_leaves_dict[cls][tree_B][leaf_B] and y != cls:
                #             # Case 6: x,y is only in leaf_B and y != leaf_label.
                #             NX0 += 1
                #     if N11 > 0 or N1X > 0 or NX1 >0 :
                #         print("************************")
                #         print(f'N11: {N11} N00: {N00}')
                #         print(f'N0X: {N0X} NX0: {NX0}')
                #         print(f'N1X: {N1X} NX1: {NX1}')
                #         print("************************")

                if len(samples_per_leaves_dict[cls][tree_A][leaf_A]) > 0 and len(samples_per_leaves_dict[cls][tree_B][leaf_B]) > 0 :
                    # The set here is the set of (x,y) present in at least leaf_A or leaf_B without repetition.
                    for xy in samples_per_leaves_dict[cls][tree_A][leaf_A] | samples_per_leaves_dict[cls][tree_B][leaf_B]:   
                        y = xy[1]
                        if xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy in samples_per_leaves_dict[cls][tree_B][leaf_B]:
                            # Case 1: x,y is in both leaves 
                            conc += 1
                        elif xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy not in samples_per_leaves_dict[cls][tree_B][leaf_B]:
                            #  Case 2: x,y is in leafA but not in leafB
                            disc += 1
                        elif xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy not in samples_per_leaves_dict[cls][tree_B][leaf_B]:
                            #  Case 3: x,y is in leafB but not in leafA
                            disc += 1   
                    # print(f'TreeA {tree_A} leafA {leaf_A} treeB {tree_B} leafB {leaf_B}')
                    # print(f'Conc {conc} disc {disc}') 
                    Q_stat = (conc  -  disc) / (conc + disc)
                    # print(f'Qstat {Q_stat}')
                    if (tree_A,leaf_A) not in corr_per_leaf:
                        corr_per_leaf[(tree_A,leaf_A)] = {(tree_B,leaf_B) : Q_stat}
                    else:
                        corr_per_leaf[(tree_A,leaf_A)].update({(tree_B,leaf_B) : Q_stat})
                    # Compute the dual in order to remove the leaf with a greater number of literals.
                    if (tree_B,leaf_B) not in corr_per_leaf:
                        corr_per_leaf[(tree_B,leaf_B)] = {(tree_A,leaf_A) : Q_stat}
                    else:
                        corr_per_leaf[(tree_B,leaf_B)].update({(tree_A,leaf_A) : Q_stat})
                    if Q_stat > 0:
                        if (tree_A,leaf_A) not in corr_per_leaf_cleaned:
                            corr_per_leaf_cleaned[(tree_A,leaf_A)] = {(tree_B,leaf_B) : Q_stat}
                        else:
                            corr_per_leaf_cleaned[(tree_A,leaf_A)].update({(tree_B,leaf_B) : Q_stat})
                        # Compute the dual in order to remove the leaf with a greater number of literals.
                        if (tree_B,leaf_B) not in corr_per_leaf_cleaned:
                            corr_per_leaf_cleaned[(tree_B,leaf_B)] = {(tree_A,leaf_A) : Q_stat}
                        else:
                            corr_per_leaf_cleaned[(tree_B,leaf_B)].update({(tree_A,leaf_A) : Q_stat})
        
    #print(f'conc {conc} disc {disc} total number of samples {len(samples_set)}')
    return corr_per_leaf,corr_per_leaf_cleaned
                    
# Compute the score for each leaf multiplying the sum(Q_stat(A,B))* Num literals.
def compute_leaves_score(corr_per_leaf):
    scores = {}
    for leafA in corr_per_leaf:
        scores[leafA] = 0
    for leafA,correlated in corr_per_leaf.items():
        num_and = 0
        for leaves,Q_val in correlated.items():
            scores[leafA] += Q_val
        splitted = leafA[1].split()
        for and_w in splitted:
            if and_w == "and":
                num_and += 1
        #print(f'Name {leafA[1]} num {num_and}')
        scores[leafA] = scores[leafA] * num_and
    # multiply by the number of literals.
    return scores

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
    
    corr_per_leaf,corr_per_leaf_cleaned = compute_leaves_correlation(samples_per_leaves_dict)
    scores = compute_leaves_score(corr_per_leaf)
    # print("********* Value not cleaned")
    # for k,v in scores.items():
    #     if v > 0:
    #         print(f'Leaf:({k[0]},{k[1]}) score: {v}')
    print("********* VAlue cleaned")
    scores2 = compute_leaves_score(corr_per_leaf_cleaned)
    for k,v in scores2.items():
        print(f'Leaf:({k[0]},{k[1]}) score: {v}')
    # for leaf,leaf_data in corr_per_leaf.items():
    #     print(f'**** Leaf {leaf} is correlated with : \n')
    #     for leaf_data,corr_value in leaf_data.items():
    #        print(f'Leaf: {leaf}  Corr Value: {corr_value}  ')
    #     print(f"\n")