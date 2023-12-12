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
                
def samples_per_leaves(classifier : Classifier):
    samples_per_leaves_dict = { c : {t.name : {  l["sop"] : [] for l in t.leaves if l["class"] == c } for t in classifier.trees } for c in classifier.model_classes}
    for x, y in  tqdm(zip(classifier.x_test, classifier.y_test), total=len(classifier.x_test), desc="Computing dataset partitions", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        for tree in classifier.trees:
            boxes_output = tree.get_boxes_output(x)
            for class_name, assertions in tree.class_assertions.items():
                for leaf in assertions:
                    if eval(leaf, boxes_output):
                        samples_per_leaves_dict[class_name][tree.name][leaf].append((x, y))
    return samples_per_leaves_dict

def compute_leaves_correlation(samples_per_leaves_dict):
    for cls, cls_data in samples_per_leaves_dict.items():
        for comb in combinations(cls_data.keys(), 2):
            tree_A = comb[0]
            tree_B = comb[1]
            
            N11, N00, N10, N01, N1X, NX1, N0X, NX0 = 0, 0, 0, 0, 0, 0, 0, 0
            for leaves in samples_per_leaves_dict[cls][tree_A], samples_per_leaves_dict[cls][tree_B]:
                leaf_A = leaves[0]
                leaf_B = leaves[1]
                for xy in set(samples_per_leaves_dict[cls][tree_A][leaf_A] + samples_per_leaves_dict[cls][tree_B][leaf_B]):
                    y = xy[1]
                    if xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy in samples_per_leaves_dict[cls][tree_B][leaf_B] and y == cls:
                        # Case 1:
                        N11 += 1
                    elif xy in samples_per_leaves_dict[cls][tree_A][leaf_A] and xy in samples_per_leaves_dict[cls][tree_B][leaf_B] and y != cls:
                        # Case 2:
                        N00 += 1
                        
                    # elif 
                        
                        
                        
            

def leaves_correlation_flow(ctx, output):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    
    create_classifier(ctx)
    samples_per_leaves_dict = samples_per_leaves(ctx.obj["classifier"])
    for c, class_data in samples_per_leaves_dict.items():
        print(c)
        for tree, tree_data in class_data.items():
            print(f"\t{tree}")
            for leaf, samples in tree_data.items():
                print(f"\t\t{leaf}: {len(samples)}")
                