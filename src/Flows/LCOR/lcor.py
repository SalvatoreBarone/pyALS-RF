"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>, Antonio Emmanuele <antonio.emmanuele@unina.it>

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
from multiprocessing import cpu_count
from itertools import combinations, product
from tqdm import tqdm
from ...Model.Classifier import Classifier
from ...Model.DecisionTree import *
from ..GREP.GREP import GREP

class LCOR(GREP):
    
    def __init__(self, classifier : Classifier, pruning_set_fraction : float = 0.5, max_loss : float = 5.0, min_resiliency : int = 0, ncpus : int = cpu_count()):
        super().__init__(classifier, pruning_set_fraction, max_loss, min_resiliency, ncpus)
        self.leaf_scores = {}
        self.corr_per_leaf = {}

    # Compute the number of samples in each leaf
    # The func returns a data structure indexed by [leaf_class,tree,leaf].
    # Each element [i,j,z] contains (test_input,value) that falls into the leaf z.
    @staticmethod
    def samples_per_leaves(classifier : Classifier):
        samples_per_leaf_dict = { c : {t.name : {  l["sop"] : set() for l in t.leaves if l["class"] == c } for t in classifier.trees } for c in classifier.model_classes} 
        for x, y in  tqdm(zip(classifier.x_test, classifier.y_test), total=len(classifier.x_test), desc="Computing dataset partitions", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
            for tree in classifier.trees:
                boxes_output = tree.get_boxes_output(x) # Get decision boxes
                for class_name, assertions in tree.class_assertions.items():
                    for leaf in assertions:
                        if eval(leaf, boxes_output): # if the Assertion is true then save (x,y) in the DS
                            samples_per_leaf_dict[class_name][tree.name][leaf].add((tuple(frozenset(x)),y[0]))
        return samples_per_leaf_dict

    # Compute the correlation between each couple of leaves.
    # returns a structure indexed by [(tree_A,leaf_A),(tree_B,leaf_B)] such that:
    # DS[(tree_A,leaf_A),(tree_B,leaf_B)] =  correlation(leaf_a,leaf_b)
    @staticmethod
    def compute_leaves_correlation(samples_per_leaf_dict):
        #corr_per_leaf = {}
        corr_per_leaf_cleaned = {}
        for cls, cls_data in samples_per_leaf_dict.items(): # for each class
            for comb in combinations(cls_data.keys(), 2): # for each couple of tree.
                tree_A = comb[0] # take the pair of tree 
                tree_B = comb[1]
                for leaves in product(list(samples_per_leaf_dict[cls][tree_A].keys()),list(samples_per_leaf_dict[cls][tree_B].keys())):
                    #N11, N00, N1X, NX1, N0X, NX0 = 0, 0, 0, 0, 0, 0
                    conc, disc = 0, 0
                    Q_stat = 0
                    leaf_A = leaves[0]
                    leaf_B = leaves[1]
                    if len(samples_per_leaf_dict[cls][tree_A][leaf_A]) > 0 and len(samples_per_leaf_dict[cls][tree_B][leaf_B]) > 0 :
                        # The set here is the set of (x,y) present in at least leaf_A or leaf_B without repetition.
                        for xy in samples_per_leaf_dict[cls][tree_A][leaf_A] | samples_per_leaf_dict[cls][tree_B][leaf_B]:   
                            y = xy[1]
                            if xy in samples_per_leaf_dict[cls][tree_A][leaf_A] and xy in samples_per_leaf_dict[cls][tree_B][leaf_B]:
                                # Case 1: x,y is in both leaves 
                                conc += 1
                            elif xy in samples_per_leaf_dict[cls][tree_A][leaf_A] and xy not in samples_per_leaf_dict[cls][tree_B][leaf_B]:
                                #  Case 2: x,y is in leafA but not in leafB
                                disc += 1
                            elif xy in samples_per_leaf_dict[cls][tree_A][leaf_A] and xy not in samples_per_leaf_dict[cls][tree_B][leaf_B]:
                                #  Case 3: x,y is in leafB but not in leafA
                                disc += 1   
                        # print(f'TreeA {tree_A} leafA {leaf_A} treeB {tree_B} leafB {leaf_B}')
                        # print(f'Conc {conc} disc {disc}') 
                        Q_stat = (conc  -  disc) / (conc + disc)

                        # # print(f'Qstat {Q_stat}')
                        # if (tree_A,leaf_A) not in corr_per_leaf:
                        #     corr_per_leaf[(tree_A,leaf_A)] = {(tree_B,leaf_B) : Q_stat}
                        # else:
                        #     corr_per_leaf[(tree_A,leaf_A)].update({(tree_B,leaf_B) : Q_stat})
                        # # Compute the dual in order to remove the leaf with a greater number of literals.
                        # if (tree_B,leaf_B) not in corr_per_leaf:
                        #     corr_per_leaf[(tree_B,leaf_B)] = {(tree_A,leaf_A) : Q_stat}
                        # else:
                        #     corr_per_leaf[(tree_B,leaf_B)].update({(tree_A,leaf_A) : Q_stat})

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
        return corr_per_leaf_cleaned
                    
    # Compute the score for each leaf multiplying the sum(Q_stat(A,B))* Num literals.
    @staticmethod
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

    def init_leaves_scores(self):
        samples_per_leaf_dict = LCOR.samples_per_leaves(self.classifier)
        self.corr_per_leaf = LCOR.compute_leaves_correlation(samples_per_leaf_dict)
        self.leaf_scores = LCOR.compute_leaves_score(self.corr_per_leaf)
        print("********* Scores: ")
        for k,v in self.leaf_scores.items():
            print(f'Leaf:({k[0]},{k[1]}) score: {v}')
    