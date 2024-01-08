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
    def samples_per_leaves(self,classifier : Classifier):
        samples_per_leaf_dict = { c : {t.name : {  l["sop"] : set() for l in t.leaves if l["class"] == c } for t in classifier.trees } for c in classifier.model_classes} 
        #for x, y in  tqdm(zip(classifier.x_test, classifier.y_test), total=len(classifier.x_test), desc="Computing dataset partitions", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        for x, y in  tqdm(zip(self.x_pruning, self.y_pruning), total=len(self.x_pruning), desc="Computing samples per leaf", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
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

                        # Consider the statistic only in case it is positively correlated.
                        # Since we're discarding the "accuracy" of a leaf, Q is positive when two leaves are
                        # taking the same decisions, thus one of them could be eliminated.
                        if Q_stat > 0 : 
                            if (cls,tree_A,leaf_A) not in corr_per_leaf_cleaned: # In case the leaf entry was not present in the map
                                corr_per_leaf_cleaned[(cls,tree_A,leaf_A)] = {(cls,tree_B,leaf_B) : Q_stat}
                            else:   # Otherwise insert the new statistic.
                                corr_per_leaf_cleaned[(cls,tree_A,leaf_A)].update({(cls,tree_B,leaf_B) : Q_stat})
                            # Compute the dual in order to remove the leaf with a greater number of literals.
                            if (cls,tree_B,leaf_B) not in corr_per_leaf_cleaned:
                                corr_per_leaf_cleaned[(cls,tree_B,leaf_B)] = {(cls,tree_A,leaf_A) : Q_stat}
                            else:
                                corr_per_leaf_cleaned[(cls,tree_B,leaf_B)].update({(cls,tree_A,leaf_A) : Q_stat})
            
        #print(f'conc {conc} disc {disc} total number of samples {len(samples_set)}')
        return corr_per_leaf_cleaned
                    
    # Compute the score for each leaf multiplying the sum(Q_stat(A,B))* Num literals.
    def compute_leaves_score(self,corr_per_leaf):
        scores = {}
        for leafA in corr_per_leaf:
            # Initialize the vector of scores only for the leaves that
            # are not already pruned.
            if leafA not in self.pruning_configuration:
                scores[leafA] = 0
        # for leafA,correlated in corr_per_leaf.items():
        #     #if leafA not in self.pruning_conf:
        #     num_and = 0
        #     for leaves,Q_val in correlated.items():
        #         scores[leafA] += Q_val
        #     splitted = leafA[2].split()
        #     for and_w in splitted:
        #         if and_w == "and":
        #             num_and += 1
        #     #print(f'Name {leafA[1]} num {num_and}')
        #     scores[leafA] = scores[leafA] * num_and # multiply by the number of literals.
        
        for leafA in scores.keys():
            #if leafA not in self.pruning_conf:
            num_and = 0
            for leafB in scores.keys():
                if leafB in corr_per_leaf[leafA]: # leafB != leafA, should also check this but for construction it leafA is not in the correlated list of leafA
                    scores[leafA] += corr_per_leaf[leafA][leafB]
            # After summing the correlations of leafA multiply by the number of and of leafA
            splitted = leafA[2].split()
            for and_w in splitted:
                if and_w == "and":
                    num_and += 1
            #print(f'Name {leafA[1]} num {num_and}')
            scores[leafA] = scores[leafA] * num_and # multiply by the number of literals.
        return scores
    
    # Initialize the scores
    def init_leaves_scores(self):
        samples_per_leaf_dict = self.samples_per_leaves(self.classifier)
        self.corr_per_leaf = LCOR.compute_leaves_correlation(samples_per_leaf_dict)
        scores = self.compute_leaves_score(self.corr_per_leaf)
        # self.leaf_scores is a list of tuples (leaf={tree,class,leaf}, score) sorted by score
        self.leaf_scores = sorted(scores.items(), key=lambda x: x[1],reverse = True) # Sort scores
        print("********* Scores: ")
        for k in self.leaf_scores:
            print(f'Class {k[0][0]} Tree:{k[0][1]} Leaf:{k[0][2]}')
            # print(f'C_type:{type(k[0][0])} Tree_type {type(k[0][1])} Leaf_type {type(k[0][2])}')
            print(f'Score: {k[1]}')
    

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
    def trim(self):
        super().trim(GREP.CostCriterion.depth) # cost_criterion is useless.
        self.init_leaves_scores() # compute the scores, ordering them, it must be executed after splitting the DS.
        #self.baseline_accuracy = self.evaluate_accuracy() # set the base accuracy, should be already done by trim.
        scores_idx = 0      # Index used for the scores list
        pruned_leaves = 0   # Number of pruned leaves
        final_acc = 0
        nro_candidates = len(self.leaf_scores)
        while self.loss < self.max_loss and len(self.leaf_scores) > scores_idx:
            tentative = copy.deepcopy(self.pruning_configuration) # save the pruning conf.
            leaf_id = self.leaf_scores[scores_idx][0] # Save the leaf id to try.  
            tentative.append(leaf_id)  # append the element with the best value.
            GREP.set_pruning_conf(self.classifier, tentative) # Set the pruning configuration
            self.accuracy = self.evaluate_accuracy() # Evaluate the accuracy
            loss = self.baseline_accuracy - self.accuracy # compute the loss
            if loss <= self.max_loss:   # If the loss is acceptable
                self.loss = loss        # Update the loss 
                final_acc = self.accuracy  # Save the real accuracy
                self.pruning_configuration.append(leaf_id) # Insert the leaf into the pruning configuration
                pruned_leaves += 1  # Increase the number of pruned leaves
                print(f'Actual acc {self.accuracy} Base {self.baseline_accuracy}')
                print(f'Idx {scores_idx} Max LEN {len(self.leaf_scores)}')
                print(f'Actual loss {self.loss}')
                scores_idx = 0 # Reset the score index
                scores = self.compute_leaves_score(self.corr_per_leaf) # Just need to recompute the scores and sort them.
                self.leaf_scores = sorted(scores.items(), key=lambda x: x[1],reverse = True) # Sort scores
            else :
                scores_idx += 1
                # print("Retrying")
                # print(f'Actual acc {self.accuracy} Base {self.baseline_accuracy}')
                # print(f'Idx {scores_idx} Max LEN {len(self.leaf_scores)}')
        print(f' N.ro candidates {nro_candidates}  N.ro pruned leaves {pruned_leaves}')
        print(f' Loss {self.loss} idx {scores_idx} remaining leaves {len(self.leaf_scores)}')
        print(f' Accuracy {final_acc} Base Accuracy{self.baseline_accuracy}')
        print(f' Max loss {self.max_loss}')
    # This function is useful for testing the score update function when the pruning set is changed. 
    # def append_pruning_conf(self,p_conf):
    #     super().trim(GREP.CostCriterion.depth) # cost_criterion is useless.
    #     for conf in p_conf:
    #         self.pruning_configuration.append(conf)