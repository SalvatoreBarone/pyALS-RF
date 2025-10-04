"""
Copyright 2021-2025 Antonio Emmanuele <antonio.emmanuele@unina.it>
                    Salvatore Barone <salvatore.barone@unina.it>
                    
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
from multiprocessing import cpu_count, Pool
from itertools import combinations, product
from tqdm import tqdm
from ...Model.Classifier import Classifier
from ...Model.DecisionTree import *
from ..GREP.GREP import GREP
import time
import csv 
import os
import json5
from scipy.stats import norm # For cut-offs.
from sklearn.model_selection import train_test_split
import re
import time
from multiprocessing import cpu_count, Pool
from pyalslib import list_partitioning

""" ALL THE FUNS WITH _2_ ARE NOT TESTED ! """
    

""" Computes the number of test set sizes to obtain an extimation of the accuracy loss.
    number of samples =                      test_set_size
                            -----------------------------------------------------
                                                                test_set_size - 1
                            1   +   error_margin^2 -----------------------------------------------------
                                                    cut_off^2 * individual_prob * (1 - individual_prob)
    test_set_size: Size of the test set.
    error_margin : Given a Probability Peval this defines the error interval size [Peval - error_margin, Peval + error_margin]
    cut_off:    The quantile of the standard normal distribution assumed a specififc confidence level (i.e. the probability that 
                the acc loss is within the interval centered in Peval).
                This value is computed internally of the function that takes as input the confidence level.
    individual_prob : The probability that a sample is present.
""" 
def compute_sample_size(test_set_size, error_margin, confidence_level, individual_prob):
    cut_off = norm.ppf(confidence_level) 
    return int(test_set_size / (1 + pow(error_margin,2) * ( (test_set_size - 1) / (pow(cut_off,2) * individual_prob * (1 - individual_prob)) ) ))

class MrAxC:
    def __recover_from_sampling_error(self, excluded_classes ):
        # Get the set of already sampled classes.
        y_flat = self.classifier.y_test.ravel()
        y_mop_flat = self.y_mop.ravel()
        # Get the set of already sampled classes.
        already_sampled_unique = list(set(self.sampled_classes))
    
        # Get the percentage in the test set of sampled classes.
        sampled_classes_testset_percentages = [np.sum(y_flat == sampled_class) for sampled_class in already_sampled_unique]
        sorted_indexes = sorted(range(len(sampled_classes_testset_percentages)), key=lambda i: sampled_classes_testset_percentages[i])  # Get sorting indices
        sampled_classes_testset_percentages = [sampled_classes_testset_percentages[i] for i in sorted_indexes]
        min_sampled_percentage = sampled_classes_testset_percentages[0]
        # Get the same value but in the mop set. 
        sampled_classes_mopset_percentages = [np.sum(y_mop_flat == sampled_class) for sampled_class in already_sampled_unique]
        sampled_classes_mopset_percentages = [sampled_classes_mopset_percentages[i] for i in sorted_indexes]
        
        
        # Perform a re-sampling for excluded classes.
        for excluded_class in excluded_classes:
            idxs = np.where(y_flat == excluded_class)[0]  # Extract indices
            excluded_class_pop_size = len(idxs)
            if excluded_class_pop_size == 0:
                self.logger.error(f"[MR-AXC] Error: The class {excluded_class} is not present in the test set despite being indicated as present.")
                exit(1)
            elif excluded_class_pop_size == 1:
                self.logger.warning(f"[MR-AXC] Warning: Only one sample for class {excluded_class} is present, it is assigned to the MOP set.")
                self.x_mop = np.append(self.x_mop, [self.classifier.x_test[idxs[0]]], axis = 0)
                self.y_mop = np.append(self.y_mop, self.classifier.y_test[idxs[0]])
                self.mop_indexes = np.append(self.mop_indexes, idxs[0])
                self.sampled_classes.append(int(excluded_class))
            else:
                self.logger.info(f"[MR-AXC] Info: Fixing sampling for class {excluded_class}")
                if excluded_class_pop_size >= min_sampled_percentage: # If it not a minority class problem, then identify the nearest sample size.
                    for idx in range(0, len(sampled_classes_testset_percentages)):
                        if excluded_class_pop_size >= sampled_classes_testset_percentages[idx]: # Untill greater ( or equal in case equal to the minimum, update mop size.)
                            mop_size = sampled_classes_mopset_percentages[idx]
                        else:
                            break
                else: # Otherwise.. simply split in half ! 
                    mop_size = int(np.ceil(excluded_class_pop_size / 2))
                self.x_mop = np.append(self.x_mop, self.classifier.x_test[idxs[:mop_size]], axis = 0)
                self.y_mop = np.append(self.y_mop, self.classifier.y_test[idxs[:mop_size]])
                self.x_val = np.append(self.x_mop, self.classifier.x_test[idxs[mop_size:]], axis = 0)
                self.y_val = np.append(self.y_mop, self.classifier.y_test[idxs[mop_size:]])
                self.mop_indexes = np.append(self.mop_indexes, idxs[:mop_size])
                self.validation_indexes = np.append(self.validation_indexes, idxs[mop_size:])
                self.sampled_classes.append(int(excluded_class))
        self.sampled_classes = sorted(self.sampled_classes)
    """ 
        Split MOP samples and validation samples.  
    """
    def sample_dse_samples(self, fraction):
        #def sample_dse_samples(self, per_class_subsampling = False):    
        classifier = self.classifier
        indexes = np.arange(0, len(classifier.x_test))
        y_flat = self.classifier.y_test.ravel()
        if fraction == None :
            mop_size = compute_sample_size(test_set_size = len(classifier.x_test), error_margin = 0.05, confidence_level = 0.95, individual_prob = 0.5)
            portion = mop_size / len(self.classifier.x_test)
            self.x_mop, self.x_val, self.y_mop, self.y_val, self.mop_indexes, self.validation_indexes = train_test_split(self.classifier.x_test, y_flat, indexes, train_size = portion)       
        else:
            self.x_mop, self.x_val, self.y_mop, self.y_val, self.mop_indexes, self.validation_indexes = train_test_split(self.classifier.x_test, y_flat, indexes, train_size = fraction)       
        
        """ It is fundamental that each test set class is in the set of sampled classes """
        self.sampled_classes = sorted([int(x) for x in list(set(self.y_mop))])
        x_test_classess= list(set(y_flat))
        # Flag added in the case a class is not used and the class is present in the test set.
        # It is fundamental to note that we don't care if the class is classified or not, if it is not present
        # in the test set ( but was present in the training set) then we can simply skip the class.
        # SO CHOOSE THE TEST SET WISELY.
        recover_class = False 
        classes_to_recover = [] # This should happen if the stratify internal check of scikit learn fails, ( we're using a custom fraction.)
        for x in x_test_classess:
            if x not in self.sampled_classes:
                self.logger.warning("[MR-AXC] Warning: In sampling, each class in the test set should be considered.")
                recover_class = True # There is at least a class to recover.
                classes_to_recover.append(x)
        # If a class was not sampled then simply insert it.
        if recover_class:
            self.logger.warning("[MR-AXC] Warning: Trying to recover from sampling error.")
            self.__recover_from_sampling_error(classes_to_recover)

    """ Compute the cost of each leaf as the number of nodes in that specific leaf. """
    def compute_leaves_costs(self):
        """ For each tree mantains the number of nodes involved in each class.  """
        self.cost_per_tree = []
        self.total_cost = 0
        for t in self.classifier.trees:
            cost_per_class = [0 for c in self.classifier.model_classes]
            for leaf in t.leaves:
                # Find the number of minterms.
                node_count = len(re.findall(r'Node_\d+', leaf["sop"]))
                # Increase the cost per each class.
                cost_per_class[int(leaf["class"])] += node_count
            # Append the cost of the single tree.
            self.cost_per_tree.append(cost_per_class)
            # Increase the total cost of the ensemble.
            self.total_cost += np.sum(cost_per_class)

    """ Transform the set of per_tree_classes (i.e. a vector where for each tree the set of classes is present)  
        into a vector where for each sample the vector of classes for each tree is considered.
    """
    def per_tree_classess_into_classes_per_tree(per_tree_classes):
        classes_per_tree = [[-1 for t in per_tree_classes] for sample in per_tree_classes[0]]
        for tree_id, tree_classes in enumerate(per_tree_classes):
            for sample_id, class_pred in enumerate(tree_classes):
                classes_per_tree[sample_id][tree_id] = class_pred
        return np.array(classes_per_tree)
    
    """ Transform the set of per_tree_classes (i.e. a vector where for each tree the set of classes is present)  
        into a vector where for each sample the vector of classes for each tree is considered.
    """
    def pertree_classess_into_perclass_pertree_acc(self, per_tree_classes, y ):
        # Iterate over the sampled classes and not the model classes.
        # It is already checked in the sample_dse_samples that, if a class is not present in the test set, it is not considered.
        perclass_pertree_acc = [[[] for t in range(len(self.classifier.trees))] for c in range(len(self.sampled_classes))]
        # For each vector of classes predicted for each tree.
        for tree_id, tree_classes in enumerate(per_tree_classes):
            for x, y_true in zip(tree_classes, y):
                # For the specific class slot y, --i.e. the true label ( direct indexing )-- append the prediction 
                # of a fixed decision tree.
                perclass_pertree_acc[self.sampled_classes.index(y_true)][tree_id].append(x) # Use index for direct indexing.        
        
        # Now compute the accuracy
        for c, class_preds in zip(self.sampled_classes, perclass_pertree_acc):
            for t_id, tree_preds in enumerate(class_preds):
                correct = 0
                for y_pred in tree_preds:
                    if y_pred == c:
                        correct += 1
                perclass_pertree_acc[self.sampled_classes.index(c)][t_id] = correct / len(tree_preds) * 100
        return perclass_pertree_acc 
    
    
    """ Given the set of predicted samples per each tree, returns for each tree the list of per_class predictions.
        per_class predictions are the predicted class for a tree, whose true label correspond to that of a specific class.
    """
    def pertree_sample_preds_2_pertree_class_preds(self, per_tree_class_preds, y):
        pertree_class_lists = [[[] for c in range(len(self.classifier.model_classes))] for i in range(len(self.classifier.trees))]
        # For each vector of classes predicted for each tree.
        for tree_id, tree_classes in enumerate(per_tree_class_preds):
            for pred, true_value in zip(per_tree_class_preds, y): # Indexing is done using the true class.
                pertree_class_lists[tree_id][true_value].append(pred)    
        return pertree_class_lists

    """ Given a list, obtained using the pertree_samples_preds_2_pertree_class_preds, this functions returns 
        the number of correctly classified samples per each tree and per each class. 
    """
    @staticmethod
    def pertree_perclass_preds_2_pertree_perclass_acc(pertree_class_list):
        pertree_perclass_acc = [[[] for c in range(len(pertree_class_list[0]))] for i in range(len(pertree_class_list))]
        for tree_id, trees_preds in enumerate(pertree_class_list):
            for class_id, per_class_preds in enumerate(trees_preds):
                for sample in per_class_preds:
                    if sample == class_id:
                        pertree_perclass_acc[tree_id][class_id] += 1
                pertree_perclass_acc[tree_id][class_id] = pertree_perclass_acc[tree_id][class_id] / len(pertree_class_list[tree_id][class_id]) * 100
        return pertree_perclass_acc
    
    """
        Get from a vector obtained through the pertree_perclass_preds_2_pertree_perclass_acc, for each class the accuracy for each tree.
    """
    @staticmethod
    def pertree_perclass_acc_2_perclass_pertree_acc(pertree_perclass_acc):
        perclass_pertree_acc = [[[] for c in range(len(pertree_perclass_acc))] for i in range(len(pertree_perclass_acc[0]))]
        for tree_id, trees_preds in enumerate(pertree_perclass_acc):
            for class_id, _ in enumerate(trees_preds):    
                perclass_pertree_acc[class_id][tree_id].append(pertree_perclass_acc[tree_id][class_id])
        return perclass_pertree_acc
    

                
    def initialize_tree_prediction_per_sample(self):
        self.logger.info("[MR-AXC] Initiating accuracy evaluation on X_MOP..")
        start = time.time()
        # self.logger.info("I've initialized XMop")
        # self.logger.info(self.x_mop)
        # exit(1)

        self.x_mop_leaves = self.classifier.compute_leaves_idx(self.x_mop)
        # self.logger.info("Ended")
        # exit(1)
        _, self.x_mop_baseline_accuracy, self.x_mop_baseline_accuracy_nodraw = self.classifier.get_accuracy_by_leaves_idx(self.x_mop_leaves, self.y_mop)
        # print(self.x_mop_baseline_accuracy)
        # print(self.classifier.x_test)
        # print(self.classifier.y_test)
        # exit(1)
        end = time.time()
        #self.x_mop_classes_transposed = self.classifier.transform_leaves_into_classess(self.x_mop_leaves)
        x_mop_classes = self.classifier.transform_leaves_into_classess(self.x_mop_leaves)
        self.x_mop_classes_transposed = x_mop_classes # This mantains the set of predictions per each tree, and is used in the Margin based heuristic.
        self.x_mop_classes = MrAxC.per_tree_classess_into_classes_per_tree(x_mop_classes) # This instead mantains per each sample the predictions of each tree 
        self.logger.info(f"[MR-AXC] Accuracy on X_MOP and Leaves initialized in ms {(end - start)* 1000}")
        self.logger.info(f"[MR-AXC] Accuracy on X_MOP :{self.x_mop_baseline_accuracy}")
        # If multicore evaluation function is used.
        if self.num_cores > 1:
            # The partitioning is ordered.
            self.p_xmop_classes = list_partitioning(self.x_mop_classes, self.num_cores)
            self.p_ymop = list_partitioning(self.y_mop, self.num_cores)
        self.logger.info("[MR-AXC] Initiating accuracy evaluation on X_VAL..")
        start = time.time()
        """ 
            I leave this commented code, to remember that I cross checked the accuracy and accuracy draw between the old method and the new one
            used to extrapolated labeled samples from the validation set.
        """
        # x_val_leaves = self.classifier.compute_leaves_idx(self.x_val)
        # self.x_val_class_labels, self.x_val_class_labels_nodraw = self.classifier.get_class_labels_by_leaves_idx(x_val_leaves, self.y_val)
        # new_accuracy, new_accuracy_nodraw = self.classifier.get_accuracy_from_labels(self.x_val_class_labels, self.x_val_class_labels_nodraw, self.y_val)
        # _, self.x_val_baseline_accuracy, self.x_val_baseline_accuracy_nodraw = self.classifier.get_accuracy_by_leaves_idx(x_val_leaves, self.y_val)
        # self.logger.info(f"[MR-AXC] NEW ACCURACY COMPUTED ON X_VAL {new_accuracy} OLD ACCURACY {self.x_val_baseline_accuracy} ")
        # self.logger.info(f"[MR-AXC] DRAW NEW ACCURACY COMPUTED ON X_VAL {new_accuracy_nodraw} OLD ACCURACY {self.x_val_baseline_accuracy_nodraw} ")
        # exit(1)
        
        x_val_leaves = self.classifier.compute_leaves_idx(self.x_val)
        self.x_val_class_labels, self.x_val_class_labels_nodraw = self.classifier.get_class_labels_by_leaves_idx(x_val_leaves, self.y_val)
        self.x_val_baseline_accuracy, self.x_val_baseline_accuracy_nodraw = self.classifier.get_accuracy_from_labels(self.x_val_class_labels, self.x_val_class_labels_nodraw, self.y_val)
        end = time.time()
        self.logger.info(f"[MR-AXC] Accuracy on X_VAL and Leaves initialized in ms {(end - start)* 1000}")
        self.logger.info(f"[MR-AXC] Accuracy on X_VAL :{self.x_val_baseline_accuracy} Draw Accuracy on X_VAL :{self.x_val_baseline_accuracy_nodraw}")
        
    """ ATTENTION: THIS FUNCTION IS DEPRECATED.
        Get the set of TMR vector predictions.
        given the set of classes per each tree (i.e. classes_per_tree) and the modular redundant configuration (i.e. class configuration)
        this function returns the output of a TMR structure ( a set of 0 or 1 for each class).
    """
    @staticmethod
    @DeprecationWarning
    def get_mr_vectors(classes_per_tree, class_configurations):
        assert len(np.shape(classes_per_tree)) == 2, "Invalid input vector, provide per each tree the list of classes for input samples"
        num_tree_per_cfg = [sum(1 for tree in cfg if tree > 0) for cfg in class_configurations]
        thds = [int(np.ceil(num_trees/2)) for num_trees in num_tree_per_cfg]
        to_ret = []
        # For each inference
        for tree_votes in classes_per_tree:
            out_vector = []
            # For each class configuration
            for c_id, config in enumerate(class_configurations):
                # If there is at least one tree in the cfg.
                if num_tree_per_cfg[c_id] > 0 :
                    # Get the predictions of the trees in configuration. 
                    tree_preds = tree_votes[config]
                    voting_trees = np.sum(tree_preds == c_id)
                    # Append 0 or 1 depending on the final outcome
                    if voting_trees > thds[c_id]:
                        out_vector.append(1)
                    else:
                        out_vector.append(0)
                else: # If the configuration has no tree directly append 0
                    out_vector.append(0)
            # Append the configuration.
            to_ret.append(out_vector)
        # Return to_ret
        return np.array(to_ret)
    

    """ Get the set of TMR vector predictions.
        given the set of classes per each tree (i.e. classes_per_tree) and the modular redundant configuration (i.e. class configuration)
        this function returns the output of a TMR structure ( a set of 0 or 1 for each class).
    """
    def get_mr_vectors(self, classes_per_tree, class_configurations):
        assert len(np.shape(classes_per_tree)) == 2, "Invalid input vector, provide per each tree the list of classes for input samples"
        num_tree_per_cfg = [sum(1 for tree in cfg if tree > 0) for cfg in class_configurations]
        thds = [int(np.ceil(num_trees/2)) for num_trees in num_tree_per_cfg]
        to_ret = []
        # For each inference
        for tree_votes in classes_per_tree:
            out_vector = []
            # For each class configuration
            for c_id, config in enumerate(class_configurations):
                # If there is at least one tree in the cfg.
                if num_tree_per_cfg[c_id] > 0 :
                    # Get the predictions of the trees in configuration. 
                    tree_preds = tree_votes[config]
                    voting_trees = np.sum(tree_preds == self.sampled_classes[c_id]) # CONSIDER ONLY THE SAMPLES CLASSES, C_ID IS THE INDEX OF THE SAMPLED CLASSES.
                    # Append 0 or 1 depending on the final outcome
                    if voting_trees > thds[c_id]:
                        out_vector.append(1)
                    else:
                        out_vector.append(0)
                else: # If the configuration has no tree directly append 0
                    out_vector.append(0)
            # Append the configuration.
            to_ret.append(out_vector)
        # Return to_ret
        return np.array(to_ret)
    

    @staticmethod
    def eval_trees_with_nodes(subtrees, samples):
        tree_labels = {}
        tree_num_nodes = {}
        for tree_id, tree in subtrees:
            tree_labels[tree_id] = []
            tree_num_nodes[tree_id] = []
            for sample in samples:
                label = tree.visit(sample)
                label_max = np.max(label)
                if label_max != 0:
                    tree_labels[tree_id].append(np.argmax(label))
                else:
                    tree_labels[tree_id].append(-1)
                tree_num_nodes[tree_id].append(0)
        return tree_labels, tree_num_nodes
    
    def mr_predict(self, class_configurations, samples, tuned_thds):
        
        if tuned_thds == None:
            thds = {}
            for cl_id, _ in enumerate(class_configurations):
                thds[cl_id] = int(np.ceil(len(class_configurations[0])/2))
        else:
            thds = tuned_thds

        # Now perform the visiting procedure of the trees.
        pool = Pool(self.num_cores)
        # Partition all the trees used for performing the inference.
        unique_trees = [(t, self.classifier.trees[t]) for t in range(len(self.classifier.trees))]
        p_trees = list_partitioning(unique_trees, self.num_cores)
        args = [(sub_trees, samples) for  sub_trees in p_trees]
        to_ret= pool.starmap(MrAxC.eval_trees_with_nodes, args)
        merged_dict_labels = {}
        # merged_dict_nodes = {}
        for labels, nodes in to_ret:
            merged_dict_labels.update(labels)  
  

        classes_per_tree = [] 
        for sampleId, _ in enumerate(samples):
            preds = []
            for treeId, _  in enumerate(self.classifier.trees):
                preds.append(merged_dict_labels[treeId][sampleId])
            classes_per_tree.append(np.array(preds))
        
        mr_vectors = self.get_mr_vectors(classes_per_tree=classes_per_tree, class_configurations=class_configurations)

        return mr_vectors
    

    def tune_thds(self, class_configurations):
        trees_list = []
        for tree_sublist in class_configurations:
            trees_list.extend(tree_sublist)
        # Get all the unique trees.
        trees_list = list(set(trees_list))
        # Now perform the visiting procedure of the trees.
        pool = Pool(self.num_cores)
        # Partition all the trees used for performing the inference.
        unique_trees = [(t, self.classifier.trees[t]) for t in trees_list]
        p_trees = list_partitioning(unique_trees, self.num_cores)
        args = [(sub_trees, self.x_mop) for  sub_trees in p_trees]
        to_ret = pool.starmap(MrAxC.eval_trees_with_nodes, args)

        merged_dict_labels = {}
        for labels, _ in to_ret:
            merged_dict_labels.update(labels)     
        
        votes_per_class = {}
           

        # For each class, identify the number of trees that labeled the samples
        for c_id, cfg in enumerate(class_configurations):
            votes_per_class[c_id] = []
            # For each sample, compute the number of votes that the value received
            for s_id, _ in enumerate(self.x_mop):
                # IF the sample is correctly labeled
                if int(self.y_mop[s_id]) == c_id:
                    votes = 0
                    # Compute the average number of trees that labels the sample
                    for tree in cfg: 
                        if merged_dict_labels[tree][s_id] == c_id : 
                            votes += 1
                    votes_per_class[c_id].append(votes)
            # print(votes_per_class)
            # print(int(np.mean(votes_per_class[c_id])))

            # In the end compute the mean as the minimum
            votes_per_class[c_id] = min(int(np.mean(votes_per_class[c_id])) - int(10 * np.std(votes_per_class[c_id])) , int(np.ceil(len(class_configurations[0]) / 2)))
        # print(votes_per_class)
        # exit(1)
        return votes_per_class
    """ 
        THIS FUNCTION IS DEPREACTED.
        Given a tmr_vector predictions and an oracle y returns the accuracy considering the draw as a missclassification and the 
        one not considering a draw as misclassification.
    """
    @staticmethod
    def get_correctly_predicted_from_vectors_static(tmr_vectors, y):
        assert len(tmr_vectors) == len(y), "The number of TMR vectors should be equal to the number of different cfgs."
        correct_draw = 0
        correct_no_draw = 0
        for vector, correct_class in zip(tmr_vectors, y):
            # Get the number of active modular redundant structures
            active_modules = np.where(np.array(vector) == 1)[0]
            nro_actives = len(active_modules)
            # If at least one cfg
            if nro_actives > 0:
                # Take always the first class.
                predicted_class = active_modules[0]
                if predicted_class == correct_class:
                    # If there is only one active module and the class is correct increase the size.
                    if nro_actives > 1 :
                        correct_no_draw += 1
                    else:
                        correct_no_draw += 1
                        correct_draw += 1

        # Return the accuracy considering the draw condition as a misclassification and the one with no missclassification.
        return correct_draw, correct_no_draw



    """ Given a tmr_vector predictions and an oracle y returns the accuracy considering the draw as a missclassification and the 
        one not considering a draw as misclassification.
    """
    def get_correctly_predicted_from_vectors(self, tmr_vectors, y):
        assert len(tmr_vectors) == len(y), "The number of TMR vectors should be equal to the number of different cfgs."
        correct_draw = 0
        correct_no_draw = 0
        for vector, correct_class in zip(tmr_vectors, y):
            # Get the number of active modular redundant structures
            active_modules = np.where(vector == 1)[0]
            nro_actives = len(active_modules)
            # If at least one cfg
            if nro_actives > 0:
                # Take always the first class.
                predicted_class = active_modules[0]
                if self.sampled_classes[predicted_class] == correct_class: # ALWAYS MAP THE INDEX OF THE TMR WITH THE CLASS
                    # If there is only one active module and the class is correct increase the size.
                    if nro_actives > 1 :
                        correct_no_draw += 1
                    else:
                        correct_no_draw += 1
                        correct_draw += 1
        # Return the accuracy considering the draw condition as a misclassification and the one with no missclassification.
        return correct_draw, correct_no_draw
    
    """ DEPRECATED
        Given a tmr_vector predictions and an oracle y returns the accuracy considering the draw as a missclassification and the 
        one not considering a draw as misclassification.
    """
    @staticmethod
    @DeprecationWarning
    def get_accuracy_from_vectors(tmr_vectors, y):
        assert len(tmr_vectors) == len(y), "The number of TMR vectors should be equal to the number of different cfgs."
        correct_draw, correct_no_draw = MrAxC.get_correctly_predicted_from_vectors(tmr_vectors, y)
        # Return the accuracy considering the draw condition as a misclassification and the one with no missclassification.
        return 100 * (correct_draw / len(y)), 100 * (correct_no_draw / len(y))
    
    """ 
        Given a tmr_vector predictions and an oracle y returns the accuracy considering the draw as a missclassification and the 
        one not considering a draw as misclassification.
    """
    def get_accuracy_from_vectors(self, tmr_vectors, y):
        assert len(tmr_vectors) == len(y), "The number of TMR vectors should be equal to the number of different cfgs."
        correct_draw, correct_no_draw = self.get_correctly_predicted_from_vectors(tmr_vectors, y)
        # Return the accuracy considering the draw condition as a misclassification and the one with no missclassification.
        return 100 * (correct_draw / len(y)), 100 * (correct_no_draw / len(y))
    

    """ DEPRECATED  """
    @staticmethod
    @DeprecationWarning
    def evaluate_mr_cfg_corr_class(per_tree_classes, y, cfg):
        pred_vectors = MrAxC.get_mr_vectors(per_tree_classes, cfg)
        return MrAxC.get_correctly_predicted_from_vectors(pred_vectors, y)
    
    
    """ Evaluates the accuracy of a configuration.
        per_tree_classes:   Vector where for each input sample, the set of votes (predicted classes), for each tree 
                            is inserted.
        y:                  Set of oracle predictions for each different tree.
        cfg:                CFG per classess ( i.e. for each different class it contains the set of trees voting for that class)
        Returns:            A tuple consisting on:
                                1-  Accuracy considering the draw condition as missclassifications.
                                2-  Accuracy considering not considering the draw conditions as missclassifications but
                                    with the first class considered.
    """
    @staticmethod
    def evaluate_mr_cfg_accuracy( per_tree_classes, y, cfg):
        pred_vectors = MrAxC.get_mr_vectors(per_tree_classes, cfg)
        return MrAxC.get_accuracy_from_vectors(pred_vectors, y)
    
    def __evaluate_xmop_single_core(self, mr_cfg):
        return MrAxC.evaluate_mr_cfg_accuracy(self.x_mop_classes, self.y_mop, mr_cfg)
    
    def __evaluate_xmop_multi_core(self, mr_cfg):
        args = [(x,y)for x, y in zip(self.p_xmop_classes, self.p_ymop)]
        corr_classified_draw_list, corr_classified_no_draw_list = self.pool.starmap(MrAxC.evaluate_mr_cfg_corr_class, args)
        return 100 * (np.sum(corr_classified_draw_list) / len(self.y_mop)), 100 * (np.sum(corr_classified_no_draw_list) /len(self.y_mop)) 
        
    """ Evaluate the accuracy on X_MOP. """
    def evaluate_mr_cfg_xmop(self, mr_cfg):
        return self.__xmop_priv_eval(mr_cfg)
    
    
    """ Given a solution, where for each class the set of trees is listed ( set of trees Per Class), transform the solution into che 
        set of classes per tree.
    """
    @staticmethod
    def cfg_per_class_in_cfg_per_tree(mr_axc, trees_per_class_cfg):
        n_trees = len(mr_axc.classifier.trees)
        per_tree_cfg = []
        for tree in range(0, n_trees):
            tree_classes = []
            for considered_class, class_cfg in enumerate(trees_per_class_cfg): # It is a list.
                if tree in class_cfg:
                    tree_classes.append(considered_class)
            per_tree_cfg.append(tree_classes)
        return per_tree_cfg
    
    """ Evaluate the savings of the current cfg.
        The cost is computed as the actual cost minus the cost of the removed parts.        
    """
    def evaluate_mr_cfg_cost(self, new_cfg):
        current_cost = self.total_cost
        # For each tree, if the class is no longer classifier 
        for tree_id, tree_costs in enumerate(self.cost_per_tree):
            # If the tree no longer classifies a class then remove the actual cost
            for class_id, class_cfg in enumerate(new_cfg):
                if tree_id not in class_cfg:
                    current_cost -= tree_costs[class_id]
        return current_cost
    
    # """ 
    #     This function accepts in input the set of predictions (in terms of leaf idx) per each
    #     different class. It returns a vector containing the set of corresponding classes per each tree.  
    # """
    # def get_per_tree_classes(self, per_tree_leaves):
    #     assert len(self.classifier.trees) == len(per_tree_leaves), "[MrAxc] The vector per_tree_leaves should mantain per each tree the set of leaf indexes"
    #     per_tree_classes = []
    #     for tree, X in zip(self.classifier.trees, per_tree_leaves):
    #             tree_classes = tree.get_classes_by_leaf_idx(X)
    #             per_tree_classes.append(tree_classes)
    #     return per_tree_classes
    
    """ Dump the validation and MOP indexes into mop_indexes and val_indexes folders. """
    def dump_mop_val_indexes(self, outdir):
        np.savetxt(os.path.join(outdir, "mop_indexes.txt"), self.mop_indexes, fmt = "%d")
        np.savetxt(os.path.join(outdir, "val_indexes.txt"), self.validation_indexes, fmt = "%d")
    
    """ Dump all the pruning configuration files. 
        This includes the:
            1- per_tree_cfg (i.e. the classes classified per each tree)
            2- per_class_cfg (i.e. the set of trees used for each class)
            3- the leaves idx for each tree.
            4- The GREP-Like pruning configuration of the Accellerator.
            5- The direction files.
    """
    def dump_cfg(self, pruning_outfiles_dict, configuration, trees_from_pruining = []):
        per_tree_cfg = MrAxC.cfg_per_class_in_cfg_per_tree(self, configuration)
        pruned_leaves = self.classifier.get_leaf_indexes_not_in_class_list(per_tree_cfg)
        # Dump the configuration per class object. 
        # for conf in configuration:
        #     for x in conf:
        #         print(type(x))
        # exit(1)
        with open(pruning_outfiles_dict["outfile_per_class_cfg"], "w") as f:
            print(configuration)
            json5.dump(configuration, f, indent = 2)
        # exit(1)
        # Dump the configuration itself.
        with open(pruning_outfiles_dict["outfile_per_tree_cfg"], "w") as f:
            json5.dump(per_tree_cfg, f, indent = 2)
        # Dump the leaf indexes.
        with open(pruning_outfiles_dict["outfile_leaves_idx"], "w") as f:
            json5.dump(pruned_leaves, f, indent = 2)
        # Dump the pruning configuration of the accellerator.
        pruning_cfg = GREP.get_pruning_cfg_from_leaves_idx(self.classifier, pruned_leaves)
        if len(trees_from_pruining) > 0:
            for tree in self.pruned_trees:
                for leaf in self.classifier.trees[tree].leaves:
                    pruning_cfg.append((str(int(leaf["class"])), str(tree), str(leaf["sop"])))
        with open(pruning_outfiles_dict["outfile_pruning_cfg"], "w") as f:
            json5.dump(pruning_cfg, f, indent = 2)
        # Dump the direction file.
        direction_file_json = self.classifier.transform_assertion_into_directions(pruning_cfg)
        with open(pruning_outfiles_dict["outfile_directions"], "w") as f:
            json5.dump(direction_file_json, f, indent = 2)
        with open(pruning_outfiles_dict["considered_classes"], "w") as f:
            json5.dump(self.sampled_classes, f, indent = 2)
        return per_tree_cfg, pruned_leaves, pruning_cfg, direction_file_json
    
    def __init__(self, classifier: Classifier, num_cores: int = 1, fraction : float = None):
        self.logger = logging.getLogger("pyALS-RF")
        self.logger.info("[MR-AXC] Initializing the module")
        self.classifier : Classifier = classifier   
        self.num_cores = num_cores
        # Select the correct function for the multicore evaluation.
        if self.num_cores > 1:
            self.__xmop_priv_eval = self.__evaluate_xmop_multi_core
        else:
            self.__xmop_priv_eval = self.__evaluate_xmop_single_core
            
        self.logger.info("[MR-AXC] Sampling classess..")
        self.sample_dse_samples(fraction)
        self.logger.info("[MR-AXC] Sampling completed.")
        self.logger.info(f"[MR-AXC] MOO-Samples: {(len(self.x_mop))}")
        self.logger.info(f"[MR-AXC] Validation-Samples: {(len(self.x_val))}")
        # Compute leaves costs
        self.logger.info(f"[MR-AXC] Initializing leaves costs.")
        self.compute_leaves_costs()
        self.logger.info(f"[MR-AXC] Computed leaves costs.")
        self.logger.info(f"[MR-AXC] Leaves costs: \r\n {self.cost_per_tree}")
        # Compute Predictions         
        self.logger.info(f"[MR-AXC] Initializing samples per leaf.")
        self.initialize_tree_prediction_per_sample()
    