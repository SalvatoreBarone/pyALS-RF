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

# Get the accuracy of a single decision tree on class c ( x_set contains only c samples).
def tree_accuracy( tree, x_set, c ):
    correctly = 0
    #for x in tqdm(x_set, desc = " Computing accuracy"):
    for x in tqdm(x_set, desc = " Computing accuracy"):
        res = np.argmax(tree.visit(x))
        if res == c: 
            correctly += 1
    return 100 * correctly / len(x_set)

def tree_visit(tree, input_samples):
    classifications = []
    for input in tqdm(input_samples, desc = "visiting"):
        pred_vec = tree.visit(input)
        # If the result is not approximated
        if np.max(pred_vec) > 0:
            classifications.append(np.argmax(tree.visit(input)))
        else:
            # Error.
            classifications.append(-1)
    return classifications

class TMR(GREP):
    
    def __init__(
                    self, classifier : Classifier,
                    pruning_set_fraction : float = 0.5,
                    ncpus : int = cpu_count(),
                    out_path : str = "./",
                    flow : str = "pruning",
                    experiment_iteration : int = 0
                ):
        super().__init__(classifier, pruning_set_fraction, max_loss = 5.0, min_resiliency = 0, ncpus = ncpus)
        self.out_path = os.path.join(out_path, "tmr") 
        logger = logging.getLogger("pyALS-RF")
        logger.info(f"Out Path  {self.out_path}")
        if os.path.exists(self.out_path) == False:
            logger.info("Generating output path")
            os.makedirs(self.out_path)
        self.flow = flow
        self.pool = Pool(self.ncpus)
        self.pruning_configuration = []
        self.exp_it = experiment_iteration

    # Approximate the function.
    def approx(self, report = True, test_samples = -1):
        # Initialize the logger 
        logger = logging.getLogger("pyALS-RF")
        # Initialize the output paths 
        report_path = os.path.join(self.out_path, "tmr.csv")
        pruning_path =  os.path.join(self.out_path, f"pruning_configuration_{self.exp_it}.json5")
        flow_store_path = os.path.join(self.out_path, ".flow.json5")
        # Split the dataset for the evaluation
        self.split_test_dataset(self.pruning_set_fraction)
        # Get the tree indexes for classes.
        self.trees_per_class = {}
        comp_time = time.time()
        # For each class, take the tree with the minimum number of classifications
        for c in self.classifier.model_classes:
            logger.info(f"[TMR] Isolating pruning samples for class {c}")
            # Get the pruning set samples associated to the specific class. 
            pruning_set_classes_x = [ ]
            for x,y in zip(self.x_pruning, self.y_pruning): 
                #print(f" Y on pruning is {str(y[0])} and c is {c} {y[0] == int(c)} {type(y[0])} {type(c)}")
                if y[0] == int(c):
                    pruning_set_classes_x.append(x)
            # Find the accuracy of each single tree for the specific class.
            logger.info(f"[TMR] Finding best trees for class {c}")
            accvec_args = [(t, pruning_set_classes_x, int(c)) for t in self.classifier.trees]        
            acc_vec = self.pool.starmap(tree_accuracy, accvec_args)
            # Get the best 3 different trees
            idx_couples = list(enumerate(acc_vec))
            idx_couples_sorted = sorted(idx_couples, key=lambda x: x[1], reverse=True)
            best_trees = [idx for idx, value in idx_couples_sorted[:3]]
            self.trees_per_class.update({int(c) : best_trees})
            logger.info(f"[TMR] Pruning remaining trees for class {c}")
            # Prune the trees not present in the configuration 
            for tree_idx in tqdm(range(0, len(self.classifier.trees)), desc = f"Removing IO Logic for class {c}"):
                # If the tree is not in the list
                if not tree_idx in best_trees:
                    # Remove all the leaves associated with the class
                    for l in self.classifier.trees[tree_idx].leaves:
                        # If the leaf is associated to the class
                        if l["class"] == c:
                            # append the leaf to the pruning configuration  
                            # The pruning configuration is : Class Label - Tree Idx - SOP 
                            self.pruning_configuration.append((c, str(tree_idx), l["sop"]))
        
        def tree_in_class(tree_idx):
            # for each class
            for c in self.classifier.model_classes:
                # If the tree having tree_idx is considered for a specific class
                # Return true
                if tree_idx in self.trees_per_class[int(c)]:
                    return True
            # The tree is not in a class, return False 
            return False
        
        def leaf_in_pruning(leaf_cfg):
            # For each leaf in pruning configuration 
            for already_pruned in self.pruning_configuration:
                # If the leaf is in the pruning cfg return true
                if already_pruned[0] == leaf_cfg[0] and already_pruned[1] == leaf_cfg[1] and already_pruned[2] == leaf_cfg[2]:
                    return True
            # else false
            return False

        logger.info("[TMR] Removing trees not in configurations")        
        # Now, for each tree not present in any best tree configuration, simply prune every leaf.
        for t in range(len(self.classifier.trees)):
            # If the tree is not in a class
            if not tree_in_class(t):
                # For each leaf
                for leaf in self.classifier.trees[t].leaves:
                    leaf_prune_cfg = (l["class"], str(t), l["sop"])
                    # If the leaf is not in the pruning cfg
                    if not leaf_in_pruning(leaf_prune_cfg):
                        self.pruning_configuration.append(leaf_prune_cfg)
            

        # Evaluate the overall computational time.
        comp_time = time.time() - comp_time
        logger.info("[TMR] Evaluating accuracy")
        # Now evaluate the accuracy of the pruned tree and save results
        # Arguments for evaluating accuracy
        self.x_validation = self.x_validation[0 : test_samples]
        self.y_validation = self.y_validation[0 : test_samples]
        self.args_evaluate_validation = [[t, self.x_validation] for t in self.classifier.p_tree]
        self.pool = self.classifier.pool
        self.baseline_accuracy_draw, self.baseline_accuracy_no_draw, exact_classifications_draw, exact_classifications_no_draw, exact_draw_counter = self.evaluate_accuracy_draw()
        self.baseline_accuracy = self.baseline_accuracy_draw # in case needed elsewhere, even if I doubt
        # Generate an unique list of trees
        self.tree_list = list({tree_idx for tmr in self.trees_per_class.values() for tree_idx in tmr})

        logger.info(f'Evaluated Baseline acc Draw considered : {self.baseline_accuracy_draw} No Draw Considered: {self.baseline_accuracy_no_draw}')

        GREP.set_pruning_conf(self.classifier, self.pruning_configuration)
        # pruned_accuracy_draw, pruned_classifications_draw, approx_draw_counter = self.visit_tmr_draw()
        # pruned_accuracy_no_draw, pruned_classifications_no_draw = self.visit_tmr_no_draw()
        pruned_accuracy_draw, pruned_accuracy_no_draw, pruned_classifications_draw, pruned_classifications_no_draw, approx_draw_counter =  self.visit_tmr_draw_multicore()
        logger.info(f'Baseline acc Draw considered : {self.baseline_accuracy_draw} No Draw Considered: {self.baseline_accuracy_no_draw} Pruned Accuracy {pruned_accuracy_draw} Pruned Accuracy No Draw {pruned_accuracy_no_draw}')
        logger.info(f'N.ro pruned leaves {len(self.pruning_configuration)}  Computational Time {comp_time}')
        logger.info(f'Exact Nro Draws. {exact_draw_counter} Approx Nro Draws {approx_draw_counter}')
        # Save the pruning configuration
        self.store_pruning_conf(pruning_path)
        # Save the flow
        with open(f"{flow_store_path}", "w") as f:
            json5.dump(self.flow, f, indent=2)
        # Save the report 
        if report :
            csv_header = [  "N.ro pruned leaves",
                            "Baseline Draw",
                            "Baseline No Draw",
                            "Pruned Accuracy", 
                            "Pruned Accuracy No Draw",
                            "Base Accuracy",
                            "Computational Time",
                            "Exp. It.",
                            "Exact Draw Ctr", 
                            "Approx Draw Ctr"]
            csv_body = [ 
                        len(self.pruning_configuration),
                        self.baseline_accuracy_draw,
                        self.baseline_accuracy_no_draw,
                        pruned_accuracy_draw,
                        pruned_accuracy_no_draw,
                        comp_time,
                        self.exp_it,
                        exact_draw_counter,
                        approx_draw_counter
                        ]
            add_csv_head = not os.path.exists(report_path)
            with open(report_path, 'a') as f:
                writer = csv.writer(f)
                if add_csv_head: 
                    writer.writerow(csv_header)
                writer.writerow(csv_body)
        # Plot tree per class.
        with open(os.path.join(self.out_path, "tree_per_class.json5"), "w") as f:
            json5.dump(self.trees_per_class, f, indent=2)
        # Now evaluating classes reports.
        labels = [int(k) for k in self.trees_per_class.keys()]
        report_exact_draw = self.calculate_class_metrics( exact_classifications_draw, labels = labels)
        report_exact_no_draw = self.calculate_class_metrics( exact_classifications_no_draw, labels = labels)
        report_pruned_draw = self.calculate_class_metrics( pruned_classifications_draw, labels = labels)
        report_pruned_no_draw = self.calculate_class_metrics( pruned_classifications_no_draw, labels = labels)
        
        with open(os.path.join(self.out_path, f"exact_draw_considered_{self.exp_it}.json"),'w') as f:
            json5.dump(report_exact_draw, f, indent=2)
        with open(os.path.join(self.out_path, f"exact_draw_not_considered_{self.exp_it}.json"),'w') as f :
            json5.dump(report_exact_no_draw, f, indent=2)
        with open(os.path.join(self.out_path, f"pruned_draw_considered_{self.exp_it}.json"),'w') as f:
            json5.dump(report_pruned_draw, f, indent=2)
        with open(os.path.join(self.out_path, f"pruned_draw_not_considered_{self.exp_it}.json"),'w') as f:
            json5.dump(report_pruned_no_draw, f, indent=2)
        
    """ This function implements the true visiting procedure for the TMR. 
        It is important to note that, in case at least two nodes identify the same class,
        then it is useless to continue. For this reason, when a class is found, the classification
        immediately stops.
        For the future, multicore implemenation will be executed.
    """
    def visit_tmr_no_draw(self):
        cc = 0
        classifications = []
        for x_val, y_val in tqdm(zip(self.x_validation, self.y_validation), desc = "Visiting tmr"):
            #print(f"Validating for sample {idx} Class {y_val[0]}")
            # For each input sample, check for each possible class.
            found = False
            for c in self.trees_per_class.keys():
                # Stop until two trees have the majority voting.
                voted_ctr = 0
                for tree_index in self.trees_per_class[int(c)]:
                    pred_vec = self.classifier.trees[tree_index].visit(x_val)
                    # np.max(vec) > 0, usefull to check if the approximated model do not find anything.
                    # In this case the model is never configured.
                    if np.argmax(pred_vec) == int(c) and np.max(pred_vec) > 0:
                        voted_ctr += 1
                if voted_ctr >= 2:
                    found = True
                    classifications.append((int(c),y_val[0]))
                    if int(c) == y_val[0]:
                        cc += 1
                    break
            # Not sure when this happens..
            # This means that all the approximated circuits made an error during classification.
            if not found:
                classifications.append((-1, y_val[0]))
        return cc / len(self.x_validation) * 100, classifications
    
    """ This function is identical to the previous one, it simply considers draw conditions as missclassifications"""
    def visit_tmr_draw(self):
        cc_draw     = 0
        cc_no_draw  = 0
        classes = []
        classifications_draw    = []
        classifications_no_draw = []
        draw_counter = 0
        for x_val, y_val in tqdm(zip(self.x_validation, self.y_validation), desc = "Visiting tmr"):
            # For each input sample, check for each possible class.
            classes = []
            # # Stopping criterion for the input sample
            # stop_for_sample = False
            for c in self.trees_per_class.keys():
                # Stop until two trees have the majority voting.
                voted_ctr = 0
                for tree_index in self.trees_per_class[int(c)]:
                    pred_vec = self.classifier.trees[tree_index].visit(x_val)
                    if np.argmax(pred_vec) == int(c) and np.max(pred_vec) > 0:
                        voted_ctr += 1
                if voted_ctr >= 2:
                    classes.append(int(c))
                    # If the first break never happen, equivalent to the break in the previous function
                    # It is identical to the number of classes.
                    if len(classes) == 1:
                        classifications_no_draw.append((int(c), y_val[0]))
                        if int(c) == y_val[0]:
                            cc_no_draw += 1
            if len(classes) == 0:
                classifications_no_draw.append((-1, y_val[0]))
            # If correctly classified
            if len(classes) == 1 and classes[0] == y_val[0]:
                cc_draw += 1
                classifications_draw.append((classes[0], y_val[0]))
            # If draw condition or approx error.
            elif len(classes) > 1:
                draw_counter += 1
                classifications_draw.append((-1, y_val[0]))
            # if approx error
            elif len(classes) == 0:
                classifications_draw.append((-1, y_val[0]))
            # If not correctly classified
            elif len(classes) == 1 and classes[0] != y_val:
                classifications_draw.append((classes[0], y_val[0]))
        return  cc_draw / len(self.x_validation) * 100, cc_no_draw /len(self.x_validation) * 100, classifications_draw, classifications_no_draw, draw_counter
    

    def visit_tmr_draw_multicore(self):
        cc_draw     = 0
        cc_no_draw  = 0
        classes = []
        classifications_draw    = []
        classifications_no_draw = []
        draw_counter = 0
        # Starmap mantains the order
        parallel_classifications = self.pool.starmap(tree_visit, [(self.classifier.trees[tree_idx], self.x_validation) for tree_idx in self.tree_list])
        #for x_val, y_val in tqdm(zip(self.x_validation, self.y_validation), desc = "Visiting tmr"):
        for val_idx in tqdm(range(0,len(self.x_validation)), desc = "Visiting tmr"):
            x_val = self.x_validation[val_idx]
            y_val = self.y_validation[val_idx]
            # For each input sample, check for each possible class.
            classes = []
            # # Stopping criterion for the input sample
            # stop_for_sample = False
            for c in self.trees_per_class.keys():
                # Stop until two trees have the majority voting.
                voted_ctr = 0
                for tree_index in self.trees_per_class[int(c)]:
                    #pred_vec = self.classifier.trees[tree_index].visit(x_val)
                    # The index of tree_idx is the index in parallel classifications
                    pred = parallel_classifications[self.tree_list.index(tree_index)][val_idx]
                    if pred == int(c) : #and np.max(pred_vec) > 0:
                        voted_ctr += 1
                if voted_ctr >= 2:
                    classes.append(int(c))
                    # If the first break never happen, equivalent to the break in the previous function
                    # It is identical to the number of classes.
                    if len(classes) == 1:
                        classifications_no_draw.append((int(c), y_val[0]))
                        if int(c) == y_val[0]:
                            cc_no_draw += 1
            if len(classes) == 0:
                classifications_no_draw.append((-1, y_val[0]))
            # If correctly classified
            if len(classes) == 1 and classes[0] == y_val[0]:
                cc_draw += 1
                classifications_draw.append((classes[0], y_val[0]))
            # If draw condition or approx error.
            elif len(classes) > 1:
                draw_counter += 1
                classifications_draw.append((-1, y_val[0]))
            # if approx error
            elif len(classes) == 0:
                classifications_draw.append((-1, y_val[0]))
            # If not correctly classified
            elif len(classes) == 1 and classes[0] != y_val:
                classifications_draw.append((classes[0], y_val[0]))
        return  cc_draw / len(self.x_validation) * 100, cc_no_draw /len(self.x_validation) * 100, classifications_draw, classifications_no_draw, draw_counter
    
    @staticmethod
    def calculate_class_metrics(predictions, labels):
        metrics = {label: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for label in labels}
        total_correct = 0
        
        for y_pred, y_real in tqdm(predictions, desc = "Evaluating TP/TN/FN/FP"):
            for label in labels:
                if y_pred == -1:  
                    if y_real != label:  
                        metrics[label]["TN"] += 1 # IN case of draw or approximation error, the model is considered as a true positive
                else:
                    if y_real == label:  
                        if y_pred == label:
                            metrics[label]["TP"] += 1  # True Positive (TP)
                            total_correct += 1
                        else:
                            metrics[label]["FN"] += 1  # False Negative (FN)
                    else:
                        if y_pred == label:
                            metrics[label]["FP"] += 1  # False Positive (FP)
                        else:
                            metrics[label]["TN"] += 1  # True Negative (TN)
        
        # Calcolo delle metriche per classe
        precision_per_class = {}
        recall_per_class = {}
        f1_per_class = {}
        accuracy_per_class = {}
        
        for label in tqdm(labels, desc = "Generating reports"):
            TP = metrics[label]["TP"]
            FP = metrics[label]["FP"]
            FN = metrics[label]["FN"]
            TN = metrics[label]["TN"]
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            
            precision_per_class[label] = precision
            recall_per_class[label] = recall
            f1_per_class[label] = f1_score
            accuracy_per_class[label] = accuracy
        
        return {
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
            "accuracy_per_class": accuracy_per_class
        }


    # # Evaluate the accuracy of a tmr approximated model. 
    # def evaluate_accuracy_tmr(self):
    #     correctly_classified_no_draw = 0
    #     number_draws = 0
    #     for x_val, y_val in tqdm(zip(self.x_validation,self.y_validation), desc = "Evaluating accuracy tmr"):
    #         correct_class = y_val[0]
    #         trees_for_class = self.trees_per_class[correct_class]
    #         # Visit all the trees 
    #         local_counter = 0
    #         # Evaluate the number of correct classification over the selected trees
    #         for tree in trees_for_class:
    #             pred_vec = self.classifier.trees[tree].visit(x_val)
    #             if np.argmax(pred_vec) == correct_class and np.max(pred_vec) > 0: 
    #                 local_counter += 1
    #         # If there are at least two trees then the classification is correct.
    #         if local_counter >= 2:
    #             # Check draw conditions 
    #             correctly_classified_no_draw += 1
    #             # Local counter for draw conditions
    #             local_counter_draw = 0
    #             # For each other TMR ( i.e. for each other class)
    #             for wrong_class, other_trees in self.trees_per_class.items():
    #                 # For other TRMS
    #                 if wrong_class != correct_class:
    #                     local_counter_draw = 0
    #                     # Check if the other tree correctly vote for their class
    #                     for ot in other_trees:
    #                         pred_vec = self.classifier.trees[ot].visit(x_val)
    #                         # When trees are approximated then maybe no class is obtained, so the argmax will return 0.
    #                         # However 0 is the index of a class, resulting in a misclassification.
    #                         if np.argmax(pred_vec) == wrong_class and np.max(pred_vec) > 0: 
    #                             local_counter_draw += 1
    #                     # If at least two trees vote for their class, we have a draw condition.
    #                     if local_counter_draw >= 2 : 
    #                         correctly_classified_no_draw -= 1
    #                         number_draws += 1
    #                         break                  
    #     return correctly_classified_no_draw / len(self.x_validation) * 100, number_draws