"""
Copyright 2021-2024 Antonio Emmanuele <antonio.emmanuele@unina.it>
                    Salvatore Barone  <salvatore.barone@unina.it>  
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
from ..Model.Classifier import Classifier
import logging
import random
from tqdm import tqdm
import json5
from scipy.stats import norm # For cut-offs.

class FaultCollection:

    def __init__(self, classifier : Classifier, nabs = None):
        
        if nabs is not None:
            assert len(nabs) == len(classifier.model_features), "Insert a valid NABS configuration"
            self.nabs = nabs
        self.list_of_fault_sites = []   
        """ Internally each fault site is represented by a tuple containing
            0 -> The type of fault (DB/BN/Feat), specified by the indexes described in the following code.
            1 -> The fault site ( dicrionary)
            Fault Site :

            key         value
            tree_name   node_name               (DB fault type)
            tree_name   {class_idx: Minterm}    (BN fault type)
            feat_name   bit                     (Feat fault type)
        """
        self.list_of_feature_fault_sites = []
        self.list_of_db_fault_sites = []
        self.list_of_bn_fault_sites = []
        """  
            The "list_of_faults" containts, per each category, the list of fault sites.
            0 is the list of feature faults
            1 is the list of DBs faults
            2 is the list of BNs faults
        """
        self.feat_faults_idx = 0
        self.db_faults_idx = 1
        self.bn_faults_idx = 2
        self.list_of_faults = [[],[],[]]
        self.list_of_fixed_values = [[], [], []]
        
        self.get_fault_sites_features(classifier = classifier)
        self.get_fault_sites_db(classifier = classifier)
        self.get_fault_sites_bn(classifier = classifier)
        
    # Initialize the fault universe of the boolean network.
    # Implying the set of assertion functions for each possible 
    # boolean network.
    def get_fault_sites_bn(self, classifier: Classifier):
        # Boolean network is a list of dictionaries where each item is a dictionary 
        # containing all the informations for a class.
        # For each tree
        for tree in classifier.trees:
            fault_per_class = {}
            # Save the minterms for each class.
            for bn in tree.boolean_networks:    
                # For each minterm.
                for minterm in bn["minterms"]:
                    c_name = bn["class"]
                    fault_site = (self.bn_faults_idx, {tree.name : { bn["class"] : minterm}})
                    self.list_of_fault_sites.append(fault_site)
                    self.list_of_bn_fault_sites.append(fault_site)
    # Initialize all the decision boxes fault sizes.
    # The fault universe of decision boxes is the set of decision boxes per each different tree.
    def get_fault_sites_db(self, classifier : Classifier) : 
        #fault_per_tree = {}
        for tree in classifier.trees:
            #    fault_per_tree.update({tree.name: [box["name"] for box in tree.decision_boxes]})
            for box in tree.decision_boxes:
                fault_site = (self.db_faults_idx, {tree.name : box["name"]})
                self.list_of_db_fault_sites.append(fault_site)
                self.list_of_fault_sites.append(fault_site)
        #return fault_per_tree

    # Initialize the fault universe for the features of the classifier.
    def get_fault_sites_features(self, classifier: Classifier):
        for f in range(0, len(classifier.trees[0].model_features)):
            for i in range(0, 64 - self.nabs[f]):
                #fault_site = (self.feat_faults_idx, {f["name"] : i})
                fault_site = (self.feat_faults_idx, {f : i})
                self.list_of_feature_fault_sites.append(fault_site)
                self.list_of_fault_sites.append(fault_site)
    
    def print_fault_sites(self):
        logger = logging.getLogger("pyALS-RF")
        logger.info("Printing the GLOBAL LIST of identified fault sites")
        for f in self.list_of_fault_sites:
            logger.info(f)
        logger.info("Printing the list of Feature Fault sites")
        for f in self.list_of_feature_fault_sites:
            logger.info(f)
        logger.info("Printing the list of DBs Fault sites")
        for f in self.list_of_db_fault_sites:
            logger.info(f)
        logger.info("Printing the list of BNs Fault sites")
        for f in self.list_of_bn_fault_sites:
            logger.info(f)
    
    def print_faults(self):
        logger = logging.getLogger("pyALS-RF")
        logger.info("Printing the set of SAMPLED faults")
        logger.info("Feat Faults")
        for f in self.list_of_faults[self.feat_faults_idx]:
            logger.info(f)
        logger.info("DBs Faults")
        for f in self.list_of_faults[self.db_faults_idx]:
            logger.info(f)
        logger.info("BNs Faults")
        for f in self.list_of_faults[self.bn_faults_idx]:
            logger.info(f)

    # Sample the faults from a specific list of faults
    # 0 -> sample from the list of all the faults
    # 1 -> sample from the feature faults
    # 2 -> sample from the DBs faults
    # 3 -> sample from the BNs faults
    def sample_faults(self, type_of_faults: str = 0, error_margin = 0.001, confidence_level = 0.95, individual_prob = 0.5):
        logger = logging.getLogger("pyALS-RF")
        logging.info("Sampling faults")
        if type_of_faults == 0:
            fault_sites = self.list_of_fault_sites 
        elif type_of_faults == 1:
            fault_sites = self.list_of_feature_fault_sites
        elif type_of_faults == 2:
            fault_sites = self.list_of_db_fault_sites
        elif type_of_faults == 3:
            fault_sites = self.list_of_bn_fault_sites
        else:
            logger.error("Unsupported type of fault ! ")
            assert 1 == 0
        
        nro_faults = FaultCollection.compute_sample_size(len(fault_sites), error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        # Get the list of faulted indexes
        faulted_indexes = FaultCollection.__sample_faults(range(0,len(fault_sites)), nro_faults)
        # Get the set of fault sites
        faults = [fault_sites[idx] for idx in faulted_indexes]
        # Get all the fixed values
        fixed_values = self.__sample_fixed_values(faults)
        # print(len(fault_sites))
        # print(len(faults))
        # exit(1)
        # Append in list
        for f, val in tqdm(zip(faults, fixed_values), desc = "Storing sampled fault sites"):
            # f[0] is the fault_type idx while f[1] is the fault.
            self.list_of_faults[f[0]].append(f[1])
            self.list_of_fixed_values[f[0]].append(val)
            
    # Given a list of fault sites (integers) returns the number of faults 
    def __sample_faults(list_of_sites, number_of_faults):
        return random.sample(list_of_sites, number_of_faults)
    
    def __sample_fixed_values(self, faults):
        fixed_values = []
        for f in faults:
            if f[0] == self.feat_faults_idx:
                fixed_values.append(random.sample([0,1], 1)[0])
            elif f[0] == self.db_faults_idx:
                fixed_values.append(random.sample([False, True], 1)[0])
            elif f[0] == self.bn_faults_idx:
                fixed_values.append(random.sample([False, True], 1)[0])
        return fixed_values
    
    # Save the "list_of_faults" into 3 different json files.
    # These json5 files directly mantaints the output as desired by the different input.
    def faults_to_json5(self, classifier, out_path):
        logger = logging.getLogger("pyALS-RF")
        # Generate the dictionary for the output of feature faults.
        logging.info("Storing Feat faults into json")
        list_feature_names = [feat["name"] for feat in classifier.model_features]
        list_tree_names =  [tree.name for tree in classifier.trees]
        list_class_names = [class_name for class_name in classifier.model_classes]
        feat_dict = {}
        fault_list = self.list_of_faults[self.feat_faults_idx]
        fixed_vals = self.list_of_fixed_values[self.feat_faults_idx]
        for feature_name in tqdm(list_feature_names, desc = "Converting Feats fault sites into DICT JSON"):
            bit_idx_list = []
            bit_idx_fixed_values = []
            for f, val in zip(fault_list, fixed_vals):
                if list((f.keys()))[0] == feature_name:
                    bit_idx_list.append(f[feature_name])
                    bit_idx_fixed_values.append(val)
            feat_dict.update( { feature_name: {"bits": bit_idx_list, "value" : bit_idx_fixed_values} })
        # logger.info("Feat dictionary")
        # logger.info(feat_dict)

        # Generate the dictionary for the output of DBs faults.
        logging.info("Storing DBs faults into json")
        # The dictionary of DBs contains for each tree the set of nodes as 
        # following:  
        # {"0": {"Node_1" : True, "Node_2" : False}, "1": {"Node_3" : False, "Node_4": True}}
        fault_list = self.list_of_faults[self.db_faults_idx]
        fixed_vals = self.list_of_fixed_values [self.db_faults_idx]
        dbs_dict = {}
        for tree_name in tqdm(list_tree_names, desc = "Converting DBs fault sites into DICT JSON"):
            dict_per_tree = {}
            for f, val in zip(fault_list, fixed_vals):
                # If the name of the tree is correct 
                # then append to the dictionary of the BNs
                if list(f.keys())[0] == tree_name:
                    dict_per_tree.update({f[tree_name] : val})
            dbs_dict.update({tree_name: dict_per_tree})
        # logger.info("DBs faults")
        # logger.info(dbs_dict)

        # Generate the dictionary for the output of BNs faults
        # For the BNs the fault dictionary is described as following:
        # For each tree, contains for each minterm the fixed value.
        # { "0" : {"0": {'(not Node_0 and not Node_1 and Node_2 and Node_22 and Node_24)': "False"}, "1" : {'(Node_0 and not Node_28 and Node_29)': "True"}}, "1": {"0": {'(not Node_0 and not Node_1 and not Node_2 and Node_3)': "False"} }}
        logging.info("Storing BNs faults into json")
        fault_list = self.list_of_faults[self.bn_faults_idx]
        fixed_vals = self.list_of_fixed_values[self.bn_faults_idx]
        bns_dict = {}
        # For each tree
        for tree_name in tqdm(list_tree_names, desc = "Converting BNs fault sites into DICT JSON"):
            dict_per_tree = {}
            for class_name in list_class_names:
                dict_per_class = {}
                for f, val in zip(fault_list, fixed_vals): 
                    # If tree is correct 
                    if list(f.keys())[0] == tree_name:
                        # If the class is correct
                        dict_minterm = f[tree_name]
                        if list(dict_minterm.keys())[0] == class_name:
                            dict_per_class.update({dict_minterm[class_name] : val})
                dict_per_tree.update({class_name: dict_per_class})
            bns_dict.update({tree_name: dict_per_tree})
        
        with open(f"{out_path}/feat_faults.json5", "w") as f:
            json5.dump(feat_dict, f, indent = 2)
        logger.info("Done dumping feat faults")

        with open(f"{out_path}/dbs_faults.json5", "w") as f:
            json5.dump(dbs_dict, f, indent = 2)
        logger.info("Done dumping DBs faults")
        
        with open(f"{out_path}/bns_faults.json5", "w") as f:
            json5.dump(bns_dict, f, indent = 2)
        logger.info("Done dumping BNs faults")
    

    def faults_to_json5_list(self, classifier, out_path):
        logger = logging.getLogger("pyALS-RF")
        # Generate the dictionary for the output of feature faults.
        logging.info("Storing Feat faults into json")

        list_feature_names = [feat["name"] for feat in classifier.model_features]
        list_tree_names =  [tree.name for tree in classifier.trees]
        list_class_names = [class_name for class_name in classifier.model_classes]
        feat_list = []
        fault_list = self.list_of_faults[self.feat_faults_idx]
        fixed_vals = self.list_of_fixed_values[self.feat_faults_idx]

        for f, val in zip(fault_list, fixed_vals):
            feature = list(f.keys())[0]
            bit = f[feature]
            feat_list.append({feature : { bit : val}})

        # Generate the dictionary for the output of DBs faults.
        logging.info("Storing DBs faults into json")
        # The dictionary of DBs contains for each tree the set of nodes as 
        # following:  
        fault_list = self.list_of_faults[self.db_faults_idx]
        fixed_vals = self.list_of_fixed_values[self.db_faults_idx]
        dbs_list = []
        for f, val in zip(fault_list, fixed_vals):
            tree_name = list(f.keys())[0]
            node_name = f[tree_name]
            dbs_list.append({tree_name : {node_name : val}})

        # Generate the dictionary for the output of BNs faults
        # For the BNs the fault dictionary is described as following:
        # For each tree, contains for each minterm the fixed value.
        # { "0" : {"0": {'(not Node_0 and not Node_1 and Node_2 and Node_22 and Node_24)': "False"}, "1" : {'(Node_0 and not Node_28 and Node_29)': "True"}}, "1": {"0": {'(not Node_0 and not Node_1 and not Node_2 and Node_3)': "False"} }}
        logging.info("Storing BNs faults into json")
        fault_list = self.list_of_faults[self.bn_faults_idx]
        fixed_vals = self.list_of_fixed_values[self.bn_faults_idx]
        bns_list = []
        for f, val in zip(fault_list, fixed_vals): 
            tree_name   = list(f.keys())[0]
            class_name  = list(f[tree_name].keys())[0]
            minterm     = f[tree_name][class_name]
            bns_list.append({tree_name : { class_name : { minterm : val }}})

        with open(f"{out_path}/feat_faults.json5", "w") as f:
            json5.dump(feat_list, f, indent = 2)
        logger.info("Done dumping feat faults")

        with open(f"{out_path}/dbs_faults.json5", "w") as f:
            json5.dump(dbs_list, f, indent = 2)
        logger.info("Done dumping DBs faults")
        
        with open(f"{out_path}/bns_faults.json5", "w") as f:
            json5.dump(bns_list, f, indent = 2)
        logger.info("Done dumping BNs faults") 
    
    """ Computes the number of injected faults using the Leveugle formula.
        number of faults =                      fault_universe_size
                                -----------------------------------------------------
                                                                    fault_universe_size - 1
                                1   +   error_margin^2 -----------------------------------------------------
                                                        cut_off^2 * individual_prob * (1 - individual_prob)
        fault_universe_size: Number of possible faults
        error_margin : Given a Probability Peval this defines the error interval size [Peval - error_margin, Peval + error_margin]
        cut_off:    The quantile of the standard normal distribution assumed a specififc confidence level (i.e. the probability that 
                    the detected faults are within the interval centered in Peval).
                    This value is computed internally of the function that takes as input the confidence level.
        individual_prob : The probability that a member of the fault universe size is malfunctioning (Usually set as 1/2)
    """ 
    def compute_sample_size(fault_universe_size, error_margin, confidence_level, individual_prob):
        cut_off = norm.ppf(confidence_level) 
        return int(fault_universe_size / (1 + pow(error_margin,2) * ( (fault_universe_size - 1) / (pow(cut_off,2) * individual_prob * (1 - individual_prob)) ) ))