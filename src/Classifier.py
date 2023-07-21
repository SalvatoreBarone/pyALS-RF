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
import numpy as np, csv, sys, pandas as pd
from xml.etree import ElementTree
from anytree import Node
from jinja2 import Environment, FileSystemLoader
from distutils.file_util import copy_file
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from pyalslib import list_partitioning
from .DecisionTree import *

class Classifier:
    __namespaces = {'pmml': 'http://www.dmg.org/PMML-4_4'}

    def __init__(self, ncpus = None):
        self.trees = []
        self.model_features = []
        self.model_classes = []
        self.ncpus = min(ncpus, cpu_count()) if ncpus is not None else cpu_count()
        self.args = None
        self.pool = None
        self.als_conf = None

    def __deepcopy__(self, memo=None):
        classifier = Classifier(self.als_conf)
        classifier.trees = copy.deepcopy(self.trees)
        classifier.model_features = copy.deepcopy(self.model_features)
        classifier.model_classes = copy.deepcopy(self.model_classes)
        classifier.ncpus = self.ncpus
        classifier.args = None
        classifier.pool = None
        classifier.als_conf = None
        return classifier

    def parse(self, pmml_file_name):
        self.trees = []
        self.model_features = []
        self.model_classes = []
        tree = ElementTree.parse(pmml_file_name)
        root = tree.getroot()
        self.__namespaces["pmml"] = get_xmlns_uri(root)
        self.get_features_and_classes(root)
        segmentation = root.find("pmml:MiningModel/pmml:Segmentation", self.__namespaces)
        if segmentation is not None:
            for tree_id, segment in enumerate(segmentation.findall("pmml:Segment", self.__namespaces)):
                print(f"Parsing tree {tree_id}... ")
                tree_model_root = segment.find("pmml:TreeModel", self.__namespaces).find("pmml:Node", self.__namespaces)
                tree = self.get_tree_model(str(tree_id), tree_model_root)
                self.trees.append(tree)
            print("\rDone")
        else:
            tree_model_root = root.find("pmml:TreeModel", self.__namespaces).find(
                "pmml:Node", self.__namespaces)
            tree = self.get_tree_model("0", tree_model_root)
            self.trees.append(tree)
        self.ncpus = min(self.ncpus, len(self.trees))
        
    def wc_parse(self, pmml_file_name, ncpus):
        self.trees = []
        self.model_features = []
        self.model_classes = []
        tree = ElementTree.parse(pmml_file_name)
        root = tree.getroot()
        self.__namespaces["pmml"] = get_xmlns_uri(root)
        self.get_features_and_classes(root)
        segmentation = root.find("pmml:MiningModel/pmml:Segmentation", self.__namespaces)
        assert segmentation is not None, "This mode is suitable only for WC DT-based MCSs"
        
        segments = segmentation.findall("pmml:Segment", self.__namespaces)
        tree = None
        for tree_id, segment in enumerate(segments):
            print(f"Parsing tree {tree_id}... ")
            if tree is None:
                tree_model_root = segment.find("pmml:TreeModel", self.__namespaces).find("pmml:Node", self.__namespaces)
                tree = self.get_tree_model("tree_0", tree_model_root, ncpus)
            print("\rDone")
        self.trees = [tree] + [ copy.deepcopy(tree) for _ in range(len(segments) - 1) ]
        for i, t in enumerate(self.trees[1:]):
            t.set_name(f"tree_{i+1}")
        self.ncpus = min(self.ncpus, len(self.trees))
        
    def wc_fix_ys_helper(self):
        for t in self.trees[1:]:
            t.yosys_helper = copy.deepcopy(self.trees[0].yosys_helper)

    def dump(self):
        print("Features:")
        for f in self.model_features:
            print("\tName: ", f["name"], ", Type: ", f["type"])
        print("\n\nClasses:")
        for c in self.model_classes:
            print("\tName: ", c)
        print("\n\nTrees:")
        for t in self.trees:
            t.dump()
            
    def read_dataset(self, dataset_csv, dataset_description):
        self.dataframe = pd.read_csv(dataset_csv, sep = ";")
        attribute_name = list(self.dataframe.keys())[:-1]
        assert len(attribute_name) == len(self.model_features), f"Mismatch in features vectors. Read {len(attribute_name)} features, buth PMML says it must be {len(self.model_features)}!"
        f_names = [ f["name"] for f in self.model_features]
        assert attribute_name == f_names, f"{attribute_name} != {f_names}"
        self.x_test = self.dataframe.loc[:, self.dataframe.columns != "Outcome"].values.tolist()
        self.y_test = sum(self.dataframe.loc[:, self.dataframe.columns == "Outcome"].values.tolist(), [])

    def brace4ALS(self, als_conf):
        self.als_conf = als_conf
        for t in self.trees:
            t.brace4ALS(als_conf)

    def reset_nabs_configuration(self):
        self.set_nabs({f["name"]: 0 for f in self.model_features})

    def reset_assertion_configuration(self):
        for t in self.trees:
            t.reset_assertion_configuration()

    def set_nabs(self, nabs):
        for tree in self.trees:
            tree.set_nabs(nabs)

    def set_assertions_configuration(self, configurations):
        for t, c in zip(self.trees, configurations):
            t.set_assertions_configuration(c)

    def set_first_stage_approximate_implementations(self, configuration):
        for t, c in zip(self.trees, configuration):
            t.set_first_stage_approximate_implementations(c)

    def get_num_of_trees(self):
        return len(self.trees)

    def get_total_bits(self):
        return sum(t.get_total_bits() for t in self.trees)

    def get_total_retained(self):
        return sum(t.get_total_retained() for t in self.trees)

    def get_als_cells_per_tree(self):
        return [len(t.get_graph().get_cells()) for t in self.trees]

    def get_als_dv_upper_bound(self):
        ub = []
        for t in self.trees:
            ub.extend(iter(t.get_als_dv_upper_bound()))
        return ub

    def get_assertions_configuration(self):
        return [t.get_assertions_configuration() for t in self.trees]

    def get_assertions_distance(self):
        return [t.get_assertions_distance() for t in self.trees]

    def get_current_required_aig_nodes(self):
        return [t.get_current_required_aig_nodes() for t in self.trees]

    def get_num_of_first_stage_approximate_implementations(self):
        return [len(t.get_first_stage_approximate_implementations()) - 1 for t in self.trees]

    def get_struct(self):
        return [tree.get_struct() for tree in self.trees]

    def enable_mt(self):
        self.args = [[t, self.x_test] for t in list_partitioning(self.trees, self.ncpus)]
        self.pool = Pool(self.ncpus)

    def evaluate_test_dataset(self):
        if self.args is None:
            print("Warning!\nMulti-threading is disabled. To enable it, call the enable_mt() member of the Classifier class")
            return sum(np.argmax(self.predict(x)) == y for x, y in tqdm( zip(self.x_test, self.y_test), total=len(self.y_test), desc="Computing accuracy...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False) ) / len(self.y_test) * 100
        outcomes = self.pool.starmap(Classifier.tree_predict, self.args)
        return sum( np.argmax([sum(s) for s in zip(*scores)]) == y for scores, y in zip(zip(*outcomes), self.y_test) ) / len(self.y_test) * 100
            
    @staticmethod
    def tree_predict(trees, x_test):
        scores = []
        for x in x_test:
            outcomes = [ t.predict(x) for t in trees ]
            scores.append([sum(s) for s in zip(*outcomes)])
        return scores
         
    def predict(self, attributes):
        outcomes = [ t.predict(attributes) for t in self.trees ]
        return [sum(s) for s in zip(*outcomes)]
    
    def get_features_and_classes(self, root):
        for child in root.find("pmml:DataDictionary", self.__namespaces).findall('pmml:DataField', self.__namespaces):
            if child.attrib["optype"] == "continuous":
                # the child is PROBABLY a feature
                self.model_features.append({
                    "name": child.attrib['name'].replace('-', '_'),
                    "type": "double" if child.attrib['dataType'] == "double" else "int"})
            elif child.attrib["optype"] == "categorical":
                # the child PROBABLY specifies model-classes
                for element in child.findall("pmml:Value", self.__namespaces):
                    self.model_classes.append(element.attrib['value'].replace('-', '_'))

    def get_tree_model(self, tree_name, tree_model_root, id=0):
        tree = Node(f"Node_{tree_model_root.attrib['id']}" if "id" in tree_model_root.attrib else f"Node_{id}", feature="", operator="", threshold_value="", boolean_expression="")
        self.get_tree_nodes_recursively(tree_model_root, tree, id)
        return DecisionTree(tree_name, tree, self.model_features, self.model_classes)

    def get_tree_nodes_recursively(self, element_tree_node, parent_tree_node, id=0):
        children = element_tree_node.findall("pmml:Node", self.__namespaces)
        assert len(
            children) == 2, f"Only binary trees are supported. Aborting. {children}"
        for child in children:
            boolean_expression = parent_tree_node.boolean_expression
            if boolean_expression:
                boolean_expression += " & "
            predicate = None
            if compound_predicate := child.find("pmml:CompoundPredicate", self.__namespaces):
                predicate = next(item for item in compound_predicate.findall(
                    "pmml:SimplePredicate", self.__namespaces) if item.attrib["operator"] != "isMissing")
            else:
                predicate = child.find("pmml:SimplePredicate", self.__namespaces)
            if predicate is not None:
                feature = predicate.attrib['field'].replace('-', '_')
                operator = predicate.attrib['operator']
                threshold_value = predicate.attrib['value']
                if operator in ('equal', 'lessThan', 'greaterThan'):
                    parent_tree_node.feature = feature
                    parent_tree_node.operator = operator
                    parent_tree_node.threshold_value = threshold_value
                    boolean_expression += parent_tree_node.name
                else:
                    boolean_expression += f"~{parent_tree_node.name}"
            if child.find("pmml:Node", self.__namespaces) is None:
                Node(f"Node_{child.attrib['id']}" if "id" in child.attrib else f"Node_{id}", parent=parent_tree_node,
                    score=child.attrib['score'].replace('-', '_'), boolean_expression=boolean_expression)
            else:
                new_tree_node = Node(f"Node_{child.attrib['id']}" if "id" in child.attrib else f"Node_{id}",
                                    parent=parent_tree_node, feature="", operator="", threshold_value="", boolean_expression=boolean_expression)
                self.get_tree_nodes_recursively(child, new_tree_node, id + 1)


def get_xmlns_uri(elem):
  if elem.tag[0] == "{":
    uri, ignore, tag = elem.tag[1:].partition("}")
  else:
    uri = None
  return uri
