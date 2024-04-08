import numpy as np, logging
from anytree import PreOrderIter
from tabulate import tabulate
from pyeda.inter import *
from .DecisionBox import *
from pyalslib import YosysHelper, ALSGraph, ALSCatalog, negate
from .DecisionTree import DecisionTree
class RegressorTree(DecisionTree):
    """ The initialization function needs to be different since a regressor doesn't have classes.
        For this reason this function firstly identify the classes by saving the score of each leaf and
        then continues with the parsing.
    """
    def __init__(self, name = None, root_node = None, features = None, classes = None, use_espresso : bool = False, weight : int = 1):
        self.name = name
        self.model_features = features
        self.attrbutes_name = [f["name"] for f in self.model_features]
        self.model_classes = classes
        self.decision_boxes = []
        self.leaves = []
        self.boolean_networks = []
        self.class_assertions = {}
        self.als_conf = None
        self.yosys_helper = None
        self.assertions_graph = None
        self.catalog = None
        self.assertions_catalog_entries = None
        self.current_als_configuration = []
        self.exact_box_output = None
        self.weight = weight
        if root_node:
            self.regressor_parse_classes(root_node)
            self.parse(root_node, use_espresso)

    """ A regressor doesn't have classes, for this reason this function simply identifies the 
        classes by saving the score of each leaf.
    """
    def regressor_parse_classes(self, root_node):
        logger = logging.getLogger("pyALS-RF")
        self.model_classes = []
        self.regressor_mapper = {}
        # For each node in the tree
        for node in PreOrderIter(root_node):
            # If the node is a leaf
            if not any(node.children):
                # Obtain its class
                self.model_classes.append(node.score)

    """ The visit function needs to be different in order to return the value of a class
        i.e. the regression value.
    """
    def visit(self, attributes):
        boxes_output = self.get_boxes_output(attributes)
        if self.als_conf is None:
            # Boolean networks are ordered by class, 
            # check the parse function.
            for class_idx, a in enumerate(self.boolean_networks):
                if eval(a["sop"], boxes_output):
                    return float(self.model_classes[class_idx]) * self.weight
        exit()
        lut_io_info = {}
        output = self.assertions_graph.evaluate(boxes_output, lut_io_info, self.current_als_configuration)[0]
        return [ o[f"\\{c}"] for c in self.model_classes ]
