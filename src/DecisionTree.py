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
import os
from pyosys import libyosys as ys
from jinja2 import Environment, FileSystemLoader
from distutils.dir_util import mkpath
from anytree import PreOrderIter
from pyeda.inter import *
from .DecisionBox import *
from pyalslib import YosysHelper, ALSGraph, ALSCatalog, ALSRewriter, negate
from multiprocessing import cpu_count
class DecisionTree:

    def __init__(self, name = None, root_node = None, features = None, classes = None):
        self.name = name
        self.model_features = features
        self.attrbutes_name = [f["name"] for f in self.model_features]
        self.model_classes = classes
        self.decision_boxes = []
        self.leaves = []
        self.assertions = []
        self.class_minterms = {}
        if root_node:
            self.get_decision_boxes(root_node)
            self.get_leaves(root_node)
            self.get_assertions()
            self.get_minterms_for_classes()
        self.als_conf = None
        self.yosys_helper = None
        self.assertions_graph = None
        self.catalog = None
        self.assertions_catalog_entries = None
        self.current_als_configuration = []
        self.exact_box_output = None
        
        
            

    def __deepcopy__(self, memo = None):
        tree = DecisionTree()
        tree.name = copy.deepcopy(self.name)
        tree.model_features = copy.deepcopy(self.model_features)
        tree.model_classes = copy.deepcopy(self.model_classes)
        tree.decision_boxes = copy.deepcopy(self.decision_boxes)
        tree.assertions = copy.deepcopy(self.assertions)
        tree.als_conf = copy.deepcopy(self.als_conf)
        tree.assertions_graph = copy.deepcopy(self.assertions_graph)
        tree.catalog = copy.deepcopy(self.catalog)
        tree.assertions_catalog_entries = copy.deepcopy(self.assertions_catalog_entries)
        tree.current_als_configuration = copy.deepcopy(self.current_als_configuration)
        tree.self.exact_box_output = copy.deepcopy(self.exact_box_output)
        return tree
    
    def brace4ALS(self, als_conf):
        if als_conf is None:
            self.als_conf = als_conf
            self.yosys_helper = YosysHelper()
            HDLGenerator.generate_design_for_als(self, self.als_conf.cut_size)
            self.assertions_graph = ALSGraph(self.yosys_helper.design)
            self.assertions_catalog_entries = ALSCatalog(self.als_conf.lut_cache, self.als_conf.solver).generate_catalog(self.yosys_helper.get_luts_set(), self.als_conf.timeout, ncpus)
            self.set_assertions_configuration([0] * self.assertions_graph.get_num_cells())
            self.yosys_helper.save_design(self.name)
         

    def get_total_bits(self):
        return 64 * len(self.decision_boxes)

    def get_total_nabs(self):
        return sum(box["box"].get_nab() for box in self.decision_boxes)

    def get_total_retained(self):
        return 64 * len(self.decision_boxes) - self.get_total_nabs()

    def get_assertions_distance(self):
        return [ self.current_als_configuration[c]["dist"] for c in self.current_als_configuration.keys() ]

    def get_current_required_aig_nodes(self):
        return sum(self.current_als_configuration[c]["gates"] for c in self.current_als_configuration.keys())

    def reset_assertion_configuration(self):
        if self.assertions_graph is not None:
            self.set_assertions_configuration([0] * self.assertions_graph.get_num_cells())

    def get_als_dv_upper_bound(self):
        return [len(e) - 1 for c in [{"name": c["name"], "spec": c["spec"]} for c in self.assertions_graph.get_cells()] for e in self.assertions_catalog_entries if e[0]["spec"] == c["spec"] or negate(e[0]["spec"]) == c["spec"]]

    def set_nabs(self, nabs):
        for box in self.decision_boxes:
            box["box"].set_nab(nabs[box["box"].get_feature()])

    def set_assertions_configuration(self, configuration):
        assert len(configuration) == self.assertions_graph.get_num_cells(), f"wrong amount of variables. Needed {self.assertions_graph.get_num_cells()}, get {len(configuration)}"
        assert len(self.assertions_catalog_entries) > 0, "Catalog cannot be empty"
        matter = {}
        for i, (c, l) in enumerate(zip(configuration, self.assertions_graph.get_cells())):
            for e in self.assertions_catalog_entries:
                try:    
                    if e[0]["spec"] == l["spec"]:
                        matter[l["name"]] = { "dist": c, "spec": e[0]["spec"], "axspec": e[c]["spec"], "gates": e[c]["gates"], "S": e[c]["S"], "P": e[c]["P"], "out_p": e[c]["out_p"], "out": e[c]["out"], "depth": e[c]["depth"]}
                    elif negate(e[0]["spec"]) == l["spec"]:
                        matter[l["name"]] = { "dist": c, "spec": negate(e[0]["spec"]), "axspec": negate(e[c]["spec"]), "gates": e[c]["gates"], "S": e[c]["S"], "P": e[c]["P"], "out_p": 1 - e[c]["out_p"], "out": e[c]["out"], "depth": e[c]["depth"]}
                except IndexError as err:
                    ub = self.get_als_dv_upper_bound()
                    print(err)
                    print(f"Tree: {self.name}")
                    print(f"Configuration: {configuration}")
                    print(f"Configuration length: {len(configuration)}")
                    print(f"Upper bound: {ub}")
                    print(f"Upper bound length: {len(ub)}")
                    print(f"Configuration[{i}]: {c}")
                    print(f"Upper bound[{i}]: {ub[i]}")
                    print(f"Cell: {l}")
                    print(f"Catalog Entries #: {len(e)}")
                    print(f"Catalog Entries: {e}")
                    exit()
        self.current_als_configuration = matter
        
    def dump(self):
        print("\tName: ", self.name)
        print("\tBoxes:")
        for b in self.decision_boxes:
            print("\t\t",  b["box"].get_name(), "(", b["box"].get_feature(), " " , b["box"].get_c_operator(), " ", b["box"].get_threshold(), "), nab ", b["box"].get_nab())
        print("\tAssertions:")
        for a in self.assertions:
            print("\t\t", a["class"], " = ", a["expression"])

    def predict(self, attributes):
        boxes_output = {
            box["box"].get_name() if self.als_conf is None else "\\" + box["box"].get_name() : 
                box["box"].compare(attributes[self.attrbutes_name.index(box["box"].get_feature())])
            for box in self.decision_boxes
        }
        if self.als_conf is None:
            return [eval(a["expression"], boxes_output) for a in self.assertions ]
        exit()
        lut_io_info = {}
        output = self.assertions_graph.evaluate(boxes_output, lut_io_info, self.current_als_configuration)[0]
        return [ o[f"\\{c}"] for c in self.model_classes ]

    def get_active_minterm(self, attributes):
        boxes_output = {
            box["box"].get_name() if self.als_conf is None else "\\" + box["box"].get_name() : 
                box["box"].compare(attributes[self.attrbutes_name.index(box["box"].get_feature())])
            for box in self.decision_boxes
        }
        for k, v in self.class_minterms.items():
            for m in v:
                if eval(m, boxes_output):
                    return k, m

    def get_decision_boxes(self, root_node):
        self.decision_boxes = []
        for node in PreOrderIter(root_node):
            if any(node.children):
                try:
                    feature = next(item for item in self.model_features if item["name"] == node.feature)
                    self.decision_boxes.append({
                        "name" : node.name,
                        "box"  : DecisionBox(node.name, node.feature, feature["type"], node.operator, node.threshold_value)})
                except Exception:
                    print(node.feature, "Feature not found")
                    print("Recognized model features", self.model_features)
                    exit()

    def get_leaves(self, root_node):
        self.leaves = [{"name": node.name, "class": node.score, "expression": f"({str(node.boolean_expression)})"} for node in PreOrderIter(root_node) if not any(node.children)]

    def get_assertion(self, class_name):
        conditions = [item["expression"] for item in self.leaves if item["class"] == class_name]
        if not conditions:
            return "False"
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return " | ".join(conditions)

    def get_assertions(self):
        self.assertions = []
        for class_name in self.model_classes:
            assertion_function = self.get_assertion(class_name)
            hdl_expression = str(espresso_exprs(expr(assertion_function))[0]).replace("~", "not ").replace("Or","func_or").replace("And","func_and")
            self.assertions.append({
                "class"      : class_name,
                "expression" : assertion_function.replace("~", "not ").replace("|", "or").replace("&", "and"),
                "minimized"  : "'0'" if assertion_function == "False" else hdl_expression})


    def get_minterms_for_classes(self):
        self.class_minterms = { c : [item["expression"].replace("~", "not ").replace("|", "or").replace("&", "and") for item in self.leaves if item["class"] == c] for c in self.model_classes}  