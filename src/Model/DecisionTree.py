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
import numpy as np, logging
from anytree import PreOrderIter
from pyeda.inter import *
from .DecisionBox import *
from pyalslib import YosysHelper, ALSGraph, ALSCatalog, negate

class DecisionTree:

    def __init__(self, name = None, root_node = None, features = None, classes = None, use_espresso : bool = False):
        self.name = name
        self.model_features = features
        self.attrbutes_name = [f["name"] for f in self.model_features]
        self.model_classes = classes
        self.decision_boxes = []
        self.leaves = []
        self.boolean_networks = []
        self.pruned_boolean_nets = []
        self.class_assertions = {}
        self.als_conf = None
        self.yosys_helper = None
        self.assertions_graph = None
        self.catalog = None
        self.assertions_catalog_entries = None
        self.current_als_configuration = []
        self.exact_box_output = None
        if root_node:
            self.parse(root_node, use_espresso)

    def __deepcopy__(self, memo = None):
        tree = DecisionTree()
        tree.name = copy.deepcopy(self.name)
        tree.model_features = copy.deepcopy(self.model_features)
        tree.model_classes = copy.deepcopy(self.model_classes)
        tree.decision_boxes = copy.deepcopy(self.decision_boxes)
        tree.boolean_networks = copy.deepcopy(self.boolean_networks)
        tree.class_assertions = copy.deepcopy(self.class_assertions)
        tree.pruned_boolean_nets = copy.deepcopy(self.pruned_boolean_nets)
        
        tree.als_conf = copy.deepcopy(self.als_conf)
        tree.assertions_graph = copy.deepcopy(self.assertions_graph)
        tree.catalog = copy.deepcopy(self.catalog)
        tree.assertions_catalog_entries = copy.deepcopy(self.assertions_catalog_entries)
        tree.current_als_configuration = copy.deepcopy(self.current_als_configuration)
        tree.exact_box_output = copy.deepcopy(self.exact_box_output)
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
            box["box"].nab = nabs[box["box"].feature_name]

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
        for a in self.boolean_networks:
            print("\t\t", a["class"], " = ", a["boolean_net"])

    def get_boxes_output(self, attributes):
        return {box["box"].name if self.als_conf is None else "\\" + box["box"].name() : box["box"].compare(attributes[self.attrbutes_name.index(box["box"].feature_name)]) for box in self.decision_boxes}
    
    def visit(self, attributes, use_pruned = False):
        boxes_output = self.get_boxes_output(attributes)
        if self.als_conf is None:
            return ([eval(a["boolean_net"], boxes_output) for a in self.pruned_boolean_nets ] if use_pruned else [eval(a["boolean_net"], boxes_output) for a in self.boolean_networks ])
        exit()
        lut_io_info = {}
        output = self.assertions_graph.evaluate(boxes_output, lut_io_info, self.current_als_configuration)[0]
        return [ o[f"\\{c}"] for c in self.model_classes ]

    def get_assertion_activation(self, attributes):
        boxes_output = self.get_boxes_output(attributes)
        prediction_as_one_hot = np.array([eval(a["boolean_net"], boxes_output) for a in self.boolean_networks ], dtype=int)
        for class_name, assertions in self.class_assertions.items():
            for minterm in assertions:
                if eval(minterm, boxes_output):
                    return class_name, minterm, prediction_as_one_hot
                
    def parse(self, root_node, use_espresso):
        logger = logging.getLogger("pyALS-RF")
        db_aliases = {}
        leaves_name = []
        self.leaves = []
        self.decision_boxes = []
        
        for node in PreOrderIter(root_node):
            if any(node.children):
                try:
                    feature = next(item for item in self.model_features if item["name"] == node.feature)
                    #! Do not instantiate DBs here!
                    # self.decision_boxes.append({
                    #     "name" : node.name,
                    #     "box"  : DecisionBox(node.name, node.feature, feature["type"], node.operator, node.threshold_value)})
                    
                    #! check that there are no db processing the same feature with the same threshold first
                    condition = (node.feature, node.threshold_value)
                    if condition not in db_aliases:
                        db_aliases[condition] = []
                    db_aliases[condition].append(node)
                    logger.debug(f"Tree {self.name}: found split node {node.name} evaluating feature \"{node.feature} {node.operator} {node.threshold_value}\"")
                except Exception:
                    print(f"\"{node.feature}\": Feature not found! Recognized model features: {self.model_features}")
                    exit()
            elif not any(node.children):
                self.leaves.append({"name": node.name, "class": node.score, "boolean_net": f"({str(node.boolean_expression)})"})
                leaves_name.append(node.name)
                logger.debug(f"Tree {self.name}: found leaf node {node.name} resulting in class \"{node.score}\" with condition when \"{str(node.boolean_expression)}\"")
        for l in self.leaves:
            for n in leaves_name:
                assert f"{n} " not in l["boolean_net"], f"Leaf name found in boolean expression! {n} found in {l}"
        for condition, aliases in db_aliases.items():
            try:
                #! db instantiation is here!
                logger.debug(f"Instantiating {aliases[0].name} DB in DT {self.name} evaluating feature \"{aliases[0].feature} {aliases[0].operator} {aliases[0].threshold_value}\"")
                feature = next(item for item in self.model_features if item["name"] == aliases[0].feature)
                self.decision_boxes.append({
                    "name" : aliases[0].name,
                    "box"  : DecisionBox(aliases[0].name, aliases[0].feature, feature["type"], aliases[0].operator, aliases[0].threshold_value)})
                for n in aliases[1:]:
                    logger.debug(f"\tMerging {n.name} to {aliases[0].name} in DT {self.name}.")
                    # #! every time a db is merged, any assertion function involving it has to be amended, replacing the name of the merged db with the retained one
                    for l in range(len(self.leaves)):
                        if f"{n.name} " in self.leaves[l]["boolean_net"] or f"{n.name})" in self.leaves[l]["boolean_net"] :
                            logger.debug(f"\t\tReplacing {n.name} with {aliases[0].name} at node {self.leaves[l]['name']}")
                            logger.debug(f"\t\tOld: {self.leaves[l]['boolean_net']}")
                            new_assertion = self.leaves[l]["boolean_net"].replace(f"{n.name} ", f"{aliases[0].name} ").replace(f"{n.name})", f"{aliases[0].name})")
                            logger.debug(f"\t\tNew: {new_assertion}")
                            assert f"{n.name} " not in new_assertion, f"Merge failed. {n.name} found in {new_assertion}"
                            assert f"{n.name})" not in new_assertion, f"Merge failed. {n.name} found in {new_assertion}"
                            assert f"{aliases[0].name}" in new_assertion, f"Merge failed. {aliases[0].name} not found in {new_assertion}"
                            self.leaves[l]['boolean_net'] = new_assertion
                            assert f"{n.name} " not in self.leaves[l]['boolean_net'], f"Merge failed. {n.name} found in {self.leaves[l]['boolean_net']}"
                            assert f"{n.name})" not in self.leaves[l]['boolean_net'], f"Merge failed. {n.name} found in {self.leaves[l]['boolean_net']}"
                            assert f"{aliases[0].name}" in self.leaves[l]['boolean_net'], f"Merge failed. {aliases[0].name} not found in {self.leaves[l]['boolean_net']}"
            except Exception as e:
                    print(e)
                    exit()            
        assert len(db_aliases) == len(self.decision_boxes), f"Error during DBs instantiation"
        logger.info(f"Tree {self.name}: {len(self.decision_boxes)} DBs instantiated.")
        for l in self.leaves:
            for n in leaves_name:
                assert f"{n} " not in l["boolean_net"], f"Leaf name found in assertion function! {n} found in {l}"
        self.boolean_networks = [ self.get_boolean_net(c, use_espresso) for c in self.model_classes ]
        for l in self.boolean_networks:
            for n in leaves_name:
                assert f"{n} " not in l["boolean_net"], f"Leaf name found in boolean expression for class {l['class']}! {n} found in {l['boolean_net']}"
        self.class_assertions = { c : [item["boolean_net"].replace("~", "not ").replace("|", "or").replace("&", "and") for item in self.leaves if item["class"] == c] for c in self.model_classes}
        
    def define_boolean_expression(self, minterms, use_espresso):
        logger = logging.getLogger("pyALS-RF")
        if not minterms:
            boolean_net = 'False'
            hdl_expression = '\'0\''
        elif len(minterms) == 1:
            boolean_net = minterms[0]
            hdl_expression = f"func_and{minterms[0].replace('~', 'not ').replace(' & ', ', ').replace(' and ', ', ')}"
        else:
            if use_espresso:
                logger.info("Using espresso heuristic logic minimizer")
                hdl_expression = str(espresso_exprs(expr(" | ".join(minterms)))[0]).replace("~", "not ").replace("Or","func_or").replace("And","func_and")
            else:
                and_gates = [f"func_and{m.replace('~', 'not ').replace(' & ', ', ').replace(' and ', ', ')}" for m in minterms]
                hdl_expression = f"func_or({', '.join(and_gates)})"
            boolean_net = " or ".join(minterms).replace("~", "not ").replace("&", "and")
        return boolean_net,hdl_expression
    
    def get_boolean_net(self, class_name : str, use_espresso : bool):
        minterms = [item["boolean_net"] for item in self.leaves if item["class"] == class_name]
        boolean_net, hdl_expression = self.define_boolean_expression(minterms, use_espresso)
        return {"class" : class_name, "minterms" : minterms, "boolean_net" : boolean_net, "hdl_expression" : hdl_expression}

    def set_pruning(self, pruning, use_espresso : bool):
        logger = logging.getLogger("pyALS-RF")
        self.pruned_boolean_nets = []
        logger.debug(f"Setting pruning configuration for {self.name}")
        for class_name, assertions in self.class_assertions.items():
            pruned = [assertion for class_label, tree_name, assertion, _ in pruning if tree_name == self.name and class_label == class_name ]            
            kept_assertions = [ assertion for assertion in assertions if assertion not in pruned ]
            boolean_net, hdl_expression = self.define_boolean_expression(kept_assertions, use_espresso)
            self.pruned_boolean_nets.append({"class" : class_name, "minterms" : [k.replace("and", "&") for k in kept_assertions], "boolean_net" : boolean_net, "hdl_expression" : hdl_expression})
            # logger.debug(f"\tClass: {class_name}")
            # logger.debug(f"\tMinterms:")
            # for m in kept_assertions:
            #     logger.debug(f"\t\t{m}")
            # logger.debug(f"Boolean Network: {boolean_net}")
            # logger.debug(f"HDL statement: {hdl_expression}")
            
    def get_assertions_cost(self):
        return sum(len(minterm.split(" & ")) for network in self.boolean_networks for minterm in network["minterms"])
    
    def get_pruned_assertions_cost(self):
        return sum(len(minterm.split(" & ")) for network in self.pruned_boolean_nets for minterm in network["minterms"])
        