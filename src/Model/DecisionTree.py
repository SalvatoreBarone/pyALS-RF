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
from tabulate import tabulate
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
        return sum(box["box"].nab for box in self.decision_boxes)

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
            print("\t\t",  b["box"].name, "(", b["box"].feature_name, " " , b["box"].get_c_operator(), " ", b["box"].threshold, "), nab ", b["box"].nab)
        print("\tAssertions:")
        for a in self.boolean_networks:
            print("\t\t", a["class"], " = ", a["sop"])

    def get_boxes_output(self, attributes):
        return {box["box"].name if self.als_conf is None else "\\" + box["box"].name() : box["box"].compare(attributes[self.attrbutes_name.index(box["box"].feature_name)]) for box in self.decision_boxes}
    
    def visit(self, attributes):
        boxes_output = self.get_boxes_output(attributes)
        if self.als_conf is None:
            return [ int(eval(a["sop"], boxes_output)) for a in self.boolean_networks ]
        exit()
        lut_io_info = {}
        output = self.assertions_graph.evaluate(boxes_output, lut_io_info, self.current_als_configuration)[0]
        return [ o[f"\\{c}"] for c in self.model_classes ]
                
    def parse(self, root_node, use_espresso):
        logger = logging.getLogger("pyALS-RF")
        db_aliases = {}
        leaves_name = []
        self.leaves = []
        self.decision_boxes = []
        nl = '\n'
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
                self.leaves.append({"name": node.name, "class": node.score, "sop": f"({str(node.boolean_expression)})"})
                leaves_name.append(node.name)
                logger.debug(f"Tree {self.name}: found leaf node {node.name} resulting in class \"{node.score}\" with condition when \"{str(node.boolean_expression)}\"")
        for l in self.leaves:
            for n in leaves_name:
                assert f"{n} " not in l["sop"], f"Leaf name found in boolean expression! {n} found in {l}"
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
                        if f"{n.name} " in self.leaves[l]["sop"] or f"{n.name})" in self.leaves[l]["sop"] :
                            logger.debug(f"\t\tReplacing {n.name} with {aliases[0].name} at node {self.leaves[l]['name']}")
                            logger.debug(f"\t\tOld: {self.leaves[l]['sop']}")
                            new_assertion = self.leaves[l]["sop"].replace(f"{n.name} ", f"{aliases[0].name} ").replace(f"{n.name})", f"{aliases[0].name})")
                            logger.debug(f"\t\tNew: {new_assertion}")
                            assert f"{n.name} " not in new_assertion, f"Merge failed. {n.name} found in {new_assertion}"
                            assert f"{n.name})" not in new_assertion, f"Merge failed. {n.name} found in {new_assertion}"
                            assert f"{aliases[0].name}" in new_assertion, f"Merge failed. {aliases[0].name} not found in {new_assertion}"
                            self.leaves[l]['sop'] = new_assertion
                            assert f"{n.name} " not in self.leaves[l]['sop'], f"Merge failed. {n.name} found in {self.leaves[l]['sop']}"
                            assert f"{n.name})" not in self.leaves[l]['sop'], f"Merge failed. {n.name} found in {self.leaves[l]['sop']}"
                            assert f"{aliases[0].name}" in self.leaves[l]['sop'], f"Merge failed. {aliases[0].name} not found in {self.leaves[l]['sop']}"
            except Exception as e:
                    print(e)
                    exit()            
        assert len(db_aliases) == len(self.decision_boxes), f"Error during DBs instantiation"
        logger.info(f"Tree {self.name}: {len(self.decision_boxes)} DBs instantiated.")
        for l in self.leaves:
            for n in leaves_name:
                assert f"{n} " not in l["sop"], f"Leaf name found in assertion function! {n} found in {l}"
        self.boolean_networks = [ self.get_boolean_net(c, use_espresso) for c in self.model_classes ]
        logger.debug(f'Tree {self.name} Boolean network:\n{tabulate([[bn["class"], f"{nl}".join(bn["minterms"]), bn["sop"].replace(" or ", f" or{nl}"), bn["hdl_expression"].replace(" or ", f" or{nl}")] for bn in self.boolean_networks], headers=["class", "minterms", "SoP", "HDL"], tablefmt="grid")}')
        for l in self.boolean_networks:
            for n in leaves_name:
                assert f"{n} " not in l["sop"], f"Leaf name found in boolean expression for class {l['class']}! {n} found in {l['sop']}"
        self.class_assertions = { c : [item["sop"] for item in self.leaves if item["class"] == c] for c in self.model_classes}
        logger.debug(f'Tree {self.name} class assertions:\n{tabulate([[k, f"{nl}".join(v), ] for k, v in self.class_assertions.items()], headers=["class", "minterms"], tablefmt="grid")}')
        
    def define_boolean_expression(self, minterms, use_espresso):
        logger = logging.getLogger("pyALS-RF")
        if not minterms:
            sop = 'False'
            hdl_expression = '\'0\''
        elif len(minterms) == 1:
            sop = hdl_expression = minterms[0]
            minterms = []
        elif use_espresso:
                print("Esresso")
                logger.info("Using espresso heuristic logic minimizer")
                minimized_expression = str(espresso_exprs(expr(" | ".join( m.replace("not ", " ~").replace("and", "&") for m in minterms)))[0]).replace("Or(","").replace("~", "not ").replace("))", ")")
                minterms = [m.replace("And(", "(").replace(",", " and") for m in minimized_expression.split(", And")]
                sop = hdl_expression = " or ".join(minterms)
        else:
            sop = hdl_expression = " or ".join(minterms)
        return minterms, sop, hdl_expression
    
    def get_boolean_net(self, class_name : str, use_espresso : bool):
        minterms = [item["sop"] for item in self.leaves if item["class"] == class_name]
        minterms, sop, hdl_expression = self.define_boolean_expression(minterms, use_espresso)
        return {"class" : class_name, "minterms" : minterms, "sop" : sop, "hdl_expression" : hdl_expression}
        