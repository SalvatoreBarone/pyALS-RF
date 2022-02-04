"""
Copyright 2021 Salvatore Barone <salvatore.barone@unina.it>

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
from pyosys import libyosys as ys
from jinja2 import Environment, FileSystemLoader
from distutils.dir_util import mkpath
from anytree import PreOrderIter
from pyeda.inter import *
from .DecisionBox import *
from .ALSGraph import *
from .ALSCatalog import *
from .ALSRewriter import *

class DecisionTree:
  __source_dir = "./resources/vhd/"
  __bnf_vhd = "bnf.vhd"
  __vhdl_assertions_source = "assertions_block.vhd.template"
  __vhdl_decision_tree_source = "decision_tree.vhd.template"

  def __init__(self, name = None, root_node = None, features = None, classes = None, als_conf = None):
    self.__name = name
    self.__model_features = features
    self.__model_classes = classes
    self.__decision_boxes = []
    self.__assertions = []
    self.__als_conf = als_conf
    self.__first_stage_approximate_implementations = None
    if root_node:
      self.__get_decision_boxes(root_node)
      self.__get_assertions(root_node)
    self.__current_configuration = []
    if als_conf:
      design = self.__generate_design_for_als(self.__als_conf.cut_size)
      self.__assertions_graph = ALSGraph(design)
      self.__assertions_catalog_entries = ALSCatalog(self.__als_conf.catalog, self.__als_conf.solver).generate_catalog(design, self.__als_conf.timeout)
      self.set_assertions_configuration([0] * self.__assertions_graph.get_num_cells())
      ys.run_pass("design -save {name}".format(name = self.__name), design)
    else:
      self.__assertions_graph = None
      self.__catalog = None
      self.__assertions_catalog_entries = None

  def __deepcopy__(self, memo = None):
    tree = DecisionTree()
    tree.__name = copy.deepcopy(self.__name)
    tree.__model_features = copy.deepcopy(self.__model_features)
    tree.__model_classes = copy.deepcopy(self.__model_classes)
    tree.__decision_boxes = copy.deepcopy(self.__decision_boxes)
    tree.__assertions = copy.deepcopy(self.__assertions)
    tree.__assertions_graph = copy.deepcopy(self.__assertions_graph)
    tree.__als_conf = copy.deepcopy(self.__als_conf)
    tree.__assertions_catalog_entries = copy.deepcopy(self.__assertions_catalog_entries)
    tree.__current_configuration = copy.deepcopy(self.__current_configuration)
    tree.__first_stage_approximate_implementations = copy.deepcopy(self.__first_stage_approximate_implementations)
    return tree
    
  def get_name(self):
    return self.__name

  def get_model_classes(self):
    return self.__model_classes

  def get_model_features(self):
    return self.__model_features
  
  def get_decision_boxes(self):
    return self.__decision_boxes

  def get_assertions(self):
    return self.__assertions

  def get_graph(self):
    return self.__assertions_graph

  def get_catalog_for_assertions(self):
    return self.__assertions_catalog_entries

  def get_total_bits(self):
    return 64 * len(self.__decision_boxes)

  def get_total_nabs(self):
    return sum([ box["box"].get_nab() for box in self.__decision_boxes ])

  def get_total_retained(self):
    return 64 * len(self.__decision_boxes) - self.get_total_nabs()

  def get_assertions_configuration(self):
    return self.__current_configuration

  def get_first_stage_approximate_implementations(self):
    return self.__first_stage_approximate_implementations

  def get_assertions_distance(self):
    return [ self.__current_configuration[c]["dist"] for c in self.__current_configuration.keys() ]

  def get_current_required_aig_nodes(self):
    return sum([ self.__current_configuration[c]["gates"] for c in self.__current_configuration.keys() ])

  def reset_assertion_configuration(self):
    self.set_assertions_configuration([0] * self.__assertions_graph.get_num_cells())

  def get_als_num_of_dv(self):
    return len(self.__assertions_graph.get_cells())

  def get_als_dv_upper_bound(self):
    return [len(e) - 1 for c in [{"name": c["name"], "spec": c["spec"]} for c in self.__assertions_graph.get_cells()] for e in self.__assertions_catalog_entries if e[0]["spec"] == c["spec"]]

  """
  @brief Allows to set the number of approximate bits for each of the decision boxes belonging to the tree.

  @param [in] nabs
              A list of dict, each of which has the following key-value pairs:
               - "name": name: name of the feature
               - "nab": nab (int) number of approximate bits for the given feature representation
              Ex.: [{"name": "feature_x", "nab" : 4}, {"name": "feature_y", "nab" : 3}, {"name": "feature_z", "nab" : 5}]

  @details
  This function allows you to set the degree of approximation to be adopted for the representation of the values of the
  features for each of the decision-boxes of which the decision tree is composed. It is used during the design-space
  exploration phase to evaluate the error introduced by the approximation and to estimate the corresponding gain in
  terms of area on the silicon. 
  """
  def set_nabs(self, nabs):
    for box in self.__decision_boxes:
      box["box"].set_nab(next(item for item in nabs if item["name"] == box["box"].get_feature())["nab"])

  # def store_first_stage_approximate_implementations(self, configurations):
  #   self.__first_stage_approximate_implementations = configurations

  """
  @brief Set the current approximate configurations for the assertion block

  @param [in] configuration
              Approximate configuration. Aach item in the list corresponds to a LUT of the circuit, and specifies which
              approximate variant to use for said LUT, i.e. what the Hamming distance between the exact specification of
              the LUT and the approximate implementation to use should be.
  """
  def set_assertions_configuration(self, configuration):
    self.__current_configuration = {l["name"]: {"dist": c, "spec": e[0]["spec"], "axspec": e[c]["spec"], "gates": e[c]["gates"], "S": e[c]["S"], "P": e[c]["P"], "out_p": e[c]["out_p"], "out": e[c]["out"], "depth": e[c]["depth"]} for c, l in zip(configuration, self.__assertions_graph.get_cells()) for e in self.__assertions_catalog_entries if e[0]["spec"] == l["spec"]}

  # def set_first_stage_approximate_implementations(self, configuration):
  #   self.set_assertions_configuration(self.__first_stage_approximate_implementations[configuration])

  def dump(self):
    print("\tName: ", self.__name)
    print("\tBoxes:")
    for b in self.__decision_boxes:
      print("\t\t",  b["box"].get_name(), "(", b["box"].get_feature(), " " , b["box"].get_c_operator(), " ", b["box"].get_threshold(), "), nab ", b["box"].get_nab())
    print("\tAssertions:")
    for a in self.__assertions:
      print("\t\t", a["class"], " = ", a["expression"])

  """
  @brief Performs a classification

  @param [in]     features_value
                  A list of dict, each of which has the following key-value pairs:
                    - "name": name: name of the feature
                    - "value": value: (int/double) value of the feature
                  Ex.: [{"name": "feature_x", "value" : 4}, {"name": "feature_y", "value" : 0.3}, {"name": "feature_z", "value" : 1.5}]

  @param [in]     assertions
                  governs which of the assertion variants has to be used to compute the class score

  @param [in,out] classes_score
                  A list of dict, each of which has the following key-value pairs:
                    - "name": name: name of the class
                    - "score": value: (int) current score of the class
                  Ex.: [{"name": "class_x", "score" : 4}, {"name": "class_y", "score" : 0}, {"name": "class_z", "score" : 1}]
                  The "score" value of the winning class will be incremented by one at the end of the classification
                  procedure
  """
  def evaluate(self, features_value, classes_score):
    boxes_output = dict()
    for box in self.__decision_boxes:
      boxes_output["\\" + box["box"].get_name()] = box["box"].compare(features_value[box["box"].get_feature()])
    output = self.__assertions_graph.evaluate(boxes_output, self.__current_configuration)
    for c in classes_score.keys():
      classes_score[c] += int(output["\\" + c])

  def generate_hdl_tree(self, destination):
    file_name = destination + "/decision_tree_" + self.__name + ".vhd"
    file_loader = FileSystemLoader(self.__source_dir)
    env = Environment(loader=file_loader)
    template = env.get_template(self.__vhdl_decision_tree_source)
    output = template.render(
      tree_name = self.__name, 
      features  = self.__model_features,
      classes = self.__model_classes,
      boxes = [ b["box"].get_struct() for b in self.__decision_boxes ])
    out_file = open(file_name, "w")
    out_file.write(output)
    out_file.close()
    return file_name

  def generate_hdl_exact_assertions(self, destination):
    module_name = "assertions_block_" + self.__name 
    file_name = destination + "/assertions_block_" + self.__name + ".vhd"
    file_loader = FileSystemLoader(self.__source_dir)
    env = Environment(loader=file_loader)
    template = env.get_template(self.__vhdl_assertions_source)
    output = template.render(
      tree_name = self.__name, 
      boxes = [b["name"] for b in self.__decision_boxes],
      classes = self.__model_classes,
      assertions = self.__assertions)
    out_file = open(file_name, "w")
    out_file.write(output)
    out_file.close()
    return file_name, module_name

  def generate_hdl_als_ax_assertions(self, destination):
    rewriter = ALSRewriter(self.__assertions_graph, self.__assertions_catalog_entries)
    rewriter.rewrite_and_save(self.__name, self.__current_configuration, destination + "assertions_block_" + {self.__name})

  def __get_decision_boxes(self, root_node):
    self.__decision_boxes = []
    for node in PreOrderIter(root_node):
      if any(node.children):
        try:
          feature = next(item for item in self.__model_features if item["name"] == node.feature)
          self.__decision_boxes.append({
            "name" : node.name,
            "box"  : DecisionBox(node.name, node.feature, feature["type"], node.operator, node.threshold_value)})
        except:
          print(node.feature, "Feature not found")
          print("Recognized model features", self.__model_features)
          exit()

  def __get_leaves(self, root_node):
    leaves = []
    for node in PreOrderIter(root_node):
      if not any(node.children):
        leaves.append({"name" : node.name, "class" : node.score, "expression" : "(" + str(node.boolean_expression) + ")" })
    return leaves

  def __get_assertion(self, leaf_set, class_name):
    conditions = [item["expression"] for item in leaf_set if item["class"] == class_name] 
    if len(conditions) == 0:
      return "False"
    elif len(conditions) == 1:
      return conditions[0]
    else:
      return " | ".join(conditions)

  def __get_assertions(self, root_node):
    self.__assertions = []
    leaves = self.__get_leaves(root_node)
    for class_name in self.__model_classes:
        assertion_function = self.__get_assertion(leaves, class_name)
        minimized_assertion = str(espresso_exprs(expr(assertion_function))[0]).replace("~", "not ").replace("Or","func_or").replace("And","func_and")
        self.__assertions.append({
          "class"      : class_name,
          "expression" : assertion_function,
          "minimized"  : "'0'" if assertion_function == "False" else minimized_assertion})

  def __generate_design_for_als(self, luts_tech):
    destination = "/tmp/EDGINESS/"
    mkpath(destination)
    mkpath(destination + "/vhd")
    file_name, module_name = self.generate_hdl_exact_assertions(destination)
    design = ys.Design()
    ys.run_pass("tee -q design -reset", design)
    ys.run_pass(f"tee -q ghdl {self.__source_dir + self.__bnf_vhd} {file_name} -e {module_name}", design)
    ys.run_pass(f"tee -q hierarchy -check -top {module_name}; tee -q prep; tee -q flatten; tee -q splitnets -ports; tee -q synth -top {module_name}; tee -q flatten; tee -q clean -purge; tee -q synth -lut {luts_tech}", design)
    return design
