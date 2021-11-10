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

"""
@brief The class implements a single decision trees.

@details
Although the adopted implementation does not provide any advantage when considering software implementations, this 
class implements a decision tree using the speculative approach proposed in Amato F, Barbareschi M, Casola V, Mazzeo A,
Romano S "Towards automatic generation of hardware classifiers". International Conference on Algorithms and
Architectures for Parallel Processing. Springer, Cham, pp 125–132
In this way, it is possible to mimic more effectively the effects that precision-scaling and functional approximation
techniques have on the model used for the final hardware implementation.

The speculative approach consists in a DT flattening so that the visiting is performed over every possible path.
In particular, each DT node contains a condition that establishes if the visiting has to continue on left sub-tree or on
right sub-tree, until a leaf is reached. Instead, in the speculative approach, predicates are performed concurrently,
regardless of the position and depth at which nodes are located: a Boolean decision variable, which indicates whether a
condition is fulfilled, is produced for each one of the evaluated predicates. In order to determine which leaf of the
DT is reached, i.e., which class the input belongs to, a Boolean function, called assertion, is defined for each
different class. Since a path that leads to a specific leaf is obtained by computing the logic-AND between the Boolean
decision variables along that path, and since it is possible to compute the logic OR between the conditions related to
different paths leading to leaves belonging to the same class, assertions can be defined as a sum of products Boolean
functions. 

As mentioned, this class also allows to apply the precision scaling and functional AIG-rewriting approximate-computing
techniques on feature values representation and assertion functions, respectively, in order to reduce hardware
requirements of the final hardware implementation.
"""
class DecisionTree:
  __source_dir = "./resources/vhd/"
  __bnf_vhd = "bnf.vhd"
  __vhdl_assertions_source = "assertions_block.vhd.template"
  __vhdl_decision_tree_source = "decision_tree.vhd.template"

  """
  @brief Constructor. Builds a new DecisionTree object.

  @param [in] name
              name of the tree; this should be unique amond trees belonging to the same classifier.

  @param [in] root_node
              root of the tree.

  @param [in] features
              list of the features employed by the model

  @param [in] classes
              list of the classes employed by the model

  @param [in] lut_tech
              
  @param [in] catalog_cache

  @param [in] smt_timeout

  """
  def __init__(self, name, root_node, features, classes, lut_tech, catalog_cache, smt_timeout):
    self.__name = name
    self.__root_node = root_node
    self.__decision_boxes = []
    self.__assertions = []
    self.__model_features = features
    self.__model_classes = classes
    self.__get_decision_boxes()
    self.__get_assertions()
    design = self.__generate_design_for_als(lut_tech)
    self.__assertions_graph = ALSGraph(design)
    self.__assertions_catalog_entries = ALSCatalog(catalog_cache, smt_timeout).generate_catalog(design)
    ys.run_pass("design -save {name}".format(name = self.__name), design)
    self.__current_configuration = []
    self.set_assertions_configuration([0] * len(self.__assertions_graph.get_cells()))
    
  def get_name(self):
    return self.__name
  
  def get_decision_boxes(self):
    return self.__decision_boxes

  def get_assertions(self):
    return self.__assertions

  def get_graph(self):
    return self.__assertions_graph

  def get_catalog_for_assertions(self):
    return self.__assertions_catalog_entries

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

  def get_total_bits(self):
    return 64 * len(self.__decision_boxes)

  def get_total_nabs(self):
    return sum([ box["box"].get_nab() for box in self.__decision_boxes ])

  def get_total_retained(self):
    return 64 * len(self.__decision_boxes) - self.get_total_nabs()

  """
  @brief Set the current approximate configurations for the assertion block

  @param [in] configuration
              Approximate configuration. Aach item in the list corresponds to a LUT of the circuit, and specifies which
              approximate variant to use for said LUT, i.e. what the Hamming distance between the exact specification of
              the LUT and the approximate implementation to use should be.
  """
  def set_assertions_configuration(self, configuration):
    self.__current_configuration = [ {"name" : l["name"], "spec" : e[c]["spec"], "gates" : e[c]["gates"] }  for c, l in zip(configuration, self.__assertions_graph.get_cells()) for e in self.__assertions_catalog_entries if e[0]["spec"] == l["spec"] ]

  def get_current_required_aig_nodes(self):
    return sum([ c["gates"] for c in self.__current_configuration ])

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
    boxes_output = []
    for box in self.__decision_boxes:
      value = next(item for item in features_value if item["name"] == box["box"].get_feature())["value"]
      boxes_output.append({"name" : box["box"].get_name(), "value" : box["box"].compare(value)})
    output = self.__assertions_graph.evaluate(boxes_output)
    for c in classes_score:
      c["score"] += 1 if next((sub for sub in output if sub['name'] == c["name"]), None)["value"]  else 0
  
  """
  @brief Generates the VHDL implementation of the decision tree.
  
  @details
  Generates the VHDL implementation of the decision tree, including each of the decision-boxes and the assertion
  function computation block.

  @param [in] destination
              path of the destination directory in which the source code will be generated
  """
  def generate_tree_vhd(self, destination):
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

  """
  @brief Generates the VHDL implementation of the assertion block for a given decision tree.
  
  @param [in] destination
              path of the destination directory in which the source code will be generated
  """
  def generate_assertions_vhd(self, destination):
    module_name = "assertions_block_" + self.__name 
    file_name = destination + "/vhd/assertions_block_" + self.__name + ".vhd"
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

  """
  @brief Generates the Verilog implementation of the approximate assertion block for a given decision tree.
  
  @param [in] destination
              path of the destination directory in which the source code will be generated
  """
  def generate_ax_assertions_v(self, destination):
    design = ys.Design()
    ys.run_pass("design -load {}".format(self.__name), design)
    for module in design.selected_whole_modules_warn():
      for cell in module.selected_cells():
        if ys.IdString("\LUT") in cell.parameters:
          # get the specification for the current cell
          c = [ c for c in self.__current_configuration if c["name"] == cell.name.str() ][0]
          print("cell: {name}; spec: {spec}; new spec {newspec}".format(name = c["name"], spec = cell.parameters[ys.IdString("\LUT")], newspec = c["spec"]))
          cell.setParam(ys.IdString("\LUT"), ys.Const.from_string(c["spec"]))
    ys.run_pass("write_verilog {dir}/assertions_block_{name}.v".format(dir = destination, name = self.__name), design)    

  def __get_decision_boxes(self):
    self.__decision_boxes = []
    for node in PreOrderIter(self.__root_node):
      if any(node.children):
        feature = next(item for item in self.__model_features if item["name"] ==node.feature)
        self.__decision_boxes.append({
          "name" : node.name, 
          "box"  : DecisionBox(node.name, node.feature, feature["type"], node.operator, node.threshold_value)})

  def __get_leaves(self):
    leaves = []
    for node in PreOrderIter(self.__root_node):
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

  def __get_assertions(self):
    self.__assertions = []
    leaves = self.__get_leaves()
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
    file_name, module_name = self.generate_assertions_vhd(destination)
    design = ys.Design()
    ys.run_pass("design -reset", design)
    ys.run_pass("ghdl {} {} -e {}".format(self.__source_dir + self.__bnf_vhd, file_name, module_name), design)
    ys.run_pass("hierarchy -check -top {}".format(module_name), design)
    ys.run_pass("prep",  design)
    ys.run_pass("splitnets -ports",  design)
    ys.run_pass("synth -lut " + str(luts_tech), design)
    return design
