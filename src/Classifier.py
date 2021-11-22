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
import sys, csv, copy
from xml.etree import ElementTree
from anytree import Node
from multiprocessing import Pool
from jinja2 import Environment, FileSystemLoader
from distutils.dir_util import copy_tree, mkpath
from distutils.file_util import copy_file
from .DecisionTree import DecisionTree
from .Utility import *

"""
@brief Decision-tree based Multiple Classifier System, implemented in python.

@details
This class implements a Decision-tree based Multiple Classifier System as proposed in Amato F, Barbareschi M, Casola V,
Mazzeo A, Romano S "Towards automatic generation of hardware classifiers". International Conference on Algorithms and
Architectures for Parallel Processing. Springer, Cham, pp 125–132.

This class supports both single-tree and random-forest DT-MCS. The latter are supported as long as they require
majority-voting logic.

From the software point of view, the outcome calculated by each of the trees is summed and the class that reaches a
score that is greater than 50% is declared the winner.
At hardware level, outcomes of the assertion functions belonging to the same class but computed by different DTs are
arranged in an array of N elements, with N being the number of DTs. A majority voter is used to state which class is the
winner.
"""
class Classifier:
  __namespaces = {'pmml': 'http://www.dmg.org/PMML-4_1'}
  __source_dir = "./resources/"
  # VHDL sources
  __vhdl_bnf_source = "vhd/bnf.vhd"
  __vhdl_reg_source = "vhd/pipe_reg.vhd"
  __vhdl_decision_box_source = "vhd/decision_box.vhd"
  __vhdl_voter_source = "vhd/voter.vhd"
  __vhdl_debugfunc_source = "vhd/debug_func.vhd"
  __vhdl_classifier_template_file = "vhd/classifier.vhd.template"
  __vhdl_tb_classifier_template_file = "vhd/tb_classifier.vhd.template"
  # sh files
  __run_synth_file = "sh/run_synth.sh"
  __run_sim_file = "sh/run_sim.sh"
  __run_all_file = "sh/run_all.sh"
  __extract_luts_file = "sh/extract_utilization.sh"
  __extract_pwr_file = "sh/extract_power.sh"
  #tcl files
  __tcl_project_file = "tcl/create_project.tcl.template"
  __tcl_sim_file = "tcl/run_sim.tcl"
  #constraints
  __constraint_file = "constraints.xdc"

  """
  @brief Constructor function

  @param  [in]  pmml_file_name (optional)
                Path of the PMML file to be parsed

  @param  [in]  nabs
                Allows to set the number of approximate bits for each of the decision boxes belonging to the tree. It
                must be a list of dict, each of which has the following key-value pairs:
                  - "name": name: name of the feature
                  - "nab": nab (int) number of approximate bits for the given feature representation
                Ex.: [{"name": "feature_x", "nab" : 4}, {"name": "feature_y", "nab" : 3}, {"name": "feature_z", "nab" : 5}]
  """
  def __init__(self, als_lut_tech, als_catalog_cache, als_smt_timeout):
    self.__trees_list_obj = []
    self.__model_features_list_dict = []
    self.__model_classes_list_str = []
    self.__als_lut_tech = als_lut_tech
    self.__als_catalog_cache = als_catalog_cache
    self.__als_smt_timeout = als_smt_timeout

  def __deepcopy__(self, memo = None):
    classifier = Classifier(self.__als_lut_tech, self.__als_catalog_cache, self.__als_smt_timeout)
    classifier.__trees_list_obj = copy.deepcopy(self.__trees_list_obj)
    classifier.__model_features_list_dict = copy.deepcopy(self.__model_features_list_dict)
    classifier.__model_classes_list_str = copy.deepcopy(self.__model_classes_list_str)
    return classifier
 
  def parse(self, pmml_file_name):
    self.__trees_list_obj = []
    self.__model_features_list_dict = []
    self.__model_classes_list_str = []
    tree = ElementTree.parse(pmml_file_name)
    root = tree.getroot()
    self.__namespaces["pmml"] = get_xmlns_uri(root)
    self.__get_classes(root)
    self.__get_features(root)
    segmentation = root.find("pmml:MiningModel/pmml:Segmentation", self.__namespaces)
    if segmentation is not None:
      tree_id = 0
      for segment in segmentation.findall("pmml:Segment", self.__namespaces):
        tree_model_root = segment.find("pmml:TreeModel", self.__namespaces).find("pmml:Node", self.__namespaces)
        tree = self.__get_tree_model(str(tree_id), tree_model_root)
        self.__trees_list_obj.append(tree)
        tree_id += 1
    else:
      tree_model_root = root.find("pmml:TreeModel", self.__namespaces).find("pmml:Node", self.__namespaces)
      tree = self.__get_tree_model("0", tree_model_root)
      self.__trees_list_obj.append(tree)

  def dump(self):
    print("Features:")
    for f in self.__model_features_list_dict:
      print("\tName: ", f["name"], ", Type: ", f["type"], ", Min: ", f["min"], ", Max: ", f["max"])
    print("\n\nClasses:")
    for c in self.__model_classes_list_str:
      print("\tName: ", c)
    print("\n\nTrees:")
    for t in self.__trees_list_obj:
      t.dump()

  def get_classes(self):
    return self.__model_classes_list_str

  def get_features(self):
    return self.__model_features_list_dict

  """
  @brief Allows to set the number of approximate bits for each of the decision boxes belonging to each of the trees.

  @param [in] nabs
              A list of dict, each of which has the following key-value pairs:
               - "name": name: name of the feature
               - "nab": nab (int) number of approximate bits for the given feature representation
              Ex.: [{"name": "feature_x", "nab" : 4}, {"name": "feature_y", "nab" : 3}, {"name": "feature_z", "nab" : 5}]

  @details
  This function allows you to set the degree of approximation to be adopted for the representation of the values of the
  features for each of the decision-boxes of which each of the decision trees is composed. It is used during the
  design-space exploration phase to evaluate the error introduced by the approximation and to estimate the corresponding
  gain in terms of area on the silicon. 
  """
  def set_nabs(self, nabs):
    for tree in self.__trees_list_obj:
      tree.set_nabs(nabs)

  def reset_nabs_configuration(self):
    nabs = [{"name" : f["name"], "nab" : 0} for f in self.__model_features_list_dict ]
    self.set_nabs(nabs)

  def get_total_bits(self):
    return sum([ t.get_total_bits() for t in self.__trees_list_obj ])

  def get_total_retained(self):
    return sum([ t.get_total_retained() for t in self.__trees_list_obj ])

  def get_als_genes_per_tree(self):
    return [ len(t.get_graph().get_cells()) for t in self.__trees_list_obj]

  def get_als_genes_upper_bound(self):
    all_entries_available = []
    for t in self.__trees_list_obj:
       cells = [ { "name" : c["name"], "spec" : c["spec"] } for c in t.get_graph().get_cells() ]
       catalog = t.get_catalog_for_assertions()
       all_entries_available.append( [ len (e) - 1 for c in cells for e in catalog if e[0]["spec"] == c["spec"] ] )
    return [ e for te in all_entries_available for e in te ]

  def reset_assertion_configuration(self):
    for t in self.__trees_list_obj:
      t.reset_assertion_configuration()

  def set_assertions_configuration(self, configurations):
    for t, c in zip(self.__trees_list_obj, configurations):
      t.set_assertions_configuration(c)

  def get_current_required_aig_nodes(self):
    return sum([ t.get_current_required_aig_nodes() for t in self.__trees_list_obj ])

  def get_struct(self):
    struct = []
    for tree in self.__trees_list_obj:
      struct.append(tree.get_struct())
    return struct
      
  def preload_dataset(self, csv_file):
    samples = []
    with open(csv_file, 'r') as data:
      for line in csv.DictReader(data, delimiter=';'):
        input_features = []
        expected_result = []
        for f in self.__model_features_list_dict:
          input_features.append({"name" : f["name"], "value" : float(line[f["name"]])})
        for c in self.__model_classes_list_str:
          expected_result.append({"name" : c, "score" : int(line[c]) })
        samples.append({"input" : input_features, "outcome" : expected_result})
    return samples

  def evaluate_preloaded_dataset(self, samples):
    correct_outcomes = 0
    for sample in samples:
      correct_outcomes +=1 if sample["outcome"] == self.__evaluate(sample["input"]) else 0
    return correct_outcomes

  def evaluate_test_dataset(self, csv_file):
    samples = self.preload_dataset(csv_file)
    return self.evaluate_preloaded_dataset(samples) / len(samples)

  def generate_implementations(self, destination):
    features = [ {"name": f["name"], "nab": 0} for f in self.__model_features_list_dict ]
    mkpath(destination)
    ax_dest = destination + "/exact/"
    mkpath(ax_dest)
    copy_file(self.__source_dir + self.__extract_luts_file, ax_dest)
    copy_file(self.__source_dir + self.__extract_pwr_file, ax_dest)
    trees_name = [ t.get_name() for t in self.__trees_list_obj ]
    file_loader = FileSystemLoader(self.__source_dir)
    env = Environment(loader=file_loader)
    template = env.get_template(self.__vhdl_classifier_template_file)
    tb_classifier_template = env.get_template(self.__vhdl_tb_classifier_template_file)
    tcl_template = env.get_template(self.__tcl_project_file)
    tcl_file = tcl_template.render(
      assertions_blocks = [ {"file_name": "assertions_block_" + n + ".vhd", "language": "VHDL"} for n in trees_name ],
      decision_trees = [ {"file_name": "decision_tree_" + n + ".vhd", "language": "VHDL"} for n in trees_name ])
    out_file = open(ax_dest + "/create_project.tcl", "w")
    out_file.write(tcl_file)
    out_file.close()
    classifier = template.render(
      trees = trees_name,
      features  = features,
      classes = self.__model_classes_list_str)
    out_file = open(ax_dest + "/classifier.vhd", "w")
    out_file.write(classifier)
    out_file.close()
    tb_classifier = tb_classifier_template.render(
      features = features,
      classes = self.__model_classes_list_str)
    out_file = open(ax_dest + "/tb_classifier.vhd", "w")
    out_file.write(tb_classifier)
    out_file.close()
    copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
    copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
    copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
    copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
    copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
    copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
    copy_file(self.__source_dir + self.__constraint_file, ax_dest)
    copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
    copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
    for tree in self.__trees_list_obj:
      tree.generate_tree_vhd(ax_dest)
      tree.generate_assertions_vhd(ax_dest)

  def generate_ps_ax_implementations(self, destination, configurations):
    mkpath(destination)
    mkpath(destination + "/ax")
    copy_file(self.__source_dir + self.__run_all_file, destination)
    copy_file(self.__source_dir + self.__extract_luts_file, destination)
    copy_file(self.__source_dir + self.__extract_pwr_file, destination)
    trees_name = [ t.get_name() for t in self.__trees_list_obj ]
    file_loader = FileSystemLoader(self.__source_dir)
    env = Environment(loader=file_loader)
    tcl_template = env.get_template(self.__tcl_project_file)
    classifier_template = env.get_template(self.__vhdl_classifier_template_file)
    tb_classifier_template = env.get_template(self.__vhdl_tb_classifier_template_file)
    tcl_file = tcl_template.render(
      assertions_blocks=[{"file_name": "assertions_block_" + n + ".vhd", "language": "VHDL"} for n in trees_name],
      decision_trees=[{"file_name": "decision_tree_" + n + ".vhd", "language": "VHDL"} for n in trees_name])
    for conf, i in zip(configurations, range(len(configurations))):
      features = [{"name": f["name"], "nab": n} for f, n in zip(self.__model_features_list_dict, conf)]
      ax_dest = destination + "/ax/configuration_" + str(i)
      mkpath(ax_dest)
      out_file = open(ax_dest + "/create_project.tcl", "w")
      out_file.write(tcl_file)
      out_file.close()
      classifier = classifier_template.render(
        trees = trees_name, 
        features = features,
        classes = self.__model_classes_list_str)
      out_file = open(ax_dest + "/classifier.vhd", "w")
      out_file.write(classifier)
      out_file.close()
      tb_classifier = tb_classifier_template.render(
        features = features,
        classes = self.__model_classes_list_str)
      out_file = open(ax_dest + "/tb_classifier.vhd", "w")
      out_file.write(tb_classifier)
      out_file.close()
      copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
      copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
      copy_file(self.__source_dir + self.__constraint_file, ax_dest)
      copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
      copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
      for tree in self.__trees_list_obj:
        tree.generate_tree_vhd(ax_dest)
        tree.generate_assertions_vhd(ax_dest)

  def generate_asl_ax_implementations(self, destination, configurations):
    features = [ {"name": f["name"], "nab": 0} for f in self.__model_features_list_dict]
    mkpath(destination)
    mkpath(destination + "/ax")
    copy_file(self.__source_dir + self.__run_all_file, destination)
    copy_file(self.__source_dir + self.__extract_luts_file, destination)
    copy_file(self.__source_dir + self.__extract_pwr_file, destination)
    trees_name = [ t.get_name() for t in self.__trees_list_obj ]
    file_loader = FileSystemLoader(self.__source_dir)
    env = Environment(loader=file_loader)
    tcl_template = env.get_template(self.__tcl_project_file)
    classifier_template = env.get_template(self.__vhdl_classifier_template_file)
    tb_classifier_template = env.get_template(self.__vhdl_tb_classifier_template_file)
    classifier = classifier_template.render(
      trees = trees_name, 
      features  = features,
      classes = self.__model_classes_list_str)
    tb_classifier = tb_classifier_template.render(
      features=features,
      classes=self.__model_classes_list_str)
    tcl_file = tcl_template.render(
      assertions_blocks=[{"file_name": "assertions_block_" + n + ".v", "language": "Verilog"} for n in trees_name],
      decision_trees=[{"file_name": "decision_tree_" + n + ".vhd", "language": "VHDL"} for n in trees_name])
    for conf, i in zip(configurations, range(len(configurations))):
      ax_dest = destination + "/ax/configuration_" + str(i)
      mkpath(ax_dest)
      out_file = open(ax_dest + "/classifier.vhd", "w")
      out_file.write(classifier)
      out_file.close()
      out_file = open(ax_dest + "/tb_classifier.vhd", "w")
      out_file.write(tb_classifier)
      out_file.close()
      out_file = open(ax_dest + "/create_project.tcl", "w")
      out_file.write(tcl_file)
      out_file.close()
      copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
      copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
      copy_file(self.__source_dir + self.__constraint_file, ax_dest)
      copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
      copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
      chunks = []
      count = 0
      for size in [ len(t.get_graph().get_cells()) for t in self.__trees_list_obj ]:
        chunks.append([conf[i+count] for i in range(size)])
        count += size
      for t, c in zip(self.__trees_list_obj, chunks):
        t.generate_tree_vhd(ax_dest)
        t.set_assertions_configuration(c)
        t.generate_ax_assertions_v(ax_dest)

  def generate_full_ax_implementations(self, destination, configurations):
    mkpath(destination)
    mkpath(destination + "/ax")
    copy_file(self.__source_dir + self.__run_all_file, destination)
    copy_file(self.__source_dir + self.__extract_luts_file, destination)
    copy_file(self.__source_dir + self.__extract_pwr_file, destination)
    trees_name = [ t.get_name() for t in self.__trees_list_obj ]
    file_loader = FileSystemLoader(self.__source_dir)
    env = Environment(loader=file_loader)
    classifier_template = env.get_template(self.__vhdl_classifier_template_file)
    tb_classifier_template = env.get_template(self.__vhdl_tb_classifier_template_file)
    tcl_template = env.get_template(self.__tcl_project_file)
    tcl_file = tcl_template.render(
      assertions_blocks=[{"file_name": "assertions_block_" + n + ".v", "language": "Verilog"} for n in trees_name],
      decision_trees=[{"file_name": "decision_tree_" + n + ".vhd", "language": "VHDL"} for n in trees_name])
    for conf, i in zip(configurations, range(len(configurations))):
      features = [ {"name": f["name"], "nab": n} for f, n in zip(self.__model_features_list_dict, conf[:len(self.__model_features_list_dict)]) ]
      ax_dest = destination + "/ax/configuration_" + str(i)
      mkpath(ax_dest)
      classifier = classifier_template.render(
        trees = trees_name, 
        features  = features,
        classes = self.__model_classes_list_str)
      out_file = open(ax_dest + "/classifier.vhd", "w")
      out_file.write(classifier)
      out_file.close()

      tb_classifier = tb_classifier_template.render(
        trees=trees_name,
        features=features,
        classes=self.__model_classes_list_str)
      out_file = open(ax_dest + "/tb_classifier.vhd", "w")
      out_file.write(tb_classifier)
      out_file.close()
      out_file = open(ax_dest + "/create_project.tcl", "w")
      out_file.write(tcl_file)
      out_file.close()
      copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
      copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
      copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
      copy_file(self.__source_dir + self.__constraint_file, ax_dest)
      copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
      copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
      chunks = []
      count = 0
      for size in [ len(t.get_graph().get_cells()) for t in self.__trees_list_obj ]:
        chunks.append([conf[i+count+len(self.__model_features_list_dict)] for i in range(size)])
        count += size
      for t, c in zip(self.__trees_list_obj, chunks):
        t.generate_tree_vhd(ax_dest)
        t.set_assertions_configuration(c)
        t.generate_ax_assertions_v(ax_dest)

  def __evaluate(self, features):
    classes_score = []
    for c in self.__model_classes_list_str:
      classes_score.append({"name" : c, "score" : 0})
    for tree in self.__trees_list_obj:
      tree.evaluate(features, classes_score)
    for c in classes_score:
      c["score"] = 0 if c["score"] < (len(self.__trees_list_obj) / 2) else 1
    return classes_score

  def __get_features(self, root):
    for child in root.find("pmml:DataDictionary", self.__namespaces).findall('pmml:DataField', self.__namespaces):
      interval = child.find("pmml:Interval", self.__namespaces)
      if interval is None:
        continue
      data_type = "double" if child.attrib['dataType'] == "double" else "int"
      self.__model_features_list_dict.append({
        "name" : child.attrib['name'].replace('-','_'), 
        "type" : data_type,
        "min"  : interval.attrib['leftMargin'],
        "max"  : interval.attrib['rightMargin']})

  def __get_classes(self, root):
    for child in root.find("pmml:DataDictionary", self.__namespaces).findall('pmml:DataField', self.__namespaces):
      if child.find("pmml:Interval", self.__namespaces) is None:
        for element in child.findall("pmml:Value", self.__namespaces):
          self.__model_classes_list_str.append(element.attrib['value'].replace('-','_'))
                  
  def __get_tree_model(self, tree_name, tree_model_root):
    tree = Node('Node_' + tree_model_root.attrib['id'], feature = "", operator = "", threshold_value = "", boolean_expression = "")
    self.__get_tree_nodes_recursively(tree_model_root, tree)
    return DecisionTree(tree_name, tree, self.__model_features_list_dict, self.__model_classes_list_str, self.__als_lut_tech, self.__als_catalog_cache, self.__als_smt_timeout)

  def __get_tree_nodes_recursively(self, element_tree_node, parent_tree_node):
    children = element_tree_node.findall("pmml:Node", self.__namespaces);
    if len(children) > 2:
      print("Only binary trees are supported. Aborting")
      sys.exit(2)
    for child in children: 
      boolean_expression = parent_tree_node.boolean_expression
      if boolean_expression:
        boolean_expression += " & "
      predicate = None
      compound_predicate = child.find("pmml:CompoundPredicate", self.__namespaces)
      if compound_predicate:
        predicate = next(item for item in compound_predicate.findall("pmml:SimplePredicate", self.__namespaces) if item.attrib["operator"] != "isMissing")
      else:
        predicate = child.find("pmml:SimplePredicate", self.__namespaces)
      if predicate is not None:
        feature         = predicate.attrib['field'].replace('-','_')
        operator        = predicate.attrib['operator']
        threshold_value = predicate.attrib['value']
        if operator in ('equal','lessThan','greaterThan'):
          parent_tree_node.feature         = feature
          parent_tree_node.operator        = operator
          parent_tree_node.threshold_value = threshold_value
          boolean_expression += parent_tree_node.name
        else:
          boolean_expression += "~" + parent_tree_node.name
      if child.find("pmml:Node", self.__namespaces) is None:
        Node('Node_' + child.attrib['id'], parent = parent_tree_node, score = child.attrib['score'].replace('-','_'), boolean_expression = boolean_expression)
      else:
        new_tree_node = Node('Node_' + child.attrib['id'], parent = parent_tree_node, feature = "", operator = "", threshold_value = "", boolean_expression = boolean_expression)
        self.__get_tree_nodes_recursively(child, new_tree_node)

def get_xmlns_uri(elem):
  if elem.tag[0] == "{":
    uri, ignore, tag = elem.tag[1:].partition("}")
  else:
    uri = None
  return uri