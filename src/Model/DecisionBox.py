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
import copy
from enum import Enum
from pyalslib import double_to_hex, apply_mask_to_double, apply_mask_to_int

class DecisionBox:
  class CompOperator(Enum):
    lessThan = 1
    equal = 2
    greaterThan = 3
    
  def __init__(self, box_name = None, feature_name = None, data_type = None, operator = None, threashold = None, nab = 0):
    self.name = box_name
    self.feature_name = feature_name
    self.data_type = data_type
    if operator:
      if operator == "greaterThan":
          self.operator = DecisionBox.CompOperator.greaterThan
      elif operator == "lessThan":
        self.operator = DecisionBox.CompOperator.lessThan
      elif operator == "equal":
        self.operator = DecisionBox.CompOperator.equal
      else:
        raise Exception("Sorry, operator not recognized") 
    self.threshold = threashold
    self.nab = int(nab)

  def __deepcopy__(self, memo = None):
    box = DecisionBox()
    box.name = copy.deepcopy(self.name)
    box.feature_name = copy.deepcopy(self.feature_name)
    box.data_type = copy.deepcopy(self.data_type)
    box.operator = copy.deepcopy(self.operator)
    box.threshold = copy.deepcopy(self.threshold)
    box.nab = copy.deepcopy(self.nab)
    return box

  def get_c_operator(self):
    if self.operator == DecisionBox.CompOperator.greaterThan:
      return ">"
    elif self.operator == DecisionBox.CompOperator.lessThan:
      return "<"
    else: 
      return "==" 

  def get_hexstr_threashold(self):
    if self.data_type == "double":
      return str(double_to_hex(self.threshold))[2:]
    else:
      return hex(int(self.threshold))[2:]

  def get_struct(self):
    c_operator = "=="
    operator = "equals"
    if self.operator == DecisionBox.CompOperator.greaterThan:
      c_operator = ">"
      operator = "greaterThan"
    elif self.operator == DecisionBox.CompOperator.lessThan:
      c_operator = "<"
      operator = "lessThan"
    threshold = str(self.threshold)
    hex_threshold = ""
    if self.data_type == "double":
      hex_threshold = str(double_to_hex(float(self.threshold)))[2:]
    else: 
      hex_threshold = hex(int(self.threshold))[2:]
    return {"name"          : self.name,
            "feature"       : self.feature_name,
            "data_type"     : self.data_type,
            "operator"      : operator,
            "c_operator"    : c_operator,
            "threshold"     : threshold,
            "threshold_hex" : hex_threshold}

  def compare(self, input):
    if self.data_type == "double":
      # Both the input value and the threshold are masked according to the configured number of approximate bits before
      # the comparison takes place.
      if self.nab == 0:
        # Whether no approximation is required, input and threshold are simply converted to the suitable data-type.
        input_to_compare = float(input) 
        threshold = float(self.threshold)
      else:
        input_to_compare = apply_mask_to_double(float(input), self.nab) 
        threshold = apply_mask_to_double(float(self.threshold), self.nab)
    elif self.nab != 0:
      input_to_compare = apply_mask_to_int(int(input), self.nab) 
      threshold = apply_mask_to_int(int(self.threshold), self.nab)
    else:
      # Whether no approximation is required, input and threshold are simply converted to the suitable data-type.
      input_to_compare = int(input) 
      threshold = int(self.threshold)
    if self.operator == DecisionBox.CompOperator.greaterThan:
      return input_to_compare > threshold
    elif self.operator == DecisionBox.CompOperator.lessThan:
      return input_to_compare < threshold
    else: 
      return input_to_compare == threshold
    
      
class FaultedBox:
      
  def __init__(self, box_name, feature_name, data_type, fixed_value):
    self.name = box_name
    self.feature_name = feature_name
    self.data_type = data_type
    self.fixed_value = fixed_value

  def compare(self, input):
    return self.fixed_value 