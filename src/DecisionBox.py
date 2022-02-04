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
import copy
from enum import Enum
from .Utility import *

class DecisionBox:
  class CompOperator(Enum):
    lessThan = 1
    equal = 2
    greaterThan = 3
    
  def __init__(self, box_name = None, feature_name = None, data_type = None, operator = None, threashold = None, nab = 0):
    self.__name = box_name
    self.__feature_name = feature_name
    self.__data_type = data_type
    if operator:
      if operator == "greaterThan":
          self.__operator = DecisionBox.CompOperator.greaterThan
      elif operator == "lessThan":
        self.__operator = DecisionBox.CompOperator.lessThan
      elif operator == "equal":
        self.__operator = DecisionBox.CompOperator.equal
      else:
        raise Exception("Sorry, operator not recognized") 
    self.__threshold = threashold
    self.__nab = int(nab)

  def __deepcopy__(self, memo = None):
    box = DecisionBox()
    box.__name = copy.deepcopy(self.__name)
    box.__feature_name = copy.deepcopy(self.__feature_name)
    box.__data_type = copy.deepcopy(self.__data_type)
    box.__operator = copy.deepcopy(self.__operator)
    box.__threshold = copy.deepcopy(self.__threshold)
    box.__nab = copy.deepcopy(self.__nab)
    return box

  def get_name(self):
    return self.__name

  def get_feature(self):
    return self.__feature_name

  def get_data_type(self):
    return self.__data_type

  def get_operator(self):
    return self.__operator

  def get_c_operator(self):
    if self.__operator == DecisionBox.CompOperator.greaterThan:
      return ">"
    elif self.__operator == DecisionBox.CompOperator.lessThan:
      return "<"
    else: 
      return "==" 

  def get_threshold(self):
    return self.__threshold

  def get_hexstr_threashold(self):
    if self.__data_type == "double":
      return str(double_to_hex(self.__threshold))[2:]
    else:
      return str(hex(int(self.__threshold)))[2:]

  def get_struct(self):
    c_operator = "=="
    operator = "equals"
    if self.__operator == DecisionBox.CompOperator.greaterThan:
      c_operator = ">"
      operator = "greaterThan"
    elif self.__operator == DecisionBox.CompOperator.lessThan:
      c_operator = "<"
      operator = "lessThan"
    threshold = str(self.__threshold)
    hex_threshold = ""
    if self.__data_type == "double":
      hex_threshold = str(double_to_hex(float(self.__threshold)))[2:]
    else: 
      hex_threshold = str(hex(int(self.__threshold)))[2:]
    return {"name"          : self.__name,
            "feature"       : self.__feature_name,
            "data_type"     : self.__data_type,
            "operator"      : operator,
            "c_operator"    : c_operator,
            "threshold"     : threshold,
            "threshold_hex" : hex_threshold}

  def get_nab(self):
    return self.__nab

  def set_nab(self, nab):
    self.__nab = int(nab)

  def compare(self, input):
    if self.__data_type == "double":
      # Both the input value and the threshold are masked according to the configured number of approximate bits before
      # the comparison takes place.
      if self.__nab != 0:
        input_to_compare = apply_mask_to_double(float(input), self.__nab) 
        threshold = apply_mask_to_double(float(self.__threshold), self.__nab)
      else:
        # Whether no approximation is required, input and threshold are simply converted to the suitable data-type.
        input_to_compare = float(input) 
        threshold = float(self.__threshold)
    else:
      # Both the input value and the threshold are masked according to the configured number of approximate bits before
      # the comparison takes place.
      if self.__nab != 0:
        input_to_compare = apply_mask_to_int(int(input), self.__nab) 
        threshold = apply_mask_to_int(int(self.__threshold), self.__nab)
      else:
        # Whether no approximation is required, input and threshold are simply converted to the suitable data-type.
        input_to_compare = int(input) 
        threshold = int(self.__threshold)
    if self.__operator == DecisionBox.CompOperator.greaterThan:
      return bool(input_to_compare > threshold)
    elif self.__operator == DecisionBox.CompOperator.lessThan:
      return bool(input_to_compare < threshold)
    else: 
      return bool(input_to_compare == threshold)
