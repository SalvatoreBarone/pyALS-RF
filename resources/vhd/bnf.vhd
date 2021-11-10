-- Copyright 2020-2021 Salvatore Barone <salvatore.barone@unina.it>
-- 
-- This is free software; you can redistribute it and/or modify it under
-- the terms of the GNU General Public License as published by the Free
-- Software Foundation; either version 3 of the License, or any later version.
-- 
-- This is distributed in the hope that it will be useful, but WITHOUT
-- ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
-- FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
-- more details.
-- 
-- You should have received a copy of the GNU General Public License along with
-- RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
-- Street, Fifth Floor, Boston, MA 02110-1301, USA.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package bnf is
  function func_or(op0, op1: std_logic) return std_logic;
  function func_or(op0, op1, op2: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61, op62: std_logic) return std_logic;
  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61, op62, op63: std_logic) return std_logic;
  
  function func_and(op0, op1: std_logic) return std_logic;
  function func_and(op0, op1, op2: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61, op62: std_logic) return std_logic;
  function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61, op62, op63: std_logic) return std_logic;
end package;

package body bnf is
  function func_or(op0, op1: std_logic) return std_logic is
  begin
    return op0 or op1;
  end function;

  function func_or(op0, op1, op2: std_logic) return std_logic is
  begin
    return op0 or op1 or op2;
  end function;

  function func_or(op0, op1, op2, op3: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3;
  end function;

  function func_or(op0, op1, op2, op3, op4: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54 or op55;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54 or op55 or op56;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54 or op55 or op56 or op57;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54 or op55 or op56 or op57 or op58;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54 or op55 or op56 or op57 or op58 or op59;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54 or op55 or op56 or op57 or op58 or op59 or op60;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54 or op55 or op56 or op57 or op58 or op59 or op60 or op61;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61, op62: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54 or op55 or op56 or op57 or op58 or op59 or op60 or op61 or op62;
  end function;

  function func_or(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61, op62, op63: std_logic) return std_logic is
  begin
    return op0 or op1 or op2 or op3 or op4 or op5 or op6 or op7 or op8 or op9 or op10 or op11 or op12 or op13 or op14 or op15 or op16 or op17 or op18 or op19 or op20 or op21 or op22 or op23 or op24 or op25 or op26 or op27 or op28 or op29 or op30 or op31 or op32 or op33 or op34 or op35 or op36 or op37 or op38 or op39 or op40 or op41 or op42 or op43 or op44 or op45 or op46 or op47 or op48 or op49 or op50 or op51 or op52 or op53 or op54 or op55 or op56 or op57 or op58 or op59 or op60 or op61 or op62 or op63;
  end function;

  function func_and(op0, op1: std_logic) return std_logic is
    begin
      return op0 and op1;
    end function;
  
    function func_and(op0, op1, op2: std_logic) return std_logic is
    begin
      return op0 and op1 and op2;
    end function;
  
    function func_and(op0, op1, op2, op3: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3;
    end function;
  
    function func_and(op0, op1, op2, op3, op4: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54 and op55;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54 and op55 and op56;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54 and op55 and op56 and op57;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54 and op55 and op56 and op57 and op58;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54 and op55 and op56 and op57 and op58 and op59;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54 and op55 and op56 and op57 and op58 and op59 and op60;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54 and op55 and op56 and op57 and op58 and op59 and op60 and op61;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61, op62: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54 and op55 and op56 and op57 and op58 and op59 and op60 and op61 and op62;
    end function;
  
    function func_and(op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16, op17, op18, op19, op20, op21, op22, op23, op24, op25, op26, op27, op28, op29, op30, op31, op32, op33, op34, op35, op36, op37, op38, op39, op40, op41, op42, op43, op44, op45, op46, op47, op48, op49, op50, op51, op52, op53, op54, op55, op56, op57, op58, op59, op60, op61, op62, op63: std_logic) return std_logic is
    begin
      return op0 and op1 and op2 and op3 and op4 and op5 and op6 and op7 and op8 and op9 and op10 and op11 and op12 and op13 and op14 and op15 and op16 and op17 and op18 and op19 and op20 and op21 and op22 and op23 and op24 and op25 and op26 and op27 and op28 and op29 and op30 and op31 and op32 and op33 and op34 and op35 and op36 and op37 and op38 and op39 and op40 and op41 and op42 and op43 and op44 and op45 and op46 and op47 and op48 and op49 and op50 and op51 and op52 and op53 and op54 and op55 and op56 and op57 and op58 and op59 and op60 and op61 and op62 and op63;
    end function;

  end package body;
