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

package common_types is 
  type comp_operator_t is (equal, greaterThan, lessThan);
	type data_type_t is (double, int);
end package;

------------------------------------------------------------------------------------------------------------------------
-- Basic comparator entity
library ieee;
use ieee.std_logic_1164.all;
library work;
use work.common_types.all;
entity basic_comparator is
  generic(
    data_width     : natural;
    comp_operator  : comp_operator_t);
  port (
    data_1 : in  std_logic_vector (data_width-1 downto 0);
    data_2 : in  std_logic_vector (data_width-1 downto 0);
    result : out std_logic;
    equals : out std_logic);
end basic_comparator;
architecture data_flow of basic_comparator is
begin
  equals <= '1' when data_1 = data_2 else '0';
  eq_comp : if comp_operator = equal generate
    result <= '1' when data_1 = data_2 else '0';
  end generate;
  gt_comp : if comp_operator = greaterThan generate
    result <= '1' when data_1 > data_2 else '0';
  end generate;
  lt_comp : if comp_operator = lessThan generate
    result <= '1' when data_1 < data_2 else '0';
  end generate;
end data_flow;


------------------------------------------------------------------------------------------------------------------------
-- Piped comparator entity
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
library work;
use work.common_types.all;
entity parallel_comparator is
  generic(
    data_width      : natural;
    comp_operator   : comp_operator_t;
    parallel_blocks : natural);
  port ( 
    clock   : in  std_logic;
    reset_n : in  std_logic;
    enable  : in  std_logic;
    data_1  : in  std_logic_vector (data_width-1 downto 0);
    data_2  : in  std_logic_vector (data_width-1 downto 0);
    result  : out std_logic);
end parallel_comparator;
architecture structural of parallel_comparator is
  component pipe_reg is
    generic(data_width : natural);
    port ( 
      clock    : in  std_logic;
      reset_n  : in  std_logic;
      enable   : in  std_logic;
      data_in  : in  std_logic_vector (data_width-1 downto 0);
      data_out : out std_logic_vector (data_width-1 downto 0));
  end component;
  component basic_comparator is
    generic(
      data_width     : natural;
      comp_operator  : comp_operator_t);
     port (
      data_1 : in  std_logic_vector (data_width-1 downto 0);
      data_2 : in  std_logic_vector (data_width-1 downto 0);
      result : out std_logic;
      equals : out std_logic);
  end component;
  signal   buffered_data_1 : std_logic_vector (data_width-1 downto 0)      := (others => '0');
  signal   buffered_data_2 : std_logic_vector (data_width-1 downto 0)      := (others => '0');
  signal   blocks_result   : std_logic_vector (parallel_blocks-1 downto 0) := (others => '0');
  signal   blocks_equal    : std_logic_vector (parallel_blocks-1 downto 0) := (others => '0');
  signal   partial_result  : std_logic_vector (parallel_blocks downto 0)   := (others => '0');
  constant bits_per_stage  : natural                                       := data_width / parallel_blocks;
  -- When using bitwidth that is not a multiple of parallel_blocks (e.g. when using precision-scaling), reducing the
  -- comparator into blocks may cause the most significant bits to not be compared (the division
  -- data_width / parallel_blocks may have non-zero remainder). For this reason, we keep track of any bits that are not
  -- compared and extend the capacity (the bitwidth) of the last comparator so that they are not excluded from the
  -- comparison.
  constant additional_bits : natural                                       := data_width mod parallel_blocks;
  constant last_comp_bits  : natural                                       := bits_per_stage + additional_bits;
begin
  inbuff_1 : pipe_reg generic map (data_width) port map (clock, reset_n, enable, data_1, buffered_data_1);
  inbuff_2 : pipe_reg generic map (data_width) port map (clock, reset_n, enable, data_2, buffered_data_2);
  comp_blocks : for i in 0 to parallel_blocks-2 generate
    comp : basic_comparator
      generic map (bits_per_stage, comp_operator)
      port map (
        buffered_data_1((i+1)*bits_per_stage-1 downto i*bits_per_stage),
        buffered_data_2((i+1)*bits_per_stage-1 downto i*bits_per_stage),
        blocks_result(i),
        blocks_equal(i));
  end generate;
  -- When using bitwidth that is not a multiple of parallel_blocks (e.g. when using precision-scaling), reducing the
  -- comparator into blocks may cause the most significant bits to not be compared (the division
  -- data_width / parallel_blocks may have non-zero remainder). For this reason, we keep track of any bits that are not
  -- compared and extend the capacity (the bitwidth) of the last comparator so that they are not excluded from the
  -- comparison.
  last_block : basic_comparator
  generic map (last_comp_bits, comp_operator)
  port map (
    buffered_data_1(data_width-1 downto data_width-last_comp_bits),
    buffered_data_2(data_width-1 downto data_width-last_comp_bits),
    blocks_result(parallel_blocks-1),
    blocks_equal(parallel_blocks-1));
  
  result_eq : if comp_operator = equal generate
    result <= and_reduce(blocks_equal);
  end generate;
  result_glt : if comp_operator = greaterThan or comp_operator = lessThan generate
    result <= partial_result(parallel_blocks);
    partial_result(0) <= '0';
    res_gen : for i in 0 to parallel_blocks-1 generate
      partial_result(i+1) <= blocks_result(i) or (partial_result(i) and blocks_equal(i));
    end generate;
  end generate;
end structural;


------------------------------------------------------------------------------------------------------------------------
-- Floating-point comparator entity
library ieee;
use ieee.std_logic_1164.all;
library work;
use work.common_types.all;
entity db_float is
  generic(
    data_width      : natural;
    comp_operator   : comp_operator_t;
    parallel_blocks : natural);
  port ( 
    clock   : in  std_logic;
    reset_n : in  std_logic;
    enable  : in  std_logic;
    data_1  : in  std_logic_vector (data_width-1 downto 0);
    data_2  : in  std_logic_vector (data_width-1 downto 0);
    result  : out std_logic);
end db_float;
architecture structural of db_float is
  component parallel_comparator is
    generic(
      data_width      : natural;
      comp_operator   : comp_operator_t;
      parallel_blocks : natural);
    port ( 
      clock   : in  std_logic;
      reset_n : in  std_logic;
      enable  : in  std_logic;
      data_1  : in  std_logic_vector (data_width-1 downto 0);
      data_2  : in  std_logic_vector (data_width-1 downto 0);
      result  : out std_logic);
  end component;
  alias  sign_a     : std_logic                               is data_1(data_width-1);
  alias  modulo_a   : std_logic_vector(data_width-2 downto 0) is data_1(data_width-2 downto 0);
  signal abs_data_1 : std_logic_vector(data_width-1 downto 0) := (others => '0');
  alias  sign_b     : std_logic                               is data_2(data_width-1);
  alias  modulo_b   : std_logic_vector(data_width-2 downto 0) is data_2(data_width-2 downto 0);
  signal abs_data_2 : std_logic_vector(data_width-1 downto 0) := '0' & data_2(data_width-2 downto 0);

  signal a_eq_b     : std_logic                               := '0';
  signal a_gt_b     : std_logic                               := '0';
  signal conditions : std_logic_vector(3 downto 0)            := (others => '0');
begin
  abs_data_1 <= '0' & modulo_a;
  abs_data_2 <= '0' & modulo_b;
  eq_comp : parallel_comparator
    generic map (data_width, equal, parallel_blocks)
    port map (clock, reset_n, enable, abs_data_1, abs_data_2, a_eq_b);
  gt_comp : parallel_comparator
    generic map (data_width, greaterThan, parallel_blocks)
    port map (clock, reset_n, enable, abs_data_1, abs_data_2, a_gt_b);
  conditions <= sign_a & sign_b & a_gt_b & a_eq_b;
  eq_comp_result : if comp_operator = equal generate
    with conditions select result <= '1' when b"0001" | b"1101", '0' when others;
  end generate;
  gt_comp_result : if comp_operator = greaterThan generate
    with conditions select result <= '1' when b"0010" | b"0100" | b"0101" | b"0110" | b"1100", '0' when others;
  end generate;
  lt_comp_result : if comp_operator = lessThan generate
    with conditions select result <= '1' when b"0000" | b"1000" | b"1001" | b"1010" | b"1110", '0' when others;
  end generate;
end structural;


------------------------------------------------------------------------------------------------------------------------
-- Integer comparator entity
library ieee;
use ieee.std_logic_1164.all;
library work;
use work.common_types.all;
entity db_int is
  generic(
    data_width      : natural;
    comp_operator   : comp_operator_t;
    parallel_blocks : natural);
  port ( 
    clock   : in  std_logic;
    reset_n : in  std_logic;
    enable  : in  std_logic;
    data_1  : in  std_logic_vector (data_width-1 downto 0);
    data_2  : in  std_logic_vector (data_width-1 downto 0);
    result  : out std_logic);
end db_int;
architecture structural of db_int is
  component parallel_comparator is
    generic(
      data_width      : natural;
      comp_operator   : comp_operator_t;
      parallel_blocks : natural);
    port ( 
      clock   : in  std_logic;
      reset_n : in  std_logic;
      enable  : in  std_logic;
      data_1  : in  std_logic_vector (data_width-1 downto 0);
      data_2  : in  std_logic_vector (data_width-1 downto 0);
      result  : out std_logic);
  end component;
begin
  comp : parallel_comparator
    generic map (data_width, comp_operator, parallel_blocks)
    port map (clock, reset_n, enable, data_1, data_2, result);
end structural;


-----------------------------------------------------------------------------------------------------------------------
-- Decision-box entity
library ieee;
use ieee.std_logic_1164.all;
library work;
use work.common_types.all;
entity decision_box is
  generic(
    data_width      : natural;
    data_type       : data_type_t;
    comp_operator   : comp_operator_t;
    parallel_blocks : natural);
  port ( 
    clock   : in  std_logic;
    reset_n : in  std_logic;
    enable  : in  std_logic;
    data_1  : in  std_logic_vector (data_width-1 downto 0);
    data_2  : in  std_logic_vector (data_width-1 downto 0);
    result  : out std_logic);
end decision_box;
architecture structural of decision_box is
  component db_float is
    generic(
      data_width      : natural;
      comp_operator   : comp_operator_t;
      parallel_blocks : natural);
    port ( 
      clock   : in  std_logic;
      reset_n : in  std_logic;
      enable  : in  std_logic;
      data_1  : in  std_logic_vector (data_width-1 downto 0);
      data_2  : in  std_logic_vector (data_width-1 downto 0);
      result  : out std_logic);
  end component;
  component db_int is
    generic(
      data_width      : natural;
      comp_operator   : comp_operator_t;
      parallel_blocks : natural);
    port ( 
      clock   : in  std_logic;
      reset_n : in  std_logic;
      enable  : in  std_logic;
      data_1  : in  std_logic_vector (data_width-1 downto 0);
      data_2  : in  std_logic_vector (data_width-1 downto 0);
      result  : out std_logic);
  end component;
begin
  assert parallel_blocks >= 1 report "parallel_blocks must be greater or equal than 1" severity failure;
  fp_comp : if data_type = double generate
    comp : db_float 
      generic map (data_width, comp_operator, parallel_blocks)
      port map (clock, reset_n, enable, data_1, data_2, result);
  end generate;
  int_comp : if data_type = int generate
    comp : db_int
      generic map (data_width, comp_operator, parallel_blocks)
      port map (clock, reset_n, enable, data_1, data_2, result);
  end generate;
end structural;