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
use std.textio.all;
use ieee.std_logic_textio.all;

library work;
use work.debug_func.all;
use work.common_types.all;

entity tb_basic_comparator is
end entity tb_basic_comparator;

architecture testbench of tb_basic_comparator is
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
  constant data_width     : natural                                  := 32;
  signal   data_1         : std_logic_vector (data_width-1 downto 0) := (others => '0');
  signal   data_2         : std_logic_vector (data_width-1 downto 0) := (others => '0');
  signal   result_eq      : std_logic                                := '0';
  signal   equal_eq       : std_logic                                := '0';
  signal   result_lt      : std_logic                                := '0';
  signal   equal_lt       : std_logic                                := '0';
  signal   result_gt      : std_logic                                := '0';
  signal   equal_gt       : std_logic                                := '0';
  file     test_oracle    : text;
begin
  uut_eq : basic_comparator generic map(data_width, equal)       port map(data_1, data_2, result_eq, equal_eq);
  uut_gt : basic_comparator generic map(data_width, greaterThan) port map(data_1, data_2, result_gt , equal_eq);
  uut_lt : basic_comparator generic map(data_width, lessThan)    port map(data_1, data_2, result_lt, equal_eq);
  stim_process : process
    variable rline              : line;
    variable space              : character;
    variable read_data_1        : std_logic_vector (data_width-1 downto 0);
    variable read_data_2        : std_logic_vector (data_width-1 downto 0);
    variable read_result_eq     : std_logic;
    variable read_result_lt     : std_logic;
    variable read_result_gt     : std_logic;
  begin
    wait for 10 ns;	
    file_open(test_oracle, "../tb/tb_basic_comparator.txt", read_mode);
    while not endfile(test_oracle) loop
      readline(test_oracle, rline);
      read(rline, read_data_1);
      read(rline, space);
      read(rline, read_data_2);
      read(rline, space);
      read(rline, read_result_eq);
      read(rline, space);
      read(rline, read_result_lt);
      read(rline, space);
      read(rline, read_result_gt);
      data_1 <= read_data_1;
      data_2 <= read_data_2;
      wait for 10 ns;
      assert equal_eq = read_result_eq report "Error! data_1=" & vec_image(data_1) & " data_2=" & vec_image(data_2) & " equal_eq=" & std_logic'image(equal_eq) & " read_result_eq=" & std_logic'image(read_result_eq) severity failure;
      assert equal_lt = read_result_eq report "Error! data_1=" & vec_image(data_1) & " data_2=" & vec_image(data_2) & " equal_lt=" & std_logic'image(equal_lt) & " read_result_eq=" & std_logic'image(read_result_eq) severity failure;
      assert equal_gt = read_result_eq report "Error! data_1=" & vec_image(data_1) & " data_2=" & vec_image(data_2) & " equal_gt=" & std_logic'image(equal_gt) & " read_result_eq=" & std_logic'image(read_result_eq) severity failure;
      assert result_eq = read_result_eq report "Error! data_1=" & vec_image(data_1) & " data_2=" & vec_image(data_2) & " result_eq=" & std_logic'image(result_eq) & " read_result_eq=" & std_logic'image(read_result_eq) severity failure;
      assert result_lt = read_result_lt report "Error! data_1=" & vec_image(data_1) & " data_2=" & vec_image(data_2) & " result_lt=" & std_logic'image(result_lt) & " read_result_lt=" & std_logic'image(read_result_lt) severity failure;
      assert result_gt = read_result_gt report "Error! data_1=" & vec_image(data_1) & " data_2=" & vec_image(data_2) & " result_gt=" & std_logic'image(result_gt) & " read_result_gt=" & std_logic'image(read_result_gt) severity failure;
    end loop;
    wait;
  end process;
end testbench;
 
