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

entity tb_parallel_comparator is
end entity tb_parallel_comparator;

architecture testbench of tb_parallel_comparator is
  component parallel_comparator is
    generic(
      data_width     : natural;
      comp_operator  : comp_operator_t;
      parallel_blocks: natural);
    port ( 
      clock   : in  std_logic;
      reset_n : in  std_logic;
      enable  : in  std_logic;
      data_1  : in  std_logic_vector (data_width-1 downto 0);
      data_2  : in  std_logic_vector (data_width-1 downto 0);
      result  : out std_logic);
  end component;
  constant data_width     : natural                                  := 32;
  constant parallel_blocks: natural                                  := 4;
  signal   clock          : std_logic                                := '0';
  signal   reset_n        : std_logic                                := '0';
  signal   enable         : std_logic                                := '0';
  signal   data_1         : std_logic_vector (data_width-1 downto 0) := (others => '0');
  signal   data_2         : std_logic_vector (data_width-1 downto 0) := (others => '0');
  signal   result_eq      : std_logic                                := '0';
  signal   result_lt      : std_logic                                := '0';
  signal   result_gt      : std_logic                                := '0';
  constant clock_period   : time                                     := 10 ns;
  signal   simulate       : std_logic                                := '1';
  file     test_oracle    : text;
begin
  uut_eq : parallel_comparator
    generic map(data_width, equal, parallel_blocks)
    port map (clock, reset_n, enable, data_1, data_2, result_eq);
  uut_gt : parallel_comparator
    generic map(data_width, greaterThan, parallel_blocks)
    port map (clock, reset_n, enable, data_1, data_2, result_gt);
  uut_lt : parallel_comparator
    generic map(data_width, lessThan, parallel_blocks)
    port map (clock, reset_n, enable, data_1, data_2, result_lt);
  clock_process : process
	begin
		while simulate = '1' loop
			clock <= not clock;
			wait for clock_period / 2;
		end loop;
		wait;
	end process clock_process;
  stim_process : process
    variable rline              : line;
    variable space              : character;
    variable read_data_1        : std_logic_vector (data_width-1 downto 0);
    variable read_data_2        : std_logic_vector (data_width-1 downto 0);
    variable read_result_eq     : std_logic;
    variable read_result_lt     : std_logic;
    variable read_result_gt     : std_logic;
  begin
    reset_n <= '0';
    wait for 10 ns;	
    reset_n <= '1';
    enable <= '1';
    file_open(test_oracle, "../tb/tb_parallel_comparator.txt", read_mode);
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
      wait for 2*clock_period;
      assert result_eq = read_result_eq report "Error! data_1=" & vec_image(data_1) & " data_2=" & vec_image(data_2) & " result_eq=" & std_logic'image(result_eq) & " read_result_eq=" & std_logic'image(read_result_eq) severity failure;
      assert result_lt = read_result_lt report "Error! data_1=" & vec_image(data_1) & " data_2=" & vec_image(data_2) & " result_lt=" & std_logic'image(result_lt) & " read_result_lt=" & std_logic'image(read_result_lt) severity failure;
      assert result_gt = read_result_gt report "Error! data_1=" & vec_image(data_1) & " data_2=" & vec_image(data_2) & " result_gt=" & std_logic'image(result_gt) & " read_result_gt=" & std_logic'image(read_result_gt) severity failure;
    end loop;
    simulate <= '0';
    wait;
  end process;
end testbench;
 
