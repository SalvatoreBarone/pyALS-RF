-- Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>
-- 
-- This file has been auto-generated by pyALS-rf
-- https://github.com/SalvatoreBarone/pyALS-rf 
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
use std.textio.all;
use ieee.std_logic_textio.all;

library work;
use work.debug_func.all;

entity tb_voter is
end tb_voter;

architecture tb_voter of tb_voter is
  component voter is
    generic	(
      data_width  : natural;
      pipe_stages : natural);
    port (
      clock    : in  std_logic;
      reset_n  : in  std_logic;
      data_in  : in  std_logic_vector (data_width-1 downto 0);
      majority : out std_logic);
  end component;
	constant data_width   : natural                                  := 16;
	constant pipe_stages  : natural                                  := 4;
	constant clock_period : time                                     := 10 ns;
	signal   clock        : std_logic                                := '0';
	signal   reset_n      : std_logic                                := '0';
	signal   data_in      : std_logic_vector (data_width-1 downto 0) := (others => '0');
	signal   majority     : std_logic                                := '0';
	signal   simulate     : std_logic                                := '1';
	file     test_cases   : text;
begin
	uut : voter generic	map (data_width, pipe_stages)	port map (clock, reset_n, data_in, majority);
	clock_process : process
	begin
		while simulate = '1' loop
			clock <= not clock;
			wait for clock_period / 2;
		end loop;
		wait;
	end process clock_process;
	stim_process : process
		variable rline : line;
		variable v_data_in : std_logic_vector(data_width-1 downto 0);
		variable v_majority : std_logic_vector(0 downto 0);
		variable space : character;
	begin
		reset_n <= '0', '1' after 5*clock_period;
		wait for 7*clock_period;
		file_open(test_cases, "../tb/tb_voter.txt", read_mode);
		while not endfile(test_cases) loop
			readline(test_cases, rline);
			read(rline, v_data_in); 
			read(rline, space);
			read(rline, v_majority);
			data_in <= v_data_in;
			wait for (pipe_stages + 1) * clock_period;
			assert majority = v_majority(0)	report	"Error!"	& " data_in= " & vec_image(data_in)	& " data_out= " & std_logic'image(majority)	& " expected= " & vec_image(v_majority) severity failure;
		end loop;
		simulate <= '0';
		wait;
	end process stim_process;
end architecture;
