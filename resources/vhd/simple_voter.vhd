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
entity simple_voter is
    generic	(
        data_width  : natural;
        pipe_stages : natural;
        threshold : natural);
  port (
    clock    : in  std_logic;
        reset_n  : in  std_logic;
        data_in  : in  std_logic_vector (data_width-1 downto 0);
       majority : out std_logic);
end simple_voter;

architecture simple_voter of simple_voter is
    component swapper_block is
    generic (data_width : natural);
    port (
      data_in  : in std_logic_vector(data_width-1 downto 0);
      data_out : out std_logic_vector(data_width-1 downto 0));
  end component;
  component pipe_reg is
    generic(data_width : natural);
    port ( 
      clock    : in  std_logic;
      reset_n  : in  std_logic;
      enable   : in  std_logic;
      data_in  : in  std_logic_vector (data_width-1 downto 0);
      data_out : out std_logic_vector (data_width-1 downto 0));
  end component;
    constant swapper_per_pipe : natural := data_width / pipe_stages;
    type matrix is array (natural range <>) of std_logic_vector (data_width-1 downto 0);
    signal intermediates : matrix (0 to data_width + pipe_stages);
begin
    assert pipe_stages mod 2 = 0 report "pipe_stages must be a multiple of two" severity failure;
    assert swapper_per_pipe >= 2 report "too many pipe stages" severity failure;
    data_in_buffer : pipe_reg	generic map (data_width => data_width)	port map (clock => clock, reset_n => reset_n, enable => '1', data_in => data_in, data_out => intermediates(0));
    majority <=	intermediates(data_width + pipe_stages)(threshold);
    chain : for i in 0 to data_width + pipe_stages - 1 generate
        pipe : if (i+1) mod (swapper_per_pipe+1) = 0 generate 
              pipe_buffer: pipe_reg	
                generic map (data_width => data_width) 
                port map (clock => clock, reset_n => reset_n, enable => '1', data_in => intermediates(i), data_out => intermediates(i+1));
        end generate;
        swapper : if (i+1) mod (swapper_per_pipe+1) /= 0 generate 
            swapper_inst: swapper_block
                generic map (data_width => data_width)
                port map(data_in => intermediates(i), data_out => intermediates(i+1));
        end generate;
    end generate;
end architecture;

