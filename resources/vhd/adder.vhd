library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity full_adder is 
port(
    bit_1       : in std_logic;
    bit_2       : in std_logic;
    carry_in    : in std_logic;
    res         : out std_logic;
    carry_out   : out std_logic);
end full_adder;
architecture Behavioural of full_adder is 
begin 
    res         <= (bit_1 xor bit_2 xor carry_in);
    carry_out   <= (bit_1 and bit_2) or (bit_2 and carry_in) or (bit_1 and carry_in); 
end Behavioural;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
entity fp_adder is
    generic(
        data_width      : natural);
    port ( 
        data_1  : in  std_logic_vector (data_width-1 downto 0);
        data_2  : in  std_logic_vector (data_width-1 downto 0);
        carry_in: in  std_logic;
        result  : out std_logic_vector (data_width-1 downto 0);
        overflow: out std_logic;
        clock : std_logic;
        reset_n : std_logic;
        enable : std_logic);
end fp_adder;

architecture Structural of fp_adder is
  component full_adder is
    port(
        bit_1       : in std_logic;
        bit_2       : in std_logic;
        carry_in    : in std_logic;
        res         : out std_logic;
        carry_out   : out std_logic);
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
signal carry_help : std_logic_vector (data_width downto 0) := (others => '0');
signal out_pipe_reg_1: std_logic_vector( data_width - 1 downto 0) := (others => '0');
signal out_pipe_reg_2: std_logic_vector( data_width - 1 downto 0) := (others => '0');
begin 
    
 -- Save the two input values
 input_collect_1 : pipe_reg 
    generic map( data_width => data_width)
    port map(
        clock => clock,
        reset_n => reset_n,
        enable => enable,
        data_in => data_1,
        data_out => out_pipe_reg_1
    );
 input_collect_2 : pipe_reg 
    generic map( data_width => data_width)
    port map(
        clock => clock,
        reset_n => reset_n,
        enable => enable,
        data_in => data_2,
        data_out => out_pipe_reg_2
    );
-- Map all the input into this value.
out_assignment: for i in 0 to data_width - 1 generate
    FA : full_adder
      port map (
        bit_1       =>  out_pipe_reg_1(i), 
        bit_2       =>  out_pipe_reg_2(i),
        carry_in    =>  carry_help(i),
        res         =>  result(i),
        carry_out   =>  carry_help(i+1)
      );
end generate;

-- Carry help 0 is the input carry
carry_help(0) <= carry_in;
-- The last one bit is the carry out
overflow <= carry_help(data_width);
end Structural;