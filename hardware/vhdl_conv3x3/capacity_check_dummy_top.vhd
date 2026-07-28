-- capacity_check_dummy_top.vhd
-- Trivial, throwaway top module used ONLY to make Vivado load a target
-- part and produce a report_utilization "Available" column for that
-- part -- i.e. to read the PART's own resource capacity directly from
-- Vivado, rather than assuming it from the part name/family.
--
-- This is NOT a project datapath and is NOT used by, or referenced by,
-- any existing verified VHDL design in this repo. It exists solely to
-- support hardware/vhdl_conv3x3/reports/free_tier_target_capacity_check.md.
-- A single D flip-flop is intentionally the entire design: its own
-- resource USE is irrelevant here (and negligible on every part checked),
-- only the device's reported "Available" totals matter.

library IEEE;
use IEEE.std_logic_1164.all;

entity capacity_check_dummy_top is
    port (
        clk   : in  std_logic;
        rst   : in  std_logic;
        d_in  : in  std_logic;
        q_out : out std_logic
    );
end entity capacity_check_dummy_top;

architecture rtl of capacity_check_dummy_top is
    signal q_reg : std_logic := '0';
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                q_reg <= '0';
            else
                q_reg <= d_in;
            end if;
        end if;
    end process;

    q_out <= q_reg;
end architecture rtl;
