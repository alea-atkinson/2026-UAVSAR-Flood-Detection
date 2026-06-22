-- mac_unit.vhd
-- Signed fixed-point multiply-accumulate (MAC) building block.
--
-- Accumulates signed(7 downto 0) products into a signed(31 downto 0) register.
-- This is the primitive operation behind all Conv2d layers in the U-Net:
--   every output pixel = sum of (weight_i * input_i) over a kernel window.
--
-- This is NOT a full U-Net implementation. It is the core arithmetic unit
-- that a Conv2d accelerator would replicate across many parallel lanes.
--
-- Bit-width rationale:
--   INT8 weights/activations: signed(7 downto 0)  -> -128 to +127
--   16-bit product per multiply: covers (-128)*(-128) = 16384  (fits signed(15..0))
--   32-bit accumulator: safely sums thousands of 16-bit products without overflow,
--   matching the INT32 accumulator used in INT8 convolution (e.g. ONNX / TensorRT).

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity mac_unit is
    port (
        clk     : in  std_logic;
        rst     : in  std_logic;             -- synchronous reset
        clear   : in  std_logic;             -- clear accumulator (start new MAC window)
        en      : in  std_logic;             -- enable: load a*b into accumulator this cycle
        a       : in  signed(7 downto 0);   -- pixel / activation (INT8)
        b       : in  signed(7 downto 0);   -- weight (INT8)
        acc_out : out signed(31 downto 0)   -- running dot-product accumulator (INT32)
    );
end entity mac_unit;

architecture rtl of mac_unit is
    signal acc_reg : signed(31 downto 0) := (others => '0');
begin

    process(clk)
        variable product : signed(15 downto 0);
    begin
        if rising_edge(clk) then
            if rst = '1' then
                acc_reg <= (others => '0');
            elsif clear = '1' then
                acc_reg <= (others => '0');
            elsif en = '1' then
                -- a * b produces signed(15 downto 0) per IEEE numeric_std;
                -- resize sign-extends to 32 bits before adding to accumulator.
                product := a * b;
                acc_reg <= acc_reg + resize(product, 32);
            end if;
        end if;
    end process;

    acc_out <= acc_reg;

end architecture rtl;
