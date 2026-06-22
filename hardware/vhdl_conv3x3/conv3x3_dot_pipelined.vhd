-- conv3x3_dot_pipelined.vhd
-- Pipelined 3×3 signed INT8 dot-product with INT32 accumulator.
--
-- This is NOT a full U-Net or Conv2d implementation.
-- It extends conv3x3_dot.vhd with pipeline registers so each arithmetic
-- stage can be mapped to separate FPGA fabric, enabling higher Fmax at
-- the cost of 3-cycle latency.
--
-- Pipeline stages:
--   Stage 1 (Multiply)    : register 9 products p_i*w_i  and bias
--   Stage 2 (Partial-Sum) : register 3 partial row sums  and bias
--   Stage 3 (Final-Out)   : register y = bias + row0 + row1 + row2
--
-- Latency  : 3 clock cycles (valid_in to valid_out)
-- Throughput: 1 result/cycle once the pipeline is full
--
-- Bit widths:
--   p_i, w_i  : signed(7  downto 0)   INT8  input pixel / weight
--   products  : signed(15 downto 0)   INT16 (auto from numeric_std "*")
--   row sums  : signed(31 downto 0)   INT32 (safe: 3 × max 16384 = 49152)
--   bias, y   : signed(31 downto 0)   INT32
--
-- FPGA mapping (Artix-7 / Cmod A7-35T):
--   9 multiplications → 9 DSP48E1 slices (one per multiply)
--   Partial-sum and final adders → LUT fabric (fast carry chains)
--   3×3 kernel requires ~9 DSPs; 10 parallel engines fit in 90 DSPs.
--   Pipelining raises Fmax: each stage's critical path is shorter than
--   the un-pipelined version, letting Vivado meet tighter constraints.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity conv3x3_dot_pipelined is
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;             -- synchronous reset (active high)
        valid_in : in  std_logic;             -- asserted for exactly one cycle per input
        -- 3×3 pixel window (row-major: p0 p1 p2 / p3 p4 p5 / p6 p7 p8)
        p0 : in signed(7 downto 0);
        p1 : in signed(7 downto 0);
        p2 : in signed(7 downto 0);
        p3 : in signed(7 downto 0);
        p4 : in signed(7 downto 0);
        p5 : in signed(7 downto 0);
        p6 : in signed(7 downto 0);
        p7 : in signed(7 downto 0);
        p8 : in signed(7 downto 0);
        -- 3×3 weight kernel (row-major, same spatial ordering as pixels)
        w0 : in signed(7 downto 0);
        w1 : in signed(7 downto 0);
        w2 : in signed(7 downto 0);
        w3 : in signed(7 downto 0);
        w4 : in signed(7 downto 0);
        w5 : in signed(7 downto 0);
        w6 : in signed(7 downto 0);
        w7 : in signed(7 downto 0);
        w8 : in signed(7 downto 0);
        -- INT32 channel bias (pipelined alongside the data)
        bias      : in  signed(31 downto 0);
        -- outputs: result valid 3 cycles after valid_in
        valid_out : out std_logic;
        y         : out signed(31 downto 0)
    );
end entity conv3x3_dot_pipelined;

architecture rtl of conv3x3_dot_pipelined is

    -- ----------------------------------------------------------------
    -- Stage 1 pipeline registers
    -- Nine 16-bit products, pipelined bias, pipelined valid flag.
    -- signed(7..0) * signed(7..0) = signed(15..0) per IEEE numeric_std.
    -- ----------------------------------------------------------------
    signal pr0_s1, pr1_s1, pr2_s1,
           pr3_s1, pr4_s1, pr5_s1,
           pr6_s1, pr7_s1, pr8_s1 : signed(15 downto 0);
    signal bias_s1  : signed(31 downto 0);
    signal valid_s1 : std_logic;

    -- ----------------------------------------------------------------
    -- Stage 2 pipeline registers
    -- Three partial row sums at 32 bits (safe for 3 × INT16 products).
    -- Bias and valid flag continue propagating.
    -- ----------------------------------------------------------------
    signal sum_row0_s2 : signed(31 downto 0);   -- pr0 + pr1 + pr2
    signal sum_row1_s2 : signed(31 downto 0);   -- pr3 + pr4 + pr5
    signal sum_row2_s2 : signed(31 downto 0);   -- pr6 + pr7 + pr8
    signal bias_s2     : signed(31 downto 0);
    signal valid_s2    : std_logic;

    -- ----------------------------------------------------------------
    -- Stage 3 pipeline registers
    -- Final output: bias + sum of all three row sums.
    -- ----------------------------------------------------------------
    signal y_s3     : signed(31 downto 0);
    signal valid_s3 : std_logic;

begin

    -- ================================================================
    -- Stage 1 — multiply and register
    -- ================================================================
    stage1 : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                pr0_s1 <= (others => '0');  pr1_s1 <= (others => '0');
                pr2_s1 <= (others => '0');  pr3_s1 <= (others => '0');
                pr4_s1 <= (others => '0');  pr5_s1 <= (others => '0');
                pr6_s1 <= (others => '0');  pr7_s1 <= (others => '0');
                pr8_s1 <= (others => '0');
                bias_s1  <= (others => '0');
                valid_s1 <= '0';
            else
                -- Nine parallel multiplications: each infers one DSP48E1
                pr0_s1 <= p0 * w0;  pr1_s1 <= p1 * w1;  pr2_s1 <= p2 * w2;
                pr3_s1 <= p3 * w3;  pr4_s1 <= p4 * w4;  pr5_s1 <= p5 * w5;
                pr6_s1 <= p6 * w6;  pr7_s1 <= p7 * w7;  pr8_s1 <= p8 * w8;
                bias_s1  <= bias;
                valid_s1 <= valid_in;
            end if;
        end if;
    end process stage1;

    -- ================================================================
    -- Stage 2 — partial-sum and register
    -- Group into three row sums (3 additions per group).
    -- resize() sign-extends INT16 products to INT32 before adding.
    -- ================================================================
    stage2 : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                sum_row0_s2 <= (others => '0');
                sum_row1_s2 <= (others => '0');
                sum_row2_s2 <= (others => '0');
                bias_s2  <= (others => '0');
                valid_s2 <= '0';
            else
                sum_row0_s2 <= resize(pr0_s1, 32) + resize(pr1_s1, 32) + resize(pr2_s1, 32);
                sum_row1_s2 <= resize(pr3_s1, 32) + resize(pr4_s1, 32) + resize(pr5_s1, 32);
                sum_row2_s2 <= resize(pr6_s1, 32) + resize(pr7_s1, 32) + resize(pr8_s1, 32);
                bias_s2  <= bias_s1;
                valid_s2 <= valid_s1;
            end if;
        end if;
    end process stage2;

    -- ================================================================
    -- Stage 3 — final sum and register
    -- y = bias + row0 + row1 + row2
    -- ================================================================
    stage3 : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                y_s3     <= (others => '0');
                valid_s3 <= '0';
            else
                y_s3     <= bias_s2 + sum_row0_s2 + sum_row1_s2 + sum_row2_s2;
                valid_s3 <= valid_s2;
            end if;
        end if;
    end process stage3;

    -- ================================================================
    -- Output ports driven from stage-3 registers
    -- ================================================================
    y         <= y_s3;
    valid_out <= valid_s3;

end architecture rtl;
