-- stream_conv3x3_3chan_kernel21_bn_relu_real_tile_q20.vhd
-- REAL-TILE Q.20-format variant of stream_conv3x3_3chan_kernel21_bn_relu_pipelined.vhd (structurally, if it existed for this kernel).
--
-- This is NOT a new convolution architecture and NOT a new numerical
-- scheme. It is structurally IDENTICAL to
-- stream_conv3x3_3chan_kernel21_bn_relu_pipelined.vhd (structurally, if it existed for this kernel): the SAME existing,
-- unmodified stream_conv3x3_3chan_cell sub-component (kernel-21 folded INT8
-- weights, bias=0), the SAME two-stage registered Q.F BatchNorm-fold + (Q.20 for this variant)
-- ReLU pipeline. The ONLY difference is which package it reads its
-- constants from: first_conv_bn_relu_kernel21_real_tile_q20_pkg (REAL-TILE
-- SCALE_FX, computed from a genuine per-tile activation scale_x) instead
-- of first_conv_bn_relu_kernel0_pkg (toy-patch SCALE_FX, scale_x=1.0
-- exact). The folded INT8 weights themselves are identical in both
-- packages, since weights do not depend on the input.
--
-- ---- Architecture (unchanged from the pipelined design) -----------------
--   stream_conv3x3_3chan_cell (existing, unmodified, kernel-0 folded
--     INT8 weights, bias=0)          -> raw_y (INT32), 4-cycle latency
--   Stage 1 (registered): product_fx_r  <= raw_y * SCALE_FX   (Q.20, exact)
--   Stage 2 (registered): biased_fx     = product_fx_r + BIAS_FX (Q.20)
--                          y_bn_relu_fx_r <= biased_fx if biased_fx > 0
--                                            else 0                 (ReLU)
--
-- ---- Scope ----------------------------------------------------------------
-- Kernel 21 ONLY (of the complete first Conv2d layer's 32 output channels; verified TOGETHER with all other 31 kernels by the combined all-32-kernel testbench). REAL UAVSAR-tile-derived
-- 6x6x3 sub-block (see real_tile_stimulus_pkg.vhd), valid (no-padding)
-- 3x3 output region only -- same scope as every other VHDL prototype in
-- this repo. No pooling, no downstream layers, no full model pipeline.
-- Not board-tested, not a measured-speedup or measured-power claim.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_conv_bn_relu_kernel21_real_tile_q20_pkg.all;

entity stream_conv3x3_3chan_kernel21_bn_relu_real_tile_q20 is
    generic (
        IMG_WIDTH : positive := 6
    );
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;        -- synchronous, active high
        valid_in : in  std_logic;

        -- Three input-channel pixel streams (row-major, one triplet per clock)
        pixel_c0 : in signed(7 downto 0);
        pixel_c1 : in signed(7 downto 0);
        pixel_c2 : in signed(7 downto 0);

        -- Output: one Q.20 fixed-point result per valid window triplet,
        -- ReLU already applied, 6-cycle latency (identical to the
        -- toy-patch pipelined design).
        valid_out    : out std_logic;
        y_bn_relu_fx : out signed(47 downto 0)
    );
end entity stream_conv3x3_3chan_kernel21_bn_relu_real_tile_q20;

architecture rtl of stream_conv3x3_3chan_kernel21_bn_relu_real_tile_q20 is

    constant ZERO32 : signed(31 downto 0) := (others => '0');

    constant SCALE_FX_SIGNED : signed(17 downto 0) := to_signed(SCALE_FX, 18);
    constant BIAS_FX_SIGNED  : signed(47 downto 0) := to_signed(BIAS_FX, 48);

    signal raw_y     : signed(31 downto 0);
    signal raw_valid : std_logic;

    signal product_fx_r : signed(47 downto 0) := (others => '0');
    signal valid_s1_r    : std_logic           := '0';

    signal biased_fx_comb : signed(47 downto 0);
    signal y_bn_relu_fx_r : signed(47 downto 0) := (others => '0');
    signal valid_s2_r      : std_logic           := '0';

begin

    -- ================================================================
    -- Raw convolution: EXISTING, UNMODIFIED stream_conv3x3_3chan_cell,
    -- fed kernel 21's FOLDED INT8 weights (IDENTICAL values to the
    -- toy-patch package) with bias=0.
    -- ================================================================
    raw_conv : entity work.stream_conv3x3_3chan_cell
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk => clk,  rst => rst,  valid_in => valid_in,
            pixel_c0 => pixel_c0,  pixel_c1 => pixel_c1,  pixel_c2 => pixel_c2,
            c0_w0 => to_signed(K21_RT_FOLDED_CH0_W(0), 8),
            c0_w1 => to_signed(K21_RT_FOLDED_CH0_W(1), 8),
            c0_w2 => to_signed(K21_RT_FOLDED_CH0_W(2), 8),
            c0_w3 => to_signed(K21_RT_FOLDED_CH0_W(3), 8),
            c0_w4 => to_signed(K21_RT_FOLDED_CH0_W(4), 8),
            c0_w5 => to_signed(K21_RT_FOLDED_CH0_W(5), 8),
            c0_w6 => to_signed(K21_RT_FOLDED_CH0_W(6), 8),
            c0_w7 => to_signed(K21_RT_FOLDED_CH0_W(7), 8),
            c0_w8 => to_signed(K21_RT_FOLDED_CH0_W(8), 8),
            c1_w0 => to_signed(K21_RT_FOLDED_CH1_W(0), 8),
            c1_w1 => to_signed(K21_RT_FOLDED_CH1_W(1), 8),
            c1_w2 => to_signed(K21_RT_FOLDED_CH1_W(2), 8),
            c1_w3 => to_signed(K21_RT_FOLDED_CH1_W(3), 8),
            c1_w4 => to_signed(K21_RT_FOLDED_CH1_W(4), 8),
            c1_w5 => to_signed(K21_RT_FOLDED_CH1_W(5), 8),
            c1_w6 => to_signed(K21_RT_FOLDED_CH1_W(6), 8),
            c1_w7 => to_signed(K21_RT_FOLDED_CH1_W(7), 8),
            c1_w8 => to_signed(K21_RT_FOLDED_CH1_W(8), 8),
            c2_w0 => to_signed(K21_RT_FOLDED_CH2_W(0), 8),
            c2_w1 => to_signed(K21_RT_FOLDED_CH2_W(1), 8),
            c2_w2 => to_signed(K21_RT_FOLDED_CH2_W(2), 8),
            c2_w3 => to_signed(K21_RT_FOLDED_CH2_W(3), 8),
            c2_w4 => to_signed(K21_RT_FOLDED_CH2_W(4), 8),
            c2_w5 => to_signed(K21_RT_FOLDED_CH2_W(5), 8),
            c2_w6 => to_signed(K21_RT_FOLDED_CH2_W(6), 8),
            c2_w7 => to_signed(K21_RT_FOLDED_CH2_W(7), 8),
            c2_w8 => to_signed(K21_RT_FOLDED_CH2_W(8), 8),
            bias      => ZERO32,
            valid_out => raw_valid,
            y         => raw_y
        );

    -- ================================================================
    -- Stage 1 (registered): product_fx = raw_y * SCALE_FX (REAL-TILE
    -- value, from first_conv_bn_relu_kernel21_real_tile_q20_pkg).
    -- ================================================================
    stage1_multiply : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                product_fx_r <= (others => '0');
                valid_s1_r    <= '0';
            else
                product_fx_r <= resize(raw_y * SCALE_FX_SIGNED, 48);
                valid_s1_r    <= raw_valid;
            end if;
        end if;
    end process stage1_multiply;

    -- ================================================================
    -- Stage 2 (registered): biased_fx = product_fx_r + BIAS_FX, then ReLU.
    -- ================================================================
    biased_fx_comb <= product_fx_r + BIAS_FX_SIGNED;

    stage2_bias_relu : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                y_bn_relu_fx_r <= (others => '0');
                valid_s2_r      <= '0';
            else
                if biased_fx_comb > 0 then
                    y_bn_relu_fx_r <= biased_fx_comb;
                else
                    y_bn_relu_fx_r <= (others => '0');
                end if;
                valid_s2_r <= valid_s1_r;
            end if;
        end if;
    end process stage2_bias_relu;

    valid_out    <= valid_s2_r;
    y_bn_relu_fx <= y_bn_relu_fx_r;

end architecture rtl;
