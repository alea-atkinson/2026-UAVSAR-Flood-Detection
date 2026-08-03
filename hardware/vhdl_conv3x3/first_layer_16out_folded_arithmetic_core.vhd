-- first_layer_16out_folded_arithmetic_core.vhd
--
-- Single-shot (NOT streamed) arithmetic-core artifact: accepts one
-- ALREADY-FLATTENED 27-value signed 3x3x3 input window per cycle and
-- computes output channels 0-15 (of the first learned Conv2d+BatchNorm2d+
-- ReLU stage's 32 total channels) in one fully pipelined pass.
--
-- ---- Why this exists -------------------------------------------------------
-- hardware/vhdl_conv3x3/first_layer_32out_folded_arithmetic_core.vhd (commit
-- 313558e4) already showed that ALL 32 output channels, fully parallel,
-- saturate this repo's Artix-7 200T target at exactly 740/740 (100%) DSPs
-- -- zero headroom left on that part. This module is a SMALLER, comfortably
-- fitting subproblem built to answer a different question: computing only
-- the first 16 of those 32 output channels, does the design stay fast, meet
-- 100 MHz timing, and leave real DSP headroom on the SAME part? It is built
-- ENTIRELY from the same existing, unmodified sub-components as the 32-output
-- core (see first_layer_16out_folded_bn_relu_real_tile_q20_pkg.vhd), just
-- looped over 16 kernels instead of 32. The 32-output core is NOT modified
-- by this artifact.
--
-- ---- Architecture -----------------------------------------------------------
-- For each of the 16 output kernels k (identical structure and pipeline
-- depth to the 32-output core -- see that file's header for the full
-- per-stage description):
--   3x conv3x3_dot_pipelined_dsp (EXISTING, UNMODIFIED) -- 3-cycle latency.
--   Stage A (4th cycle): raw_sum(k) = y_c0(k) + y_c1(k) + y_c2(k)
--   Stage B (5th cycle): product_fx(k) = raw_sum(k) * SCALE_FX_LUT(k)  (Q.20)
--   Stage C (6th cycle): biased_fx = product_fx(k) + BIAS_FX_LUT(k); ReLU
-- All 16 kernels run in this same 6-cycle pipeline, fully in parallel (48
-- conv3x3_dot_pipelined_dsp instances total -- HALF of the 32-output core's
-- 96). A new window may be presented on every clock cycle -- 1 window/cycle
-- steady-state throughput.
--
-- ---- Latency: 6 cycles total (valid_in to valid_out), same as the 32-output
-- core. -----------------------------------------------------------------------
--
-- ---- Scope ------------------------------------------------------------------
-- This is an ARITHMETIC-CORE ARTIFACT ONLY:
--   - No image streaming, no line buffers, no sliding-window generation
--     (the caller is assumed to have already extracted the 27-value window;
--     see first_layer_16out_folded_arithmetic_core_streaming_front_end.vhd
--     for the integrated streaming wrapper).
--   - No padding.
--   - Only 16 of the first Conv2d layer's 32 output channels -- NOT the
--     full first layer, NOT a full U-Net, NOT board-tested, no measured
--     hardware speedup or power claim.
-- Synthesis results (LUTs/registers/DSPs/BRAM/WNS/power) are reported
-- separately in hardware/vhdl_conv3x3/reports/
-- first_layer_16out_folded_arithmetic_core_streaming_front_end_summary.md,
-- from an actual Vivado run -- not estimated here.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_16out_folded_bn_relu_real_tile_q20_pkg.all;

entity first_layer_16out_folded_arithmetic_core is
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;        -- synchronous, active high
        valid_in : in  std_logic;        -- one flattened window presented this cycle

        -- One already-flattened, already-captured 3x3x3 window: 9 pixels
        -- per input channel, row-major (p0=top-left .. p8=bottom-right).
        c0_p0 : in signed(7 downto 0);  c0_p1 : in signed(7 downto 0);  c0_p2 : in signed(7 downto 0);
        c0_p3 : in signed(7 downto 0);  c0_p4 : in signed(7 downto 0);  c0_p5 : in signed(7 downto 0);
        c0_p6 : in signed(7 downto 0);  c0_p7 : in signed(7 downto 0);  c0_p8 : in signed(7 downto 0);

        c1_p0 : in signed(7 downto 0);  c1_p1 : in signed(7 downto 0);  c1_p2 : in signed(7 downto 0);
        c1_p3 : in signed(7 downto 0);  c1_p4 : in signed(7 downto 0);  c1_p5 : in signed(7 downto 0);
        c1_p6 : in signed(7 downto 0);  c1_p7 : in signed(7 downto 0);  c1_p8 : in signed(7 downto 0);

        c2_p0 : in signed(7 downto 0);  c2_p1 : in signed(7 downto 0);  c2_p2 : in signed(7 downto 0);
        c2_p3 : in signed(7 downto 0);  c2_p4 : in signed(7 downto 0);  c2_p5 : in signed(7 downto 0);
        c2_p6 : in signed(7 downto 0);  c2_p7 : in signed(7 downto 0);  c2_p8 : in signed(7 downto 0);

        -- Output: 16 Q.20 fixed-point folded Conv-BN-ReLU results (output
        -- channels 0-15 of 32), ReLU already applied, 6-cycle latency,
        -- 1 window/cycle throughput.
        valid_out : out std_logic;
        y0  : out signed(47 downto 0);  y1  : out signed(47 downto 0);
        y2  : out signed(47 downto 0);  y3  : out signed(47 downto 0);
        y4  : out signed(47 downto 0);  y5  : out signed(47 downto 0);
        y6  : out signed(47 downto 0);  y7  : out signed(47 downto 0);
        y8  : out signed(47 downto 0);  y9  : out signed(47 downto 0);
        y10 : out signed(47 downto 0);  y11 : out signed(47 downto 0);
        y12 : out signed(47 downto 0);  y13 : out signed(47 downto 0);
        y14 : out signed(47 downto 0);  y15 : out signed(47 downto 0)
    );
end entity first_layer_16out_folded_arithmetic_core;

architecture rtl of first_layer_16out_folded_arithmetic_core is

    constant ZERO32 : signed(31 downto 0) := (others => '0');

    type y16_int32_array_t is array (0 to NUM_KERNELS - 1) of signed(31 downto 0);
    type y16_fx_array_t    is array (0 to NUM_KERNELS - 1) of signed(47 downto 0);
    type y16_out_array_t   is array (0 to NUM_KERNELS - 1) of signed(47 downto 0);

    -- Per-channel, per-kernel raw dot-product outputs (3 cycles latency,
    -- from the existing conv3x3_dot_pipelined_dsp component).
    signal y_c0_dot : y16_int32_array_t;
    signal y_c1_dot : y16_int32_array_t;
    signal y_c2_dot : y16_int32_array_t;
    signal dot_valid : std_logic_vector(0 to NUM_KERNELS - 1);

    -- Stage A: registered final sum (bias + y_c0 + y_c1 + y_c2), 4th cycle.
    signal raw_sum_r   : y16_int32_array_t := (others => (others => '0'));
    signal raw_valid_r : std_logic := '0';

    -- Stage B: registered Q.20 scale multiply, 5th cycle.
    signal product_fx_r : y16_fx_array_t := (others => (others => '0'));
    signal valid_s1_r    : std_logic := '0';

    -- Stage C: registered bias-add + ReLU, 6th cycle.
    signal y_bn_relu_fx_r : y16_out_array_t := (others => (others => '0'));
    signal valid_s2_r      : std_logic := '0';

begin

    -- ================================================================
    -- 16 x 3 = 48 parallel raw INT8 dot products (EXISTING, UNMODIFIED
    -- conv3x3_dot_pipelined_dsp component -- same IP used by the
    -- 32-output folded core, HALF as many instances).
    -- ================================================================
    gen_kernels : for k in 0 to NUM_KERNELS - 1 generate
    begin

        dot_c0 : entity work.conv3x3_dot_pipelined_dsp
            port map (
                clk => clk, rst => rst, valid_in => valid_in,
                p0 => c0_p0, p1 => c0_p1, p2 => c0_p2,
                p3 => c0_p3, p4 => c0_p4, p5 => c0_p5,
                p6 => c0_p6, p7 => c0_p7, p8 => c0_p8,
                w0 => to_signed(CH0_WEIGHT_LUT(k)(0), 8),
                w1 => to_signed(CH0_WEIGHT_LUT(k)(1), 8),
                w2 => to_signed(CH0_WEIGHT_LUT(k)(2), 8),
                w3 => to_signed(CH0_WEIGHT_LUT(k)(3), 8),
                w4 => to_signed(CH0_WEIGHT_LUT(k)(4), 8),
                w5 => to_signed(CH0_WEIGHT_LUT(k)(5), 8),
                w6 => to_signed(CH0_WEIGHT_LUT(k)(6), 8),
                w7 => to_signed(CH0_WEIGHT_LUT(k)(7), 8),
                w8 => to_signed(CH0_WEIGHT_LUT(k)(8), 8),
                bias      => ZERO32,
                valid_out => dot_valid(k),
                y         => y_c0_dot(k)
            );

        dot_c1 : entity work.conv3x3_dot_pipelined_dsp
            port map (
                clk => clk, rst => rst, valid_in => valid_in,
                p0 => c1_p0, p1 => c1_p1, p2 => c1_p2,
                p3 => c1_p3, p4 => c1_p4, p5 => c1_p5,
                p6 => c1_p6, p7 => c1_p7, p8 => c1_p8,
                w0 => to_signed(CH1_WEIGHT_LUT(k)(0), 8),
                w1 => to_signed(CH1_WEIGHT_LUT(k)(1), 8),
                w2 => to_signed(CH1_WEIGHT_LUT(k)(2), 8),
                w3 => to_signed(CH1_WEIGHT_LUT(k)(3), 8),
                w4 => to_signed(CH1_WEIGHT_LUT(k)(4), 8),
                w5 => to_signed(CH1_WEIGHT_LUT(k)(5), 8),
                w6 => to_signed(CH1_WEIGHT_LUT(k)(6), 8),
                w7 => to_signed(CH1_WEIGHT_LUT(k)(7), 8),
                w8 => to_signed(CH1_WEIGHT_LUT(k)(8), 8),
                bias      => ZERO32,
                valid_out => open,
                y         => y_c1_dot(k)
            );

        dot_c2 : entity work.conv3x3_dot_pipelined_dsp
            port map (
                clk => clk, rst => rst, valid_in => valid_in,
                p0 => c2_p0, p1 => c2_p1, p2 => c2_p2,
                p3 => c2_p3, p4 => c2_p4, p5 => c2_p5,
                p6 => c2_p6, p7 => c2_p7, p8 => c2_p8,
                w0 => to_signed(CH2_WEIGHT_LUT(k)(0), 8),
                w1 => to_signed(CH2_WEIGHT_LUT(k)(1), 8),
                w2 => to_signed(CH2_WEIGHT_LUT(k)(2), 8),
                w3 => to_signed(CH2_WEIGHT_LUT(k)(3), 8),
                w4 => to_signed(CH2_WEIGHT_LUT(k)(4), 8),
                w5 => to_signed(CH2_WEIGHT_LUT(k)(5), 8),
                w6 => to_signed(CH2_WEIGHT_LUT(k)(6), 8),
                w7 => to_signed(CH2_WEIGHT_LUT(k)(7), 8),
                w8 => to_signed(CH2_WEIGHT_LUT(k)(8), 8),
                bias      => ZERO32,
                valid_out => open,
                y         => y_c2_dot(k)
            );

    end generate gen_kernels;

    -- ================================================================
    -- Stages A/B/C: registered final-sum, Q.20 BN-fold multiply, and
    -- bias-add+ReLU -- for all 16 kernels in parallel. A static
    -- (locally-static-bound) "for k" loop inside a single clocked
    -- process describes 16 independent register slices; it is NOT a
    -- shared/looped hardware resource.
    -- ================================================================
    pipeline_tail : process(clk)
        variable biased_fx_v : signed(47 downto 0);
    begin
        if rising_edge(clk) then
            if rst = '1' then
                raw_valid_r <= '0';
                valid_s1_r   <= '0';
                valid_s2_r   <= '0';
                for k in 0 to NUM_KERNELS - 1 loop
                    raw_sum_r(k)      <= (others => '0');
                    product_fx_r(k)   <= (others => '0');
                    y_bn_relu_fx_r(k) <= (others => '0');
                end loop;
            else
                -- Stage A (4th cycle): raw_sum(k) = y_c0(k) + y_c1(k) + y_c2(k)
                -- (bias = 0, matching the first Conv2d layer -- no direct bias).
                for k in 0 to NUM_KERNELS - 1 loop
                    raw_sum_r(k) <= y_c0_dot(k) + y_c1_dot(k) + y_c2_dot(k);
                end loop;
                raw_valid_r <= dot_valid(0);

                -- Stage B (5th cycle): product_fx(k) = raw_sum(k) * SCALE_FX_LUT(k)
                for k in 0 to NUM_KERNELS - 1 loop
                    product_fx_r(k) <= resize(raw_sum_r(k) * to_signed(SCALE_FX_LUT(k), 18), 48);
                end loop;
                valid_s1_r <= raw_valid_r;

                -- Stage C (6th cycle): biased_fx(k) = product_fx(k) + BIAS_FX_LUT(k); ReLU
                for k in 0 to NUM_KERNELS - 1 loop
                    biased_fx_v := product_fx_r(k) + to_signed(BIAS_FX_LUT(k), 48);
                    if biased_fx_v > 0 then
                        y_bn_relu_fx_r(k) <= biased_fx_v;
                    else
                        y_bn_relu_fx_r(k) <= (others => '0');
                    end if;
                end loop;
                valid_s2_r <= valid_s1_r;
            end if;
        end if;
    end process pipeline_tail;

    valid_out <= valid_s2_r;

    y0  <= y_bn_relu_fx_r(0);   y1  <= y_bn_relu_fx_r(1);   y2  <= y_bn_relu_fx_r(2);
    y3  <= y_bn_relu_fx_r(3);   y4  <= y_bn_relu_fx_r(4);   y5  <= y_bn_relu_fx_r(5);
    y6  <= y_bn_relu_fx_r(6);   y7  <= y_bn_relu_fx_r(7);   y8  <= y_bn_relu_fx_r(8);
    y9  <= y_bn_relu_fx_r(9);   y10 <= y_bn_relu_fx_r(10);  y11 <= y_bn_relu_fx_r(11);
    y12 <= y_bn_relu_fx_r(12);  y13 <= y_bn_relu_fx_r(13);  y14 <= y_bn_relu_fx_r(14);
    y15 <= y_bn_relu_fx_r(15);

end architecture rtl;
