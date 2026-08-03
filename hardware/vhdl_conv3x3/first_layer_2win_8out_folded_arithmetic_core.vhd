-- first_layer_2win_8out_folded_arithmetic_core.vhd
--
-- SPATIAL-PARALLELISM arithmetic-core artifact: accepts TWO independent,
-- ALREADY-FLATTENED 27-value signed 3x3x3 input windows PER CYCLE (one per
-- spatial "lane") and computes output channels 0-7 (of the first learned
-- Conv2d+BatchNorm2d+ReLU stage's 32 total channels) for EACH lane, in one
-- fully pipelined pass.
--
-- ---- Why this exists -------------------------------------------------------
-- first_layer_16out_folded_arithmetic_core.vhd (commit `fe79e096`) already
-- showed that computing 16 of the 32 output channels for ONE spatial
-- window per cycle leaves real DSP headroom (405/740, 45.27% free) on the
-- Artix-7 200T target. This module tests a DIFFERENT resource trade for
-- that same DSP budget: instead of one spatial window with 16 output
-- channels, compute TWO independent spatial windows per cycle, each with
-- only 8 output channels (2 x 8 = 16 total per-cycle output-channel
-- computations either way -- the SAME total MAC/cycle budget as the
-- 16-output design, just reshaped from "1 window x 16 channels" to
-- "2 windows x 8 channels"). The question: does the Artix-7 200T support
-- at least 2 spatial lanes for a real trained first-layer subproblem
-- without hitting the same 100%-DSP wall as the 32-output design?
--
-- This is an ARITHMETIC-CORE-ONLY test: both lanes consume ALREADY-
-- FLATTENED windows (the caller is assumed to have already extracted
-- both 3x3x3 windows -- e.g. from two different spatial positions in the
-- same image, or two different images). NO 2-pixel/cycle streaming front
-- end is built here -- that is explicitly future work (see Scope below).
--
-- ---- Architecture -----------------------------------------------------------
-- For each spatial lane L in {0, 1}, and each of the 8 output kernels k
-- (identical per-kernel structure and pipeline depth to the 16-output and
-- 32-output cores -- see those files' headers for the full per-stage
-- description):
--   3x conv3x3_dot_pipelined_dsp (EXISTING, UNMODIFIED) -- 3-cycle latency.
--   Stage A (4th cycle): raw_sum(L,k) = y_c0(L,k) + y_c1(L,k) + y_c2(L,k)
--   Stage B (5th cycle): product_fx(L,k) = raw_sum(L,k) * SCALE_FX_LUT(k)  (Q.20)
--   Stage C (6th cycle): biased_fx = product_fx(L,k) + BIAS_FX_LUT(k); ReLU
-- Both lanes' 8 kernels each run in this same 6-cycle pipeline, fully in
-- parallel: 2 lanes x 8 kernels x 3 channels = 48 conv3x3_dot_pipelined_dsp
-- instances total -- the SAME total instance count as the 16-output
-- single-lane core's 48 (16 kernels x 3 channels), just reorganized into
-- 2 lanes of 8 kernels each. Both lanes share clk/rst/valid_in (they are
-- always presented and consumed in lockstep) -- a new PAIR of windows may
-- be presented on every clock cycle, giving a THEORETICAL 2 windows/cycle
-- steady-state throughput once this core is fed by a matching front end
-- (not built here).
--
-- ---- Latency: 6 cycles total (valid_in to valid_out), same as the
-- 16-output and 32-output single-lane cores. -------------------------------
--
-- ---- Scope ------------------------------------------------------------------
-- This is an ARITHMETIC-CORE ARTIFACT ONLY:
--   - No image streaming, no line buffers, no sliding-window generation,
--     no 2-pixel/cycle streaming front end (the caller is assumed to have
--     already extracted BOTH 27-value windows). A matching streaming front
--     end is explicitly FUTURE WORK, not part of this task.
--   - No padding.
--   - Only 8 of the first Conv2d layer's 32 output channels, PER LANE --
--     NOT the full first layer, NOT a full U-Net, NOT board-tested, no
--     measured hardware speedup or power claim.
--   - Does NOT claim FPGA is faster than GPU.
-- Synthesis results (LUTs/registers/DSPs/BRAM/WNS/power) are reported
-- separately in hardware/vhdl_conv3x3/reports/
-- first_layer_2win_8out_folded_arithmetic_core_summary.md, from an actual
-- Vivado run -- not estimated here.
--
-- ---- Existing 16-output and 32-output artifacts are NOT modified ----------
-- This is a new, independent file. first_layer_16out_folded_arithmetic_core.vhd
-- and first_layer_32out_folded_arithmetic_core.vhd (and their packages) are
-- untouched.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_8out_folded_bn_relu_real_tile_q20_pkg.all;

entity first_layer_2win_8out_folded_arithmetic_core is
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;        -- synchronous, active high
        valid_in : in  std_logic;        -- one PAIR of flattened windows presented this cycle

        -- Lane 0: one already-flattened, already-captured 3x3x3 window: 9
        -- pixels per input channel, row-major (p0=top-left .. p8=bottom-right).
        w0_c0_p0 : in signed(7 downto 0);  w0_c0_p1 : in signed(7 downto 0);  w0_c0_p2 : in signed(7 downto 0);
        w0_c0_p3 : in signed(7 downto 0);  w0_c0_p4 : in signed(7 downto 0);  w0_c0_p5 : in signed(7 downto 0);
        w0_c0_p6 : in signed(7 downto 0);  w0_c0_p7 : in signed(7 downto 0);  w0_c0_p8 : in signed(7 downto 0);

        w0_c1_p0 : in signed(7 downto 0);  w0_c1_p1 : in signed(7 downto 0);  w0_c1_p2 : in signed(7 downto 0);
        w0_c1_p3 : in signed(7 downto 0);  w0_c1_p4 : in signed(7 downto 0);  w0_c1_p5 : in signed(7 downto 0);
        w0_c1_p6 : in signed(7 downto 0);  w0_c1_p7 : in signed(7 downto 0);  w0_c1_p8 : in signed(7 downto 0);

        w0_c2_p0 : in signed(7 downto 0);  w0_c2_p1 : in signed(7 downto 0);  w0_c2_p2 : in signed(7 downto 0);
        w0_c2_p3 : in signed(7 downto 0);  w0_c2_p4 : in signed(7 downto 0);  w0_c2_p5 : in signed(7 downto 0);
        w0_c2_p6 : in signed(7 downto 0);  w0_c2_p7 : in signed(7 downto 0);  w0_c2_p8 : in signed(7 downto 0);

        -- Lane 1: a SECOND, INDEPENDENT already-flattened 3x3x3 window.
        w1_c0_p0 : in signed(7 downto 0);  w1_c0_p1 : in signed(7 downto 0);  w1_c0_p2 : in signed(7 downto 0);
        w1_c0_p3 : in signed(7 downto 0);  w1_c0_p4 : in signed(7 downto 0);  w1_c0_p5 : in signed(7 downto 0);
        w1_c0_p6 : in signed(7 downto 0);  w1_c0_p7 : in signed(7 downto 0);  w1_c0_p8 : in signed(7 downto 0);

        w1_c1_p0 : in signed(7 downto 0);  w1_c1_p1 : in signed(7 downto 0);  w1_c1_p2 : in signed(7 downto 0);
        w1_c1_p3 : in signed(7 downto 0);  w1_c1_p4 : in signed(7 downto 0);  w1_c1_p5 : in signed(7 downto 0);
        w1_c1_p6 : in signed(7 downto 0);  w1_c1_p7 : in signed(7 downto 0);  w1_c1_p8 : in signed(7 downto 0);

        w1_c2_p0 : in signed(7 downto 0);  w1_c2_p1 : in signed(7 downto 0);  w1_c2_p2 : in signed(7 downto 0);
        w1_c2_p3 : in signed(7 downto 0);  w1_c2_p4 : in signed(7 downto 0);  w1_c2_p5 : in signed(7 downto 0);
        w1_c2_p6 : in signed(7 downto 0);  w1_c2_p7 : in signed(7 downto 0);  w1_c2_p8 : in signed(7 downto 0);

        -- Output: 8 Q.20 fixed-point folded Conv-BN-ReLU results (output
        -- channels 0-7 of 32) PER LANE -- 16 outputs total per valid
        -- cycle, ReLU already applied, 6-cycle latency.
        valid_out : out std_logic;
        y0_0 : out signed(47 downto 0);  y0_1 : out signed(47 downto 0);
        y0_2 : out signed(47 downto 0);  y0_3 : out signed(47 downto 0);
        y0_4 : out signed(47 downto 0);  y0_5 : out signed(47 downto 0);
        y0_6 : out signed(47 downto 0);  y0_7 : out signed(47 downto 0);

        y1_0 : out signed(47 downto 0);  y1_1 : out signed(47 downto 0);
        y1_2 : out signed(47 downto 0);  y1_3 : out signed(47 downto 0);
        y1_4 : out signed(47 downto 0);  y1_5 : out signed(47 downto 0);
        y1_6 : out signed(47 downto 0);  y1_7 : out signed(47 downto 0)
    );
end entity first_layer_2win_8out_folded_arithmetic_core;

architecture rtl of first_layer_2win_8out_folded_arithmetic_core is

    constant ZERO32    : signed(31 downto 0) := (others => '0');
    constant NUM_LANES : integer := 2;

    -- Per-lane, per-channel 9-pixel window vectors, assembled from the
    -- named input ports so the kernel/channel dot-product generate loop
    -- below can index by (lane, channel) uniformly.
    type pixel9_t         is array (0 to 8) of signed(7 downto 0);
    type lane_channels_t  is array (0 to 2) of pixel9_t;               -- indexed by input channel 0-2
    type all_lanes_pix_t  is array (0 to NUM_LANES - 1) of lane_channels_t;

    signal lane_pixels : all_lanes_pix_t;

    type y_int32_array_t is array (0 to NUM_KERNELS - 1) of signed(31 downto 0);
    type y_fx_array_t    is array (0 to NUM_KERNELS - 1) of signed(47 downto 0);
    type lane_int32_t    is array (0 to NUM_LANES - 1) of y_int32_array_t;
    type lane_fx_t       is array (0 to NUM_LANES - 1) of y_fx_array_t;

    -- Per-lane, per-channel, per-kernel raw dot-product outputs (3 cycles
    -- latency, from the existing conv3x3_dot_pipelined_dsp component).
    signal y_c0_dot : lane_int32_t;
    signal y_c1_dot : lane_int32_t;
    signal y_c2_dot : lane_int32_t;

    -- dot_c0's valid_out, per lane and kernel -- both lanes share timing
    -- (same valid_in), so only lane 0/kernel 0's value is actually read
    -- downstream; all instances are wired unconditionally (VHDL port maps
    -- cannot use conditional "when/else" actuals).
    type lane_valid_t is array (0 to NUM_LANES - 1) of std_logic_vector(0 to NUM_KERNELS - 1);
    signal dot_valid : lane_valid_t;

    -- Stage A: registered final sum (y_c0 + y_c1 + y_c2), 4th cycle.
    signal raw_sum_r   : lane_int32_t := (others => (others => (others => '0')));
    signal raw_valid_r : std_logic := '0';

    -- Stage B: registered Q.20 scale multiply, 5th cycle.
    signal product_fx_r : lane_fx_t := (others => (others => (others => '0')));
    signal valid_s1_r    : std_logic := '0';

    -- Stage C: registered bias-add + ReLU, 6th cycle.
    signal y_bn_relu_fx_r : lane_fx_t := (others => (others => (others => '0')));
    signal valid_s2_r      : std_logic := '0';

begin

    -- ================================================================
    -- Assemble each lane's 3 named-port 9-pixel channel vectors into the
    -- lane_pixels array (pure wiring, no arithmetic).
    -- ================================================================
    lane_pixels(0)(0) <= (w0_c0_p0, w0_c0_p1, w0_c0_p2, w0_c0_p3, w0_c0_p4, w0_c0_p5, w0_c0_p6, w0_c0_p7, w0_c0_p8);
    lane_pixels(0)(1) <= (w0_c1_p0, w0_c1_p1, w0_c1_p2, w0_c1_p3, w0_c1_p4, w0_c1_p5, w0_c1_p6, w0_c1_p7, w0_c1_p8);
    lane_pixels(0)(2) <= (w0_c2_p0, w0_c2_p1, w0_c2_p2, w0_c2_p3, w0_c2_p4, w0_c2_p5, w0_c2_p6, w0_c2_p7, w0_c2_p8);

    lane_pixels(1)(0) <= (w1_c0_p0, w1_c0_p1, w1_c0_p2, w1_c0_p3, w1_c0_p4, w1_c0_p5, w1_c0_p6, w1_c0_p7, w1_c0_p8);
    lane_pixels(1)(1) <= (w1_c1_p0, w1_c1_p1, w1_c1_p2, w1_c1_p3, w1_c1_p4, w1_c1_p5, w1_c1_p6, w1_c1_p7, w1_c1_p8);
    lane_pixels(1)(2) <= (w1_c2_p0, w1_c2_p1, w1_c2_p2, w1_c2_p3, w1_c2_p4, w1_c2_p5, w1_c2_p6, w1_c2_p7, w1_c2_p8);

    -- ================================================================
    -- 2 lanes x 8 kernels x 3 channels = 48 parallel raw INT8 dot
    -- products (EXISTING, UNMODIFIED conv3x3_dot_pipelined_dsp component
    -- -- the SAME total instance count as the 16-output single-lane
    -- core's 48, just reorganized into 2 lanes of 8 kernels).
    -- ================================================================
    gen_lanes : for lane in 0 to NUM_LANES - 1 generate
    begin
        gen_kernels : for k in 0 to NUM_KERNELS - 1 generate
        begin

            dot_c0 : entity work.conv3x3_dot_pipelined_dsp
                port map (
                    clk => clk, rst => rst, valid_in => valid_in,
                    p0 => lane_pixels(lane)(0)(0), p1 => lane_pixels(lane)(0)(1), p2 => lane_pixels(lane)(0)(2),
                    p3 => lane_pixels(lane)(0)(3), p4 => lane_pixels(lane)(0)(4), p5 => lane_pixels(lane)(0)(5),
                    p6 => lane_pixels(lane)(0)(6), p7 => lane_pixels(lane)(0)(7), p8 => lane_pixels(lane)(0)(8),
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
                    valid_out => dot_valid(lane)(k),
                    y         => y_c0_dot(lane)(k)
                );

            dot_c1 : entity work.conv3x3_dot_pipelined_dsp
                port map (
                    clk => clk, rst => rst, valid_in => valid_in,
                    p0 => lane_pixels(lane)(1)(0), p1 => lane_pixels(lane)(1)(1), p2 => lane_pixels(lane)(1)(2),
                    p3 => lane_pixels(lane)(1)(3), p4 => lane_pixels(lane)(1)(4), p5 => lane_pixels(lane)(1)(5),
                    p6 => lane_pixels(lane)(1)(6), p7 => lane_pixels(lane)(1)(7), p8 => lane_pixels(lane)(1)(8),
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
                    y         => y_c1_dot(lane)(k)
                );

            dot_c2 : entity work.conv3x3_dot_pipelined_dsp
                port map (
                    clk => clk, rst => rst, valid_in => valid_in,
                    p0 => lane_pixels(lane)(2)(0), p1 => lane_pixels(lane)(2)(1), p2 => lane_pixels(lane)(2)(2),
                    p3 => lane_pixels(lane)(2)(3), p4 => lane_pixels(lane)(2)(4), p5 => lane_pixels(lane)(2)(5),
                    p6 => lane_pixels(lane)(2)(6), p7 => lane_pixels(lane)(2)(7), p8 => lane_pixels(lane)(2)(8),
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
                    y         => y_c2_dot(lane)(k)
                );

        end generate gen_kernels;
    end generate gen_lanes;

    -- ================================================================
    -- Stages A/B/C: registered final-sum, Q.20 BN-fold multiply, and
    -- bias-add+ReLU -- for both lanes' 8 kernels each, in parallel. A
    -- static (locally-static-bound) "for lane"/"for k" nested loop inside
    -- a single clocked process describes 2 x 8 = 16 independent register
    -- slices; it is NOT a shared/looped hardware resource.
    -- ================================================================
    pipeline_tail : process(clk)
        variable biased_fx_v : signed(47 downto 0);
    begin
        if rising_edge(clk) then
            if rst = '1' then
                raw_valid_r <= '0';
                valid_s1_r   <= '0';
                valid_s2_r   <= '0';
                for lane in 0 to NUM_LANES - 1 loop
                    for k in 0 to NUM_KERNELS - 1 loop
                        raw_sum_r(lane)(k)      <= (others => '0');
                        product_fx_r(lane)(k)   <= (others => '0');
                        y_bn_relu_fx_r(lane)(k) <= (others => '0');
                    end loop;
                end loop;
            else
                -- Stage A (4th cycle): raw_sum(lane,k) = y_c0 + y_c1 + y_c2
                -- (bias = 0, matching the first Conv2d layer -- no direct bias).
                for lane in 0 to NUM_LANES - 1 loop
                    for k in 0 to NUM_KERNELS - 1 loop
                        raw_sum_r(lane)(k) <= y_c0_dot(lane)(k) + y_c1_dot(lane)(k) + y_c2_dot(lane)(k);
                    end loop;
                end loop;
                raw_valid_r <= dot_valid(0)(0);

                -- Stage B (5th cycle): product_fx(lane,k) = raw_sum(lane,k) * SCALE_FX_LUT(k)
                for lane in 0 to NUM_LANES - 1 loop
                    for k in 0 to NUM_KERNELS - 1 loop
                        product_fx_r(lane)(k) <= resize(raw_sum_r(lane)(k) * to_signed(SCALE_FX_LUT(k), 18), 48);
                    end loop;
                end loop;
                valid_s1_r <= raw_valid_r;

                -- Stage C (6th cycle): biased_fx(lane,k) = product_fx(lane,k) + BIAS_FX_LUT(k); ReLU
                for lane in 0 to NUM_LANES - 1 loop
                    for k in 0 to NUM_KERNELS - 1 loop
                        biased_fx_v := product_fx_r(lane)(k) + to_signed(BIAS_FX_LUT(k), 48);
                        if biased_fx_v > 0 then
                            y_bn_relu_fx_r(lane)(k) <= biased_fx_v;
                        else
                            y_bn_relu_fx_r(lane)(k) <= (others => '0');
                        end if;
                    end loop;
                end loop;
                valid_s2_r <= valid_s1_r;
            end if;
        end if;
    end process pipeline_tail;

    valid_out <= valid_s2_r;

    y0_0 <= y_bn_relu_fx_r(0)(0);  y0_1 <= y_bn_relu_fx_r(0)(1);  y0_2 <= y_bn_relu_fx_r(0)(2);  y0_3 <= y_bn_relu_fx_r(0)(3);
    y0_4 <= y_bn_relu_fx_r(0)(4);  y0_5 <= y_bn_relu_fx_r(0)(5);  y0_6 <= y_bn_relu_fx_r(0)(6);  y0_7 <= y_bn_relu_fx_r(0)(7);

    y1_0 <= y_bn_relu_fx_r(1)(0);  y1_1 <= y_bn_relu_fx_r(1)(1);  y1_2 <= y_bn_relu_fx_r(1)(2);  y1_3 <= y_bn_relu_fx_r(1)(3);
    y1_4 <= y_bn_relu_fx_r(1)(4);  y1_5 <= y_bn_relu_fx_r(1)(5);  y1_6 <= y_bn_relu_fx_r(1)(6);  y1_7 <= y_bn_relu_fx_r(1)(7);

end architecture rtl;
