-- first_layer_32out_folded_arithmetic_core.vhd
--
-- Single-shot (NOT streamed) arithmetic-core prototype: accepts one
-- ALREADY-FLATTENED 27-value signed 3x3x3 input window per cycle and
-- computes all 32 folded Conv-BN-ReLU output channels of the first
-- learned Conv2d+BatchNorm2d+ReLU stage in one fully pipelined pass.
--
-- ---- Why this exists -------------------------------------------------------
-- scripts/benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py estimates a
-- speed-oriented Artix-7 200T FPGA core with these assumptions: one
-- pre-extracted, flattened 27-value window consumed per cycle; all 32
-- folded Conv-BN-ReLU output channels produced per cycle; a 6-cycle
-- pipeline fill (4 cycles raw conv + 2 cycles BN-fold/ReLU); 100 MHz
-- primary target. That benchmark's FPGA numbers were an ESTIMATE, not a
-- synthesizable design. This module is the concrete hardware artifact that
-- tests whether those assumptions actually hold: it is a real,
-- synthesizable VHDL-2008 design implementing exactly that interface and
-- exactly that pipeline depth, built ENTIRELY from existing, unmodified
-- sub-components and existing, already-verified weight/scale/bias
-- constants (see first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd).
--
-- ---- Architecture -----------------------------------------------------------
-- For each of the 32 output kernels k:
--   3x conv3x3_dot_pipelined_dsp (EXISTING, UNMODIFIED, DSP-steered 3x3
--     dot-product IP -- the SAME component used by the already-synthesized
--     740/740-DSP 32-output direct-parallel design) computes the raw INT8
--     dot product for each of the 3 input channels against kernel k's raw
--     INT8 weights, bias=0.  Latency: 3 cycles.
--   Stage A (registered, 4th cycle): raw_sum(k) <= y_c0(k) + y_c1(k) + y_c2(k)
--     (mirrors stream_conv3x3_3chan_cell's own final-summation stage).
--   Stage B (registered, 5th cycle): product_fx(k) <= raw_sum(k) * SCALE_FX_LUT(k)
--     (Q.20, exact -- mirrors the existing kernel0/kernel2 pipelined
--     BN-fold-multiply stage).
--   Stage C (registered, 6th cycle): biased_fx = product_fx(k) + BIAS_FX_LUT(k);
--     y(k) <= biased_fx if biased_fx > 0 else 0  (ReLU -- mirrors the
--     existing bias-add+ReLU stage).
-- All 32 kernels run in this same 6-cycle pipeline, fully in parallel (96
-- conv3x3_dot_pipelined_dsp instances total). Because there is no
-- streaming/windowing logic (the caller presents an already-flattened
-- window every cycle), a NEW window may be presented on every clock cycle
-- -- 1 window/cycle steady-state throughput, matching the benchmark's
-- assumption exactly.
--
-- ---- Latency: 6 cycles total (valid_in to valid_out), matching the
-- benchmark's FPGA_PIPELINE_FILL_CYCLES = 6 exactly. -----------------------
--
-- ---- Scope ------------------------------------------------------------------
-- This is an ARITHMETIC-CORE PROTOTYPE ONLY:
--   - No image streaming, no line buffers, no sliding-window generation
--     (the caller is assumed to have already extracted the 27-value window).
--   - No padding.
--   - No full U-Net (this is one Conv2d+BatchNorm+ReLU stage only).
--   - Not board-tested; no measured hardware speedup or power claim.
-- Synthesis results (LUTs/registers/DSPs/BRAM/WNS/power) are reported
-- separately in hardware/vhdl_conv3x3/reports/
-- first_layer_32out_folded_arithmetic_core_summary.md, from an actual
-- Vivado run -- not estimated here.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_32out_folded_bn_relu_real_tile_q20_pkg.all;

entity first_layer_32out_folded_arithmetic_core is
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

        -- Output: all 32 Q.20 fixed-point folded Conv-BN-ReLU results,
        -- ReLU already applied, 6-cycle latency, 1 window/cycle throughput.
        valid_out : out std_logic;
        y0  : out signed(47 downto 0);  y1  : out signed(47 downto 0);
        y2  : out signed(47 downto 0);  y3  : out signed(47 downto 0);
        y4  : out signed(47 downto 0);  y5  : out signed(47 downto 0);
        y6  : out signed(47 downto 0);  y7  : out signed(47 downto 0);
        y8  : out signed(47 downto 0);  y9  : out signed(47 downto 0);
        y10 : out signed(47 downto 0);  y11 : out signed(47 downto 0);
        y12 : out signed(47 downto 0);  y13 : out signed(47 downto 0);
        y14 : out signed(47 downto 0);  y15 : out signed(47 downto 0);
        y16 : out signed(47 downto 0);  y17 : out signed(47 downto 0);
        y18 : out signed(47 downto 0);  y19 : out signed(47 downto 0);
        y20 : out signed(47 downto 0);  y21 : out signed(47 downto 0);
        y22 : out signed(47 downto 0);  y23 : out signed(47 downto 0);
        y24 : out signed(47 downto 0);  y25 : out signed(47 downto 0);
        y26 : out signed(47 downto 0);  y27 : out signed(47 downto 0);
        y28 : out signed(47 downto 0);  y29 : out signed(47 downto 0);
        y30 : out signed(47 downto 0);  y31 : out signed(47 downto 0)
    );
end entity first_layer_32out_folded_arithmetic_core;

architecture rtl of first_layer_32out_folded_arithmetic_core is

    constant ZERO32 : signed(31 downto 0) := (others => '0');

    type y32_int32_array_t is array (0 to NUM_KERNELS - 1) of signed(31 downto 0);
    type y32_fx_array_t    is array (0 to NUM_KERNELS - 1) of signed(47 downto 0);
    type y32_out_array_t   is array (0 to NUM_KERNELS - 1) of signed(47 downto 0);

    -- Per-channel, per-kernel raw dot-product outputs (3 cycles latency,
    -- from the existing conv3x3_dot_pipelined_dsp component).
    signal y_c0_dot : y32_int32_array_t;
    signal y_c1_dot : y32_int32_array_t;
    signal y_c2_dot : y32_int32_array_t;
    signal dot_valid : std_logic_vector(0 to NUM_KERNELS - 1);

    -- Stage A: registered final sum (bias + y_c0 + y_c1 + y_c2), 4th cycle.
    signal raw_sum_r   : y32_int32_array_t := (others => (others => '0'));
    signal raw_valid_r : std_logic := '0';

    -- Stage B: registered Q.20 scale multiply, 5th cycle.
    signal product_fx_r : y32_fx_array_t := (others => (others => '0'));
    signal valid_s1_r    : std_logic := '0';

    -- Stage C: registered bias-add + ReLU, 6th cycle.
    signal y_bn_relu_fx_r : y32_out_array_t := (others => (others => '0'));
    signal valid_s2_r      : std_logic := '0';

begin

    -- ================================================================
    -- 32 x 3 = 96 parallel raw INT8 dot products (EXISTING, UNMODIFIED
    -- conv3x3_dot_pipelined_dsp component -- same IP already synthesized
    -- in the 740/740-DSP 32-output direct-parallel design).
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
    -- bias-add+ReLU -- for all 32 kernels in parallel. A static
    -- (locally-static-bound) "for k" loop inside a single clocked
    -- process describes 32 independent register slices; it is NOT a
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
    y15 <= y_bn_relu_fx_r(15);  y16 <= y_bn_relu_fx_r(16);  y17 <= y_bn_relu_fx_r(17);
    y18 <= y_bn_relu_fx_r(18);  y19 <= y_bn_relu_fx_r(19);  y20 <= y_bn_relu_fx_r(20);
    y21 <= y_bn_relu_fx_r(21);  y22 <= y_bn_relu_fx_r(22);  y23 <= y_bn_relu_fx_r(23);
    y24 <= y_bn_relu_fx_r(24);  y25 <= y_bn_relu_fx_r(25);  y26 <= y_bn_relu_fx_r(26);
    y27 <= y_bn_relu_fx_r(27);  y28 <= y_bn_relu_fx_r(28);  y29 <= y_bn_relu_fx_r(29);
    y30 <= y_bn_relu_fx_r(30);  y31 <= y_bn_relu_fx_r(31);

end architecture rtl;
