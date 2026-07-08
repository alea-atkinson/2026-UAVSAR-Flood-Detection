-- stream_conv3x3_3chan_32out_cell_dsp.vhd
-- DSP-aware 32-output extension of stream_conv3x3_3chan_24out_cell_dsp.vhd.
--
-- This implements the COMPLETE first Conv2d layer: enc1.block.0.weight has
-- shape [32, 3, 3, 3], so all 32 output channels (kernels 0-31) are
-- represented here. This is still NOT a full U-Net or full model pipeline
-- -- it is structurally IDENTICAL to stream_conv3x3_3chan_24out_cell_dsp.vhd
-- (same window-sharing structure, same conv3x3_dot_pipelined_dsp
-- DSP-steered multiply mapping), just mechanically extended from 24 to 32
-- output kernels, from first_layer_kernels0_to31_pkg.
--
-- Motivation: real synthesized DSP-aware data points already exist for 8,
-- 16, and 24 output channels on the Artix-7 200T
-- (hardware/vhdl_conv3x3/first_layer_24out_dsp_200t_summary.md):
--   8-output:  668 LUTs,  410 regs, 203/740 DSPs, 0 warnings
--   16-output: 929 LUTs,  418 regs, 405/740 DSPs, 0 warnings
--   24-output: 3374 LUTs, 706 regs, 731/740 DSPs, 1 warning (DSP overutilized,
--              CARRY4/LUT fallback evidence in the final cross-channel
--              summation stage -- see
--              hardware/vhdl_conv3x3/first_layer_24out_dsp_fallback_analysis.md)
-- This design completes the first-layer output-channel scaling study by
-- testing whether the COMPLETE 32-output first Conv2d layer synthesizes at
-- all on the 200T, and what DSP/LUT split Vivado chooses. It should be
-- treated as a mixed DSP+LUT mapping experiment, not expected to fit
-- cleanly. Not board-tested, not a measured-speedup or measured-power
-- claim, and not the full U-Net or model pipeline.
--
-- ---- Architecture (window reuse, unchanged from the 8/16/24-output DSP variants) --
-- Three window3x3_stream instances (one PER INPUT CHANNEL, not per kernel):
-- all thirty-two output kernels read the SAME c0_p0..c0_p8 / c1_p* / c2_p*
-- window signals, just with different weight constants.
--
--   window3x3_stream x3               sliding 3x3 window buffers, shared by all kernels
--   conv3x3_dot_pipelined_dsp x96     32 kernels x 3 input channels, zero bias each,
--                                     DSP-steered multiplies
--   final_sum x32 (1 register each)   y{k}_r <= bias_k + yk_c0 + yk_c1 + yk_c2
--
-- Because all input channels (and therefore all kernels) share the same
-- valid_in, every window generator is in lock-step and every dot-product's
-- valid_out is identical. The cell uses kernel 0 / channel 0's dot-product
-- valid to gate all thirty-two final summation stages.
--
-- ---- Latency ---------------------------------------------------------------
-- 4 clock cycles from triggering pixel to visible output -- IDENTICAL to
-- stream_conv3x3_3chan_24out_cell_dsp.vhd, since conv3x3_dot_pipelined_dsp has
-- the same 3-stage pipeline latency regardless of kernel count.
--
-- ---- Weight source -----------------------------------------------------
-- Kernels 0-31 (INT8 weights, symmetric per-tensor quantization) are taken
-- directly from hardware/vhdl_conv3x3/first_layer_kernels0_to31_pkg.vhd,
-- generated from enc1.block.0.weight of the trained Alea-tuned checkpoint
-- by scripts/export_first_layer_multi_kernel_vhdl_vectors.py 0-31.
-- Bias is 0 for all thirty-two kernels (Conv2d has no direct bias;
-- BatchNorm is NOT folded in here).
--
-- ---- Arithmetic width ---------------------------------------------------
-- INT8 x INT8 -> INT32 per dot product, identical to the 1/4/8/16/24-output
-- cells. 96 DSP-mappable multiply groups (32 kernels x 3 channels x
-- 3x3=9 taps each = 864 multiplies total) run in parallel.
--
-- ---- Scope ----------------------------------------------------------------
-- This is the complete 32-output first-layer convolution HARDWARE
-- PROTOTYPE (the full first Conv2d layer's output channels), not full
-- U-Net FPGA inference, not board-tested, and does not include BatchNorm
-- folding, padding, ReLU, or a measured speedup/power figure. It extends
-- the existing verified 24-output DSP-aware design; it is not a new
-- architecture.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_kernels0_to31_pkg.all;

entity stream_conv3x3_3chan_32out_cell_dsp is
    generic (
        IMG_WIDTH : positive := 5
    );
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;        -- synchronous, active high
        valid_in : in  std_logic;

        -- Three input-channel pixel streams (row-major, one triplet per clock)
        pixel_c0 : in signed(7 downto 0);
        pixel_c1 : in signed(7 downto 0);
        pixel_c2 : in signed(7 downto 0);

        -- Output: thirty-two INT32 results (one per kernel) per valid window
        -- triplet, 4-cycle latency -- identical timing to the 1/4/8/16/24-output cells.
        valid_out : out std_logic;
        y0 : out signed(31 downto 0);
        y1 : out signed(31 downto 0);
        y2 : out signed(31 downto 0);
        y3 : out signed(31 downto 0);
        y4 : out signed(31 downto 0);
        y5 : out signed(31 downto 0);
        y6 : out signed(31 downto 0);
        y7 : out signed(31 downto 0);
        y8 : out signed(31 downto 0);
        y9 : out signed(31 downto 0);
        y10 : out signed(31 downto 0);
        y11 : out signed(31 downto 0);
        y12 : out signed(31 downto 0);
        y13 : out signed(31 downto 0);
        y14 : out signed(31 downto 0);
        y15 : out signed(31 downto 0);
        y16 : out signed(31 downto 0);
        y17 : out signed(31 downto 0);
        y18 : out signed(31 downto 0);
        y19 : out signed(31 downto 0);
        y20 : out signed(31 downto 0);
        y21 : out signed(31 downto 0);
        y22 : out signed(31 downto 0);
        y23 : out signed(31 downto 0);
        y24 : out signed(31 downto 0);
        y25 : out signed(31 downto 0);
        y26 : out signed(31 downto 0);
        y27 : out signed(31 downto 0);
        y28 : out signed(31 downto 0);
        y29 : out signed(31 downto 0);
        y30 : out signed(31 downto 0);
        y31 : out signed(31 downto 0)
    );
end entity stream_conv3x3_3chan_32out_cell_dsp;

architecture rtl of stream_conv3x3_3chan_32out_cell_dsp is

    -- ----------------------------------------------------------------
    -- Zero bias constant driven into each per-channel dot product.
    -- Each kernel's shared bias is added only once in its final
    -- summation stage.
    -- ----------------------------------------------------------------
    constant ZERO32 : signed(31 downto 0) := (others => '0');

    -- ----------------------------------------------------------------
    -- Convert the package's plain-integer kernel constants to
    -- signed(7 downto 0) arrays, once, at elaboration time.
    -- ----------------------------------------------------------------
    type signed8_kernel_t is array (0 to 8) of signed(7 downto 0);

    function to_signed_kernel(k : int8_kernel_t) return signed8_kernel_t is
        variable r : signed8_kernel_t;
    begin
        for i in 0 to 8 loop
            r(i) := to_signed(k(i), 8);
        end loop;
        return r;
    end function to_signed_kernel;

    constant K0_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL0_CH0_W);
    constant K0_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL0_CH1_W);
    constant K0_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL0_CH2_W);
    constant K0_BIAS32 : signed(31 downto 0) := to_signed(KERNEL0_BIAS, 32);

    constant K1_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL1_CH0_W);
    constant K1_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL1_CH1_W);
    constant K1_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL1_CH2_W);
    constant K1_BIAS32 : signed(31 downto 0) := to_signed(KERNEL1_BIAS, 32);

    constant K2_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL2_CH0_W);
    constant K2_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL2_CH1_W);
    constant K2_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL2_CH2_W);
    constant K2_BIAS32 : signed(31 downto 0) := to_signed(KERNEL2_BIAS, 32);

    constant K3_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL3_CH0_W);
    constant K3_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL3_CH1_W);
    constant K3_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL3_CH2_W);
    constant K3_BIAS32 : signed(31 downto 0) := to_signed(KERNEL3_BIAS, 32);

    constant K4_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL4_CH0_W);
    constant K4_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL4_CH1_W);
    constant K4_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL4_CH2_W);
    constant K4_BIAS32 : signed(31 downto 0) := to_signed(KERNEL4_BIAS, 32);

    constant K5_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL5_CH0_W);
    constant K5_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL5_CH1_W);
    constant K5_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL5_CH2_W);
    constant K5_BIAS32 : signed(31 downto 0) := to_signed(KERNEL5_BIAS, 32);

    constant K6_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL6_CH0_W);
    constant K6_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL6_CH1_W);
    constant K6_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL6_CH2_W);
    constant K6_BIAS32 : signed(31 downto 0) := to_signed(KERNEL6_BIAS, 32);

    constant K7_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL7_CH0_W);
    constant K7_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL7_CH1_W);
    constant K7_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL7_CH2_W);
    constant K7_BIAS32 : signed(31 downto 0) := to_signed(KERNEL7_BIAS, 32);

    constant K8_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL8_CH0_W);
    constant K8_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL8_CH1_W);
    constant K8_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL8_CH2_W);
    constant K8_BIAS32 : signed(31 downto 0) := to_signed(KERNEL8_BIAS, 32);

    constant K9_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL9_CH0_W);
    constant K9_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL9_CH1_W);
    constant K9_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL9_CH2_W);
    constant K9_BIAS32 : signed(31 downto 0) := to_signed(KERNEL9_BIAS, 32);

    constant K10_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL10_CH0_W);
    constant K10_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL10_CH1_W);
    constant K10_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL10_CH2_W);
    constant K10_BIAS32 : signed(31 downto 0) := to_signed(KERNEL10_BIAS, 32);

    constant K11_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL11_CH0_W);
    constant K11_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL11_CH1_W);
    constant K11_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL11_CH2_W);
    constant K11_BIAS32 : signed(31 downto 0) := to_signed(KERNEL11_BIAS, 32);

    constant K12_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL12_CH0_W);
    constant K12_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL12_CH1_W);
    constant K12_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL12_CH2_W);
    constant K12_BIAS32 : signed(31 downto 0) := to_signed(KERNEL12_BIAS, 32);

    constant K13_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL13_CH0_W);
    constant K13_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL13_CH1_W);
    constant K13_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL13_CH2_W);
    constant K13_BIAS32 : signed(31 downto 0) := to_signed(KERNEL13_BIAS, 32);

    constant K14_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL14_CH0_W);
    constant K14_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL14_CH1_W);
    constant K14_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL14_CH2_W);
    constant K14_BIAS32 : signed(31 downto 0) := to_signed(KERNEL14_BIAS, 32);

    constant K15_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL15_CH0_W);
    constant K15_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL15_CH1_W);
    constant K15_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL15_CH2_W);
    constant K15_BIAS32 : signed(31 downto 0) := to_signed(KERNEL15_BIAS, 32);

    constant K16_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL16_CH0_W);
    constant K16_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL16_CH1_W);
    constant K16_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL16_CH2_W);
    constant K16_BIAS32 : signed(31 downto 0) := to_signed(KERNEL16_BIAS, 32);

    constant K17_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL17_CH0_W);
    constant K17_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL17_CH1_W);
    constant K17_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL17_CH2_W);
    constant K17_BIAS32 : signed(31 downto 0) := to_signed(KERNEL17_BIAS, 32);

    constant K18_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL18_CH0_W);
    constant K18_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL18_CH1_W);
    constant K18_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL18_CH2_W);
    constant K18_BIAS32 : signed(31 downto 0) := to_signed(KERNEL18_BIAS, 32);

    constant K19_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL19_CH0_W);
    constant K19_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL19_CH1_W);
    constant K19_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL19_CH2_W);
    constant K19_BIAS32 : signed(31 downto 0) := to_signed(KERNEL19_BIAS, 32);

    constant K20_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL20_CH0_W);
    constant K20_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL20_CH1_W);
    constant K20_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL20_CH2_W);
    constant K20_BIAS32 : signed(31 downto 0) := to_signed(KERNEL20_BIAS, 32);

    constant K21_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL21_CH0_W);
    constant K21_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL21_CH1_W);
    constant K21_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL21_CH2_W);
    constant K21_BIAS32 : signed(31 downto 0) := to_signed(KERNEL21_BIAS, 32);

    constant K22_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL22_CH0_W);
    constant K22_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL22_CH1_W);
    constant K22_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL22_CH2_W);
    constant K22_BIAS32 : signed(31 downto 0) := to_signed(KERNEL22_BIAS, 32);

    constant K23_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL23_CH0_W);
    constant K23_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL23_CH1_W);
    constant K23_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL23_CH2_W);
    constant K23_BIAS32 : signed(31 downto 0) := to_signed(KERNEL23_BIAS, 32);

    constant K24_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL24_CH0_W);
    constant K24_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL24_CH1_W);
    constant K24_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL24_CH2_W);
    constant K24_BIAS32 : signed(31 downto 0) := to_signed(KERNEL24_BIAS, 32);

    constant K25_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL25_CH0_W);
    constant K25_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL25_CH1_W);
    constant K25_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL25_CH2_W);
    constant K25_BIAS32 : signed(31 downto 0) := to_signed(KERNEL25_BIAS, 32);

    constant K26_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL26_CH0_W);
    constant K26_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL26_CH1_W);
    constant K26_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL26_CH2_W);
    constant K26_BIAS32 : signed(31 downto 0) := to_signed(KERNEL26_BIAS, 32);

    constant K27_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL27_CH0_W);
    constant K27_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL27_CH1_W);
    constant K27_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL27_CH2_W);
    constant K27_BIAS32 : signed(31 downto 0) := to_signed(KERNEL27_BIAS, 32);

    constant K28_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL28_CH0_W);
    constant K28_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL28_CH1_W);
    constant K28_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL28_CH2_W);
    constant K28_BIAS32 : signed(31 downto 0) := to_signed(KERNEL28_BIAS, 32);

    constant K29_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL29_CH0_W);
    constant K29_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL29_CH1_W);
    constant K29_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL29_CH2_W);
    constant K29_BIAS32 : signed(31 downto 0) := to_signed(KERNEL29_BIAS, 32);

    constant K30_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL30_CH0_W);
    constant K30_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL30_CH1_W);
    constant K30_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL30_CH2_W);
    constant K30_BIAS32 : signed(31 downto 0) := to_signed(KERNEL30_BIAS, 32);

    constant K31_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL31_CH0_W);
    constant K31_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL31_CH1_W);
    constant K31_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL31_CH2_W);
    constant K31_BIAS32 : signed(31 downto 0) := to_signed(KERNEL31_BIAS, 32);

    -- ----------------------------------------------------------------
    -- Internal handshakes: window-generator -> dot-product.
    -- All three channels share the same valid_in so their win_valid
    -- signals are always identical in value and timing.
    -- ----------------------------------------------------------------
    signal win_valid_c0, win_valid_c1, win_valid_c2 : std_logic;

    -- ----------------------------------------------------------------
    -- 3x3 pixel windows (9 signals per input channel), shared by all
    -- thirty-two output kernels.
    -- ----------------------------------------------------------------
    signal c0_p0, c0_p1, c0_p2, c0_p3, c0_p4, c0_p5,
           c0_p6, c0_p7, c0_p8 : signed(7 downto 0);

    signal c1_p0, c1_p1, c1_p2, c1_p3, c1_p4, c1_p5,
           c1_p6, c1_p7, c1_p8 : signed(7 downto 0);

    signal c2_p0, c2_p1, c2_p2, c2_p3, c2_p4, c2_p5,
           c2_p6, c2_p7, c2_p8 : signed(7 downto 0);

    -- ----------------------------------------------------------------
    -- Per-kernel, per-channel dot-product outputs (zero bias baked in)
    -- ----------------------------------------------------------------
    signal y0_c0, y0_c1, y0_c2 : signed(31 downto 0);
    signal y1_c0, y1_c1, y1_c2 : signed(31 downto 0);
    signal y2_c0, y2_c1, y2_c2 : signed(31 downto 0);
    signal y3_c0, y3_c1, y3_c2 : signed(31 downto 0);
    signal y4_c0, y4_c1, y4_c2 : signed(31 downto 0);
    signal y5_c0, y5_c1, y5_c2 : signed(31 downto 0);
    signal y6_c0, y6_c1, y6_c2 : signed(31 downto 0);
    signal y7_c0, y7_c1, y7_c2 : signed(31 downto 0);
    signal y8_c0, y8_c1, y8_c2 : signed(31 downto 0);
    signal y9_c0, y9_c1, y9_c2 : signed(31 downto 0);
    signal y10_c0, y10_c1, y10_c2 : signed(31 downto 0);
    signal y11_c0, y11_c1, y11_c2 : signed(31 downto 0);
    signal y12_c0, y12_c1, y12_c2 : signed(31 downto 0);
    signal y13_c0, y13_c1, y13_c2 : signed(31 downto 0);
    signal y14_c0, y14_c1, y14_c2 : signed(31 downto 0);
    signal y15_c0, y15_c1, y15_c2 : signed(31 downto 0);
    signal y16_c0, y16_c1, y16_c2 : signed(31 downto 0);
    signal y17_c0, y17_c1, y17_c2 : signed(31 downto 0);
    signal y18_c0, y18_c1, y18_c2 : signed(31 downto 0);
    signal y19_c0, y19_c1, y19_c2 : signed(31 downto 0);
    signal y20_c0, y20_c1, y20_c2 : signed(31 downto 0);
    signal y21_c0, y21_c1, y21_c2 : signed(31 downto 0);
    signal y22_c0, y22_c1, y22_c2 : signed(31 downto 0);
    signal y23_c0, y23_c1, y23_c2 : signed(31 downto 0);
    signal y24_c0, y24_c1, y24_c2 : signed(31 downto 0);
    signal y25_c0, y25_c1, y25_c2 : signed(31 downto 0);
    signal y26_c0, y26_c1, y26_c2 : signed(31 downto 0);
    signal y27_c0, y27_c1, y27_c2 : signed(31 downto 0);
    signal y28_c0, y28_c1, y28_c2 : signed(31 downto 0);
    signal y29_c0, y29_c1, y29_c2 : signed(31 downto 0);
    signal y30_c0, y30_c1, y30_c2 : signed(31 downto 0);
    signal y31_c0, y31_c1, y31_c2 : signed(31 downto 0);

    -- Single shared valid gate: kernel 0 / channel 0's dot valid_out.
    -- All multiply-group dot products share identical valid_in timing, so
    -- their valid_out signals are identical; only one needs to be observed.
    signal valid_dot : std_logic;

    -- ----------------------------------------------------------------
    -- Final registered summation stage, one per kernel (adds that
    -- kernel's bias once).
    -- ----------------------------------------------------------------
    signal y0_r, y1_r, y2_r, y3_r, y4_r, y5_r, y6_r, y7_r, y8_r, y9_r, y10_r, y11_r, y12_r, y13_r, y14_r, y15_r, y16_r, y17_r, y18_r, y19_r, y20_r, y21_r, y22_r, y23_r, y24_r, y25_r, y26_r, y27_r, y28_r, y29_r, y30_r, y31_r : signed(31 downto 0) := (others => '0');
    signal valid_r                : std_logic            := '0';

begin

    -- ================================================================
    -- Window generators -- ONE PER INPUT CHANNEL, shared by all 32 kernels.
    -- ================================================================
    win_gen_c0 : entity work.window3x3_stream
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk => clk,  rst => rst,  valid_in => valid_in,
            pixel_in  => pixel_c0,
            valid_out => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8
        );

    win_gen_c1 : entity work.window3x3_stream
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk => clk,  rst => rst,  valid_in => valid_in,
            pixel_in  => pixel_c1,
            valid_out => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8
        );

    win_gen_c2 : entity work.window3x3_stream
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk => clk,  rst => rst,  valid_in => valid_in,
            pixel_in  => pixel_c2,
            valid_out => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8
        );

    -- ================================================================
    -- Kernel 0 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k0_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K0_CH0(0),  w1 => K0_CH0(1),  w2 => K0_CH0(2),
            w3 => K0_CH0(3),  w4 => K0_CH0(4),  w5 => K0_CH0(5),
            w6 => K0_CH0(6),  w7 => K0_CH0(7),  w8 => K0_CH0(8),
            bias      => ZERO32,
            valid_out => valid_dot,
            y         => y0_c0
        );

    dot_k0_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K0_CH1(0),  w1 => K0_CH1(1),  w2 => K0_CH1(2),
            w3 => K0_CH1(3),  w4 => K0_CH1(4),  w5 => K0_CH1(5),
            w6 => K0_CH1(6),  w7 => K0_CH1(7),  w8 => K0_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y0_c1
        );

    dot_k0_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K0_CH2(0),  w1 => K0_CH2(1),  w2 => K0_CH2(2),
            w3 => K0_CH2(3),  w4 => K0_CH2(4),  w5 => K0_CH2(5),
            w6 => K0_CH2(6),  w7 => K0_CH2(7),  w8 => K0_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y0_c2
        );

    -- ================================================================
    -- Kernel 1 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k1_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K1_CH0(0),  w1 => K1_CH0(1),  w2 => K1_CH0(2),
            w3 => K1_CH0(3),  w4 => K1_CH0(4),  w5 => K1_CH0(5),
            w6 => K1_CH0(6),  w7 => K1_CH0(7),  w8 => K1_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y1_c0
        );

    dot_k1_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K1_CH1(0),  w1 => K1_CH1(1),  w2 => K1_CH1(2),
            w3 => K1_CH1(3),  w4 => K1_CH1(4),  w5 => K1_CH1(5),
            w6 => K1_CH1(6),  w7 => K1_CH1(7),  w8 => K1_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y1_c1
        );

    dot_k1_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K1_CH2(0),  w1 => K1_CH2(1),  w2 => K1_CH2(2),
            w3 => K1_CH2(3),  w4 => K1_CH2(4),  w5 => K1_CH2(5),
            w6 => K1_CH2(6),  w7 => K1_CH2(7),  w8 => K1_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y1_c2
        );

    -- ================================================================
    -- Kernel 2 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k2_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K2_CH0(0),  w1 => K2_CH0(1),  w2 => K2_CH0(2),
            w3 => K2_CH0(3),  w4 => K2_CH0(4),  w5 => K2_CH0(5),
            w6 => K2_CH0(6),  w7 => K2_CH0(7),  w8 => K2_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y2_c0
        );

    dot_k2_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K2_CH1(0),  w1 => K2_CH1(1),  w2 => K2_CH1(2),
            w3 => K2_CH1(3),  w4 => K2_CH1(4),  w5 => K2_CH1(5),
            w6 => K2_CH1(6),  w7 => K2_CH1(7),  w8 => K2_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y2_c1
        );

    dot_k2_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K2_CH2(0),  w1 => K2_CH2(1),  w2 => K2_CH2(2),
            w3 => K2_CH2(3),  w4 => K2_CH2(4),  w5 => K2_CH2(5),
            w6 => K2_CH2(6),  w7 => K2_CH2(7),  w8 => K2_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y2_c2
        );

    -- ================================================================
    -- Kernel 3 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k3_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K3_CH0(0),  w1 => K3_CH0(1),  w2 => K3_CH0(2),
            w3 => K3_CH0(3),  w4 => K3_CH0(4),  w5 => K3_CH0(5),
            w6 => K3_CH0(6),  w7 => K3_CH0(7),  w8 => K3_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y3_c0
        );

    dot_k3_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K3_CH1(0),  w1 => K3_CH1(1),  w2 => K3_CH1(2),
            w3 => K3_CH1(3),  w4 => K3_CH1(4),  w5 => K3_CH1(5),
            w6 => K3_CH1(6),  w7 => K3_CH1(7),  w8 => K3_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y3_c1
        );

    dot_k3_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K3_CH2(0),  w1 => K3_CH2(1),  w2 => K3_CH2(2),
            w3 => K3_CH2(3),  w4 => K3_CH2(4),  w5 => K3_CH2(5),
            w6 => K3_CH2(6),  w7 => K3_CH2(7),  w8 => K3_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y3_c2
        );

    -- ================================================================
    -- Kernel 4 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k4_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K4_CH0(0),  w1 => K4_CH0(1),  w2 => K4_CH0(2),
            w3 => K4_CH0(3),  w4 => K4_CH0(4),  w5 => K4_CH0(5),
            w6 => K4_CH0(6),  w7 => K4_CH0(7),  w8 => K4_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y4_c0
        );

    dot_k4_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K4_CH1(0),  w1 => K4_CH1(1),  w2 => K4_CH1(2),
            w3 => K4_CH1(3),  w4 => K4_CH1(4),  w5 => K4_CH1(5),
            w6 => K4_CH1(6),  w7 => K4_CH1(7),  w8 => K4_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y4_c1
        );

    dot_k4_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K4_CH2(0),  w1 => K4_CH2(1),  w2 => K4_CH2(2),
            w3 => K4_CH2(3),  w4 => K4_CH2(4),  w5 => K4_CH2(5),
            w6 => K4_CH2(6),  w7 => K4_CH2(7),  w8 => K4_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y4_c2
        );

    -- ================================================================
    -- Kernel 5 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k5_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K5_CH0(0),  w1 => K5_CH0(1),  w2 => K5_CH0(2),
            w3 => K5_CH0(3),  w4 => K5_CH0(4),  w5 => K5_CH0(5),
            w6 => K5_CH0(6),  w7 => K5_CH0(7),  w8 => K5_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y5_c0
        );

    dot_k5_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K5_CH1(0),  w1 => K5_CH1(1),  w2 => K5_CH1(2),
            w3 => K5_CH1(3),  w4 => K5_CH1(4),  w5 => K5_CH1(5),
            w6 => K5_CH1(6),  w7 => K5_CH1(7),  w8 => K5_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y5_c1
        );

    dot_k5_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K5_CH2(0),  w1 => K5_CH2(1),  w2 => K5_CH2(2),
            w3 => K5_CH2(3),  w4 => K5_CH2(4),  w5 => K5_CH2(5),
            w6 => K5_CH2(6),  w7 => K5_CH2(7),  w8 => K5_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y5_c2
        );

    -- ================================================================
    -- Kernel 6 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k6_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K6_CH0(0),  w1 => K6_CH0(1),  w2 => K6_CH0(2),
            w3 => K6_CH0(3),  w4 => K6_CH0(4),  w5 => K6_CH0(5),
            w6 => K6_CH0(6),  w7 => K6_CH0(7),  w8 => K6_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y6_c0
        );

    dot_k6_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K6_CH1(0),  w1 => K6_CH1(1),  w2 => K6_CH1(2),
            w3 => K6_CH1(3),  w4 => K6_CH1(4),  w5 => K6_CH1(5),
            w6 => K6_CH1(6),  w7 => K6_CH1(7),  w8 => K6_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y6_c1
        );

    dot_k6_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K6_CH2(0),  w1 => K6_CH2(1),  w2 => K6_CH2(2),
            w3 => K6_CH2(3),  w4 => K6_CH2(4),  w5 => K6_CH2(5),
            w6 => K6_CH2(6),  w7 => K6_CH2(7),  w8 => K6_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y6_c2
        );

    -- ================================================================
    -- Kernel 7 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k7_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K7_CH0(0),  w1 => K7_CH0(1),  w2 => K7_CH0(2),
            w3 => K7_CH0(3),  w4 => K7_CH0(4),  w5 => K7_CH0(5),
            w6 => K7_CH0(6),  w7 => K7_CH0(7),  w8 => K7_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y7_c0
        );

    dot_k7_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K7_CH1(0),  w1 => K7_CH1(1),  w2 => K7_CH1(2),
            w3 => K7_CH1(3),  w4 => K7_CH1(4),  w5 => K7_CH1(5),
            w6 => K7_CH1(6),  w7 => K7_CH1(7),  w8 => K7_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y7_c1
        );

    dot_k7_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K7_CH2(0),  w1 => K7_CH2(1),  w2 => K7_CH2(2),
            w3 => K7_CH2(3),  w4 => K7_CH2(4),  w5 => K7_CH2(5),
            w6 => K7_CH2(6),  w7 => K7_CH2(7),  w8 => K7_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y7_c2
        );

    -- ================================================================
    -- Kernel 8 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k8_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K8_CH0(0),  w1 => K8_CH0(1),  w2 => K8_CH0(2),
            w3 => K8_CH0(3),  w4 => K8_CH0(4),  w5 => K8_CH0(5),
            w6 => K8_CH0(6),  w7 => K8_CH0(7),  w8 => K8_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y8_c0
        );

    dot_k8_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K8_CH1(0),  w1 => K8_CH1(1),  w2 => K8_CH1(2),
            w3 => K8_CH1(3),  w4 => K8_CH1(4),  w5 => K8_CH1(5),
            w6 => K8_CH1(6),  w7 => K8_CH1(7),  w8 => K8_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y8_c1
        );

    dot_k8_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K8_CH2(0),  w1 => K8_CH2(1),  w2 => K8_CH2(2),
            w3 => K8_CH2(3),  w4 => K8_CH2(4),  w5 => K8_CH2(5),
            w6 => K8_CH2(6),  w7 => K8_CH2(7),  w8 => K8_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y8_c2
        );

    -- ================================================================
    -- Kernel 9 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k9_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K9_CH0(0),  w1 => K9_CH0(1),  w2 => K9_CH0(2),
            w3 => K9_CH0(3),  w4 => K9_CH0(4),  w5 => K9_CH0(5),
            w6 => K9_CH0(6),  w7 => K9_CH0(7),  w8 => K9_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y9_c0
        );

    dot_k9_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K9_CH1(0),  w1 => K9_CH1(1),  w2 => K9_CH1(2),
            w3 => K9_CH1(3),  w4 => K9_CH1(4),  w5 => K9_CH1(5),
            w6 => K9_CH1(6),  w7 => K9_CH1(7),  w8 => K9_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y9_c1
        );

    dot_k9_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K9_CH2(0),  w1 => K9_CH2(1),  w2 => K9_CH2(2),
            w3 => K9_CH2(3),  w4 => K9_CH2(4),  w5 => K9_CH2(5),
            w6 => K9_CH2(6),  w7 => K9_CH2(7),  w8 => K9_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y9_c2
        );

    -- ================================================================
    -- Kernel 10 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k10_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K10_CH0(0),  w1 => K10_CH0(1),  w2 => K10_CH0(2),
            w3 => K10_CH0(3),  w4 => K10_CH0(4),  w5 => K10_CH0(5),
            w6 => K10_CH0(6),  w7 => K10_CH0(7),  w8 => K10_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y10_c0
        );

    dot_k10_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K10_CH1(0),  w1 => K10_CH1(1),  w2 => K10_CH1(2),
            w3 => K10_CH1(3),  w4 => K10_CH1(4),  w5 => K10_CH1(5),
            w6 => K10_CH1(6),  w7 => K10_CH1(7),  w8 => K10_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y10_c1
        );

    dot_k10_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K10_CH2(0),  w1 => K10_CH2(1),  w2 => K10_CH2(2),
            w3 => K10_CH2(3),  w4 => K10_CH2(4),  w5 => K10_CH2(5),
            w6 => K10_CH2(6),  w7 => K10_CH2(7),  w8 => K10_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y10_c2
        );

    -- ================================================================
    -- Kernel 11 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k11_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K11_CH0(0),  w1 => K11_CH0(1),  w2 => K11_CH0(2),
            w3 => K11_CH0(3),  w4 => K11_CH0(4),  w5 => K11_CH0(5),
            w6 => K11_CH0(6),  w7 => K11_CH0(7),  w8 => K11_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y11_c0
        );

    dot_k11_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K11_CH1(0),  w1 => K11_CH1(1),  w2 => K11_CH1(2),
            w3 => K11_CH1(3),  w4 => K11_CH1(4),  w5 => K11_CH1(5),
            w6 => K11_CH1(6),  w7 => K11_CH1(7),  w8 => K11_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y11_c1
        );

    dot_k11_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K11_CH2(0),  w1 => K11_CH2(1),  w2 => K11_CH2(2),
            w3 => K11_CH2(3),  w4 => K11_CH2(4),  w5 => K11_CH2(5),
            w6 => K11_CH2(6),  w7 => K11_CH2(7),  w8 => K11_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y11_c2
        );

    -- ================================================================
    -- Kernel 12 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k12_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K12_CH0(0),  w1 => K12_CH0(1),  w2 => K12_CH0(2),
            w3 => K12_CH0(3),  w4 => K12_CH0(4),  w5 => K12_CH0(5),
            w6 => K12_CH0(6),  w7 => K12_CH0(7),  w8 => K12_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y12_c0
        );

    dot_k12_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K12_CH1(0),  w1 => K12_CH1(1),  w2 => K12_CH1(2),
            w3 => K12_CH1(3),  w4 => K12_CH1(4),  w5 => K12_CH1(5),
            w6 => K12_CH1(6),  w7 => K12_CH1(7),  w8 => K12_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y12_c1
        );

    dot_k12_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K12_CH2(0),  w1 => K12_CH2(1),  w2 => K12_CH2(2),
            w3 => K12_CH2(3),  w4 => K12_CH2(4),  w5 => K12_CH2(5),
            w6 => K12_CH2(6),  w7 => K12_CH2(7),  w8 => K12_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y12_c2
        );

    -- ================================================================
    -- Kernel 13 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k13_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K13_CH0(0),  w1 => K13_CH0(1),  w2 => K13_CH0(2),
            w3 => K13_CH0(3),  w4 => K13_CH0(4),  w5 => K13_CH0(5),
            w6 => K13_CH0(6),  w7 => K13_CH0(7),  w8 => K13_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y13_c0
        );

    dot_k13_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K13_CH1(0),  w1 => K13_CH1(1),  w2 => K13_CH1(2),
            w3 => K13_CH1(3),  w4 => K13_CH1(4),  w5 => K13_CH1(5),
            w6 => K13_CH1(6),  w7 => K13_CH1(7),  w8 => K13_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y13_c1
        );

    dot_k13_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K13_CH2(0),  w1 => K13_CH2(1),  w2 => K13_CH2(2),
            w3 => K13_CH2(3),  w4 => K13_CH2(4),  w5 => K13_CH2(5),
            w6 => K13_CH2(6),  w7 => K13_CH2(7),  w8 => K13_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y13_c2
        );

    -- ================================================================
    -- Kernel 14 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k14_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K14_CH0(0),  w1 => K14_CH0(1),  w2 => K14_CH0(2),
            w3 => K14_CH0(3),  w4 => K14_CH0(4),  w5 => K14_CH0(5),
            w6 => K14_CH0(6),  w7 => K14_CH0(7),  w8 => K14_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y14_c0
        );

    dot_k14_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K14_CH1(0),  w1 => K14_CH1(1),  w2 => K14_CH1(2),
            w3 => K14_CH1(3),  w4 => K14_CH1(4),  w5 => K14_CH1(5),
            w6 => K14_CH1(6),  w7 => K14_CH1(7),  w8 => K14_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y14_c1
        );

    dot_k14_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K14_CH2(0),  w1 => K14_CH2(1),  w2 => K14_CH2(2),
            w3 => K14_CH2(3),  w4 => K14_CH2(4),  w5 => K14_CH2(5),
            w6 => K14_CH2(6),  w7 => K14_CH2(7),  w8 => K14_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y14_c2
        );

    -- ================================================================
    -- Kernel 15 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k15_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K15_CH0(0),  w1 => K15_CH0(1),  w2 => K15_CH0(2),
            w3 => K15_CH0(3),  w4 => K15_CH0(4),  w5 => K15_CH0(5),
            w6 => K15_CH0(6),  w7 => K15_CH0(7),  w8 => K15_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y15_c0
        );

    dot_k15_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K15_CH1(0),  w1 => K15_CH1(1),  w2 => K15_CH1(2),
            w3 => K15_CH1(3),  w4 => K15_CH1(4),  w5 => K15_CH1(5),
            w6 => K15_CH1(6),  w7 => K15_CH1(7),  w8 => K15_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y15_c1
        );

    dot_k15_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K15_CH2(0),  w1 => K15_CH2(1),  w2 => K15_CH2(2),
            w3 => K15_CH2(3),  w4 => K15_CH2(4),  w5 => K15_CH2(5),
            w6 => K15_CH2(6),  w7 => K15_CH2(7),  w8 => K15_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y15_c2
        );

    -- ================================================================
    -- Kernel 16 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k16_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K16_CH0(0),  w1 => K16_CH0(1),  w2 => K16_CH0(2),
            w3 => K16_CH0(3),  w4 => K16_CH0(4),  w5 => K16_CH0(5),
            w6 => K16_CH0(6),  w7 => K16_CH0(7),  w8 => K16_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y16_c0
        );

    dot_k16_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K16_CH1(0),  w1 => K16_CH1(1),  w2 => K16_CH1(2),
            w3 => K16_CH1(3),  w4 => K16_CH1(4),  w5 => K16_CH1(5),
            w6 => K16_CH1(6),  w7 => K16_CH1(7),  w8 => K16_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y16_c1
        );

    dot_k16_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K16_CH2(0),  w1 => K16_CH2(1),  w2 => K16_CH2(2),
            w3 => K16_CH2(3),  w4 => K16_CH2(4),  w5 => K16_CH2(5),
            w6 => K16_CH2(6),  w7 => K16_CH2(7),  w8 => K16_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y16_c2
        );

    -- ================================================================
    -- Kernel 17 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k17_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K17_CH0(0),  w1 => K17_CH0(1),  w2 => K17_CH0(2),
            w3 => K17_CH0(3),  w4 => K17_CH0(4),  w5 => K17_CH0(5),
            w6 => K17_CH0(6),  w7 => K17_CH0(7),  w8 => K17_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y17_c0
        );

    dot_k17_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K17_CH1(0),  w1 => K17_CH1(1),  w2 => K17_CH1(2),
            w3 => K17_CH1(3),  w4 => K17_CH1(4),  w5 => K17_CH1(5),
            w6 => K17_CH1(6),  w7 => K17_CH1(7),  w8 => K17_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y17_c1
        );

    dot_k17_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K17_CH2(0),  w1 => K17_CH2(1),  w2 => K17_CH2(2),
            w3 => K17_CH2(3),  w4 => K17_CH2(4),  w5 => K17_CH2(5),
            w6 => K17_CH2(6),  w7 => K17_CH2(7),  w8 => K17_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y17_c2
        );

    -- ================================================================
    -- Kernel 18 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k18_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K18_CH0(0),  w1 => K18_CH0(1),  w2 => K18_CH0(2),
            w3 => K18_CH0(3),  w4 => K18_CH0(4),  w5 => K18_CH0(5),
            w6 => K18_CH0(6),  w7 => K18_CH0(7),  w8 => K18_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y18_c0
        );

    dot_k18_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K18_CH1(0),  w1 => K18_CH1(1),  w2 => K18_CH1(2),
            w3 => K18_CH1(3),  w4 => K18_CH1(4),  w5 => K18_CH1(5),
            w6 => K18_CH1(6),  w7 => K18_CH1(7),  w8 => K18_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y18_c1
        );

    dot_k18_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K18_CH2(0),  w1 => K18_CH2(1),  w2 => K18_CH2(2),
            w3 => K18_CH2(3),  w4 => K18_CH2(4),  w5 => K18_CH2(5),
            w6 => K18_CH2(6),  w7 => K18_CH2(7),  w8 => K18_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y18_c2
        );

    -- ================================================================
    -- Kernel 19 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k19_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K19_CH0(0),  w1 => K19_CH0(1),  w2 => K19_CH0(2),
            w3 => K19_CH0(3),  w4 => K19_CH0(4),  w5 => K19_CH0(5),
            w6 => K19_CH0(6),  w7 => K19_CH0(7),  w8 => K19_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y19_c0
        );

    dot_k19_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K19_CH1(0),  w1 => K19_CH1(1),  w2 => K19_CH1(2),
            w3 => K19_CH1(3),  w4 => K19_CH1(4),  w5 => K19_CH1(5),
            w6 => K19_CH1(6),  w7 => K19_CH1(7),  w8 => K19_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y19_c1
        );

    dot_k19_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K19_CH2(0),  w1 => K19_CH2(1),  w2 => K19_CH2(2),
            w3 => K19_CH2(3),  w4 => K19_CH2(4),  w5 => K19_CH2(5),
            w6 => K19_CH2(6),  w7 => K19_CH2(7),  w8 => K19_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y19_c2
        );

    -- ================================================================
    -- Kernel 20 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k20_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K20_CH0(0),  w1 => K20_CH0(1),  w2 => K20_CH0(2),
            w3 => K20_CH0(3),  w4 => K20_CH0(4),  w5 => K20_CH0(5),
            w6 => K20_CH0(6),  w7 => K20_CH0(7),  w8 => K20_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y20_c0
        );

    dot_k20_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K20_CH1(0),  w1 => K20_CH1(1),  w2 => K20_CH1(2),
            w3 => K20_CH1(3),  w4 => K20_CH1(4),  w5 => K20_CH1(5),
            w6 => K20_CH1(6),  w7 => K20_CH1(7),  w8 => K20_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y20_c1
        );

    dot_k20_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K20_CH2(0),  w1 => K20_CH2(1),  w2 => K20_CH2(2),
            w3 => K20_CH2(3),  w4 => K20_CH2(4),  w5 => K20_CH2(5),
            w6 => K20_CH2(6),  w7 => K20_CH2(7),  w8 => K20_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y20_c2
        );

    -- ================================================================
    -- Kernel 21 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k21_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K21_CH0(0),  w1 => K21_CH0(1),  w2 => K21_CH0(2),
            w3 => K21_CH0(3),  w4 => K21_CH0(4),  w5 => K21_CH0(5),
            w6 => K21_CH0(6),  w7 => K21_CH0(7),  w8 => K21_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y21_c0
        );

    dot_k21_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K21_CH1(0),  w1 => K21_CH1(1),  w2 => K21_CH1(2),
            w3 => K21_CH1(3),  w4 => K21_CH1(4),  w5 => K21_CH1(5),
            w6 => K21_CH1(6),  w7 => K21_CH1(7),  w8 => K21_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y21_c1
        );

    dot_k21_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K21_CH2(0),  w1 => K21_CH2(1),  w2 => K21_CH2(2),
            w3 => K21_CH2(3),  w4 => K21_CH2(4),  w5 => K21_CH2(5),
            w6 => K21_CH2(6),  w7 => K21_CH2(7),  w8 => K21_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y21_c2
        );

    -- ================================================================
    -- Kernel 22 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k22_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K22_CH0(0),  w1 => K22_CH0(1),  w2 => K22_CH0(2),
            w3 => K22_CH0(3),  w4 => K22_CH0(4),  w5 => K22_CH0(5),
            w6 => K22_CH0(6),  w7 => K22_CH0(7),  w8 => K22_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y22_c0
        );

    dot_k22_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K22_CH1(0),  w1 => K22_CH1(1),  w2 => K22_CH1(2),
            w3 => K22_CH1(3),  w4 => K22_CH1(4),  w5 => K22_CH1(5),
            w6 => K22_CH1(6),  w7 => K22_CH1(7),  w8 => K22_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y22_c1
        );

    dot_k22_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K22_CH2(0),  w1 => K22_CH2(1),  w2 => K22_CH2(2),
            w3 => K22_CH2(3),  w4 => K22_CH2(4),  w5 => K22_CH2(5),
            w6 => K22_CH2(6),  w7 => K22_CH2(7),  w8 => K22_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y22_c2
        );

    -- ================================================================
    -- Kernel 23 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k23_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K23_CH0(0),  w1 => K23_CH0(1),  w2 => K23_CH0(2),
            w3 => K23_CH0(3),  w4 => K23_CH0(4),  w5 => K23_CH0(5),
            w6 => K23_CH0(6),  w7 => K23_CH0(7),  w8 => K23_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y23_c0
        );

    dot_k23_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K23_CH1(0),  w1 => K23_CH1(1),  w2 => K23_CH1(2),
            w3 => K23_CH1(3),  w4 => K23_CH1(4),  w5 => K23_CH1(5),
            w6 => K23_CH1(6),  w7 => K23_CH1(7),  w8 => K23_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y23_c1
        );

    dot_k23_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K23_CH2(0),  w1 => K23_CH2(1),  w2 => K23_CH2(2),
            w3 => K23_CH2(3),  w4 => K23_CH2(4),  w5 => K23_CH2(5),
            w6 => K23_CH2(6),  w7 => K23_CH2(7),  w8 => K23_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y23_c2
        );

    -- ================================================================
    -- Kernel 24 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k24_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K24_CH0(0),  w1 => K24_CH0(1),  w2 => K24_CH0(2),
            w3 => K24_CH0(3),  w4 => K24_CH0(4),  w5 => K24_CH0(5),
            w6 => K24_CH0(6),  w7 => K24_CH0(7),  w8 => K24_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y24_c0
        );

    dot_k24_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K24_CH1(0),  w1 => K24_CH1(1),  w2 => K24_CH1(2),
            w3 => K24_CH1(3),  w4 => K24_CH1(4),  w5 => K24_CH1(5),
            w6 => K24_CH1(6),  w7 => K24_CH1(7),  w8 => K24_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y24_c1
        );

    dot_k24_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K24_CH2(0),  w1 => K24_CH2(1),  w2 => K24_CH2(2),
            w3 => K24_CH2(3),  w4 => K24_CH2(4),  w5 => K24_CH2(5),
            w6 => K24_CH2(6),  w7 => K24_CH2(7),  w8 => K24_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y24_c2
        );

    -- ================================================================
    -- Kernel 25 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k25_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K25_CH0(0),  w1 => K25_CH0(1),  w2 => K25_CH0(2),
            w3 => K25_CH0(3),  w4 => K25_CH0(4),  w5 => K25_CH0(5),
            w6 => K25_CH0(6),  w7 => K25_CH0(7),  w8 => K25_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y25_c0
        );

    dot_k25_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K25_CH1(0),  w1 => K25_CH1(1),  w2 => K25_CH1(2),
            w3 => K25_CH1(3),  w4 => K25_CH1(4),  w5 => K25_CH1(5),
            w6 => K25_CH1(6),  w7 => K25_CH1(7),  w8 => K25_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y25_c1
        );

    dot_k25_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K25_CH2(0),  w1 => K25_CH2(1),  w2 => K25_CH2(2),
            w3 => K25_CH2(3),  w4 => K25_CH2(4),  w5 => K25_CH2(5),
            w6 => K25_CH2(6),  w7 => K25_CH2(7),  w8 => K25_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y25_c2
        );

    -- ================================================================
    -- Kernel 26 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k26_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K26_CH0(0),  w1 => K26_CH0(1),  w2 => K26_CH0(2),
            w3 => K26_CH0(3),  w4 => K26_CH0(4),  w5 => K26_CH0(5),
            w6 => K26_CH0(6),  w7 => K26_CH0(7),  w8 => K26_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y26_c0
        );

    dot_k26_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K26_CH1(0),  w1 => K26_CH1(1),  w2 => K26_CH1(2),
            w3 => K26_CH1(3),  w4 => K26_CH1(4),  w5 => K26_CH1(5),
            w6 => K26_CH1(6),  w7 => K26_CH1(7),  w8 => K26_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y26_c1
        );

    dot_k26_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K26_CH2(0),  w1 => K26_CH2(1),  w2 => K26_CH2(2),
            w3 => K26_CH2(3),  w4 => K26_CH2(4),  w5 => K26_CH2(5),
            w6 => K26_CH2(6),  w7 => K26_CH2(7),  w8 => K26_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y26_c2
        );

    -- ================================================================
    -- Kernel 27 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k27_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K27_CH0(0),  w1 => K27_CH0(1),  w2 => K27_CH0(2),
            w3 => K27_CH0(3),  w4 => K27_CH0(4),  w5 => K27_CH0(5),
            w6 => K27_CH0(6),  w7 => K27_CH0(7),  w8 => K27_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y27_c0
        );

    dot_k27_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K27_CH1(0),  w1 => K27_CH1(1),  w2 => K27_CH1(2),
            w3 => K27_CH1(3),  w4 => K27_CH1(4),  w5 => K27_CH1(5),
            w6 => K27_CH1(6),  w7 => K27_CH1(7),  w8 => K27_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y27_c1
        );

    dot_k27_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K27_CH2(0),  w1 => K27_CH2(1),  w2 => K27_CH2(2),
            w3 => K27_CH2(3),  w4 => K27_CH2(4),  w5 => K27_CH2(5),
            w6 => K27_CH2(6),  w7 => K27_CH2(7),  w8 => K27_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y27_c2
        );

    -- ================================================================
    -- Kernel 28 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k28_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K28_CH0(0),  w1 => K28_CH0(1),  w2 => K28_CH0(2),
            w3 => K28_CH0(3),  w4 => K28_CH0(4),  w5 => K28_CH0(5),
            w6 => K28_CH0(6),  w7 => K28_CH0(7),  w8 => K28_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y28_c0
        );

    dot_k28_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K28_CH1(0),  w1 => K28_CH1(1),  w2 => K28_CH1(2),
            w3 => K28_CH1(3),  w4 => K28_CH1(4),  w5 => K28_CH1(5),
            w6 => K28_CH1(6),  w7 => K28_CH1(7),  w8 => K28_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y28_c1
        );

    dot_k28_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K28_CH2(0),  w1 => K28_CH2(1),  w2 => K28_CH2(2),
            w3 => K28_CH2(3),  w4 => K28_CH2(4),  w5 => K28_CH2(5),
            w6 => K28_CH2(6),  w7 => K28_CH2(7),  w8 => K28_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y28_c2
        );

    -- ================================================================
    -- Kernel 29 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k29_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K29_CH0(0),  w1 => K29_CH0(1),  w2 => K29_CH0(2),
            w3 => K29_CH0(3),  w4 => K29_CH0(4),  w5 => K29_CH0(5),
            w6 => K29_CH0(6),  w7 => K29_CH0(7),  w8 => K29_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y29_c0
        );

    dot_k29_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K29_CH1(0),  w1 => K29_CH1(1),  w2 => K29_CH1(2),
            w3 => K29_CH1(3),  w4 => K29_CH1(4),  w5 => K29_CH1(5),
            w6 => K29_CH1(6),  w7 => K29_CH1(7),  w8 => K29_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y29_c1
        );

    dot_k29_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K29_CH2(0),  w1 => K29_CH2(1),  w2 => K29_CH2(2),
            w3 => K29_CH2(3),  w4 => K29_CH2(4),  w5 => K29_CH2(5),
            w6 => K29_CH2(6),  w7 => K29_CH2(7),  w8 => K29_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y29_c2
        );

    -- ================================================================
    -- Kernel 30 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k30_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K30_CH0(0),  w1 => K30_CH0(1),  w2 => K30_CH0(2),
            w3 => K30_CH0(3),  w4 => K30_CH0(4),  w5 => K30_CH0(5),
            w6 => K30_CH0(6),  w7 => K30_CH0(7),  w8 => K30_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y30_c0
        );

    dot_k30_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K30_CH1(0),  w1 => K30_CH1(1),  w2 => K30_CH1(2),
            w3 => K30_CH1(3),  w4 => K30_CH1(4),  w5 => K30_CH1(5),
            w6 => K30_CH1(6),  w7 => K30_CH1(7),  w8 => K30_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y30_c1
        );

    dot_k30_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K30_CH2(0),  w1 => K30_CH2(1),  w2 => K30_CH2(2),
            w3 => K30_CH2(3),  w4 => K30_CH2(4),  w5 => K30_CH2(5),
            w6 => K30_CH2(6),  w7 => K30_CH2(7),  w8 => K30_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y30_c2
        );

    -- ================================================================
    -- Kernel 31 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as every other kernel.
    -- ================================================================
    dot_k31_c0 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K31_CH0(0),  w1 => K31_CH0(1),  w2 => K31_CH0(2),
            w3 => K31_CH0(3),  w4 => K31_CH0(4),  w5 => K31_CH0(5),
            w6 => K31_CH0(6),  w7 => K31_CH0(7),  w8 => K31_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y31_c0
        );

    dot_k31_c1 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K31_CH1(0),  w1 => K31_CH1(1),  w2 => K31_CH1(2),
            w3 => K31_CH1(3),  w4 => K31_CH1(4),  w5 => K31_CH1(5),
            w6 => K31_CH1(6),  w7 => K31_CH1(7),  w8 => K31_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y31_c1
        );

    dot_k31_c2 : entity work.conv3x3_dot_pipelined_dsp
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K31_CH2(0),  w1 => K31_CH2(1),  w2 => K31_CH2(2),
            w3 => K31_CH2(3),  w4 => K31_CH2(4),  w5 => K31_CH2(5),
            w6 => K31_CH2(6),  w7 => K31_CH2(7),  w8 => K31_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y31_c2
        );

    -- ================================================================
    -- Final registered summation stage, one per kernel.
    --
    -- Adds each kernel's three per-channel dot-product results and its
    -- own bias in one registered step, gated by the shared valid_dot.
    -- Same structure as the 8/16/24-output DSP-aware cells' final_sum,
    -- replicated 32x -- this is why the cell's overall latency (4 cycles)
    -- is unchanged from the 1/4/8/16/24-output cells.
    -- ================================================================
    final_sum : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                y0_r    <= (others => '0');
                y1_r    <= (others => '0');
                y2_r    <= (others => '0');
                y3_r    <= (others => '0');
                y4_r    <= (others => '0');
                y5_r    <= (others => '0');
                y6_r    <= (others => '0');
                y7_r    <= (others => '0');
                y8_r    <= (others => '0');
                y9_r    <= (others => '0');
                y10_r    <= (others => '0');
                y11_r    <= (others => '0');
                y12_r    <= (others => '0');
                y13_r    <= (others => '0');
                y14_r    <= (others => '0');
                y15_r    <= (others => '0');
                y16_r    <= (others => '0');
                y17_r    <= (others => '0');
                y18_r    <= (others => '0');
                y19_r    <= (others => '0');
                y20_r    <= (others => '0');
                y21_r    <= (others => '0');
                y22_r    <= (others => '0');
                y23_r    <= (others => '0');
                y24_r    <= (others => '0');
                y25_r    <= (others => '0');
                y26_r    <= (others => '0');
                y27_r    <= (others => '0');
                y28_r    <= (others => '0');
                y29_r    <= (others => '0');
                y30_r    <= (others => '0');
                y31_r    <= (others => '0');
                valid_r <= '0';
            else
                y0_r    <= K0_BIAS32 + y0_c0 + y0_c1 + y0_c2;
                y1_r    <= K1_BIAS32 + y1_c0 + y1_c1 + y1_c2;
                y2_r    <= K2_BIAS32 + y2_c0 + y2_c1 + y2_c2;
                y3_r    <= K3_BIAS32 + y3_c0 + y3_c1 + y3_c2;
                y4_r    <= K4_BIAS32 + y4_c0 + y4_c1 + y4_c2;
                y5_r    <= K5_BIAS32 + y5_c0 + y5_c1 + y5_c2;
                y6_r    <= K6_BIAS32 + y6_c0 + y6_c1 + y6_c2;
                y7_r    <= K7_BIAS32 + y7_c0 + y7_c1 + y7_c2;
                y8_r    <= K8_BIAS32 + y8_c0 + y8_c1 + y8_c2;
                y9_r    <= K9_BIAS32 + y9_c0 + y9_c1 + y9_c2;
                y10_r    <= K10_BIAS32 + y10_c0 + y10_c1 + y10_c2;
                y11_r    <= K11_BIAS32 + y11_c0 + y11_c1 + y11_c2;
                y12_r    <= K12_BIAS32 + y12_c0 + y12_c1 + y12_c2;
                y13_r    <= K13_BIAS32 + y13_c0 + y13_c1 + y13_c2;
                y14_r    <= K14_BIAS32 + y14_c0 + y14_c1 + y14_c2;
                y15_r    <= K15_BIAS32 + y15_c0 + y15_c1 + y15_c2;
                y16_r    <= K16_BIAS32 + y16_c0 + y16_c1 + y16_c2;
                y17_r    <= K17_BIAS32 + y17_c0 + y17_c1 + y17_c2;
                y18_r    <= K18_BIAS32 + y18_c0 + y18_c1 + y18_c2;
                y19_r    <= K19_BIAS32 + y19_c0 + y19_c1 + y19_c2;
                y20_r    <= K20_BIAS32 + y20_c0 + y20_c1 + y20_c2;
                y21_r    <= K21_BIAS32 + y21_c0 + y21_c1 + y21_c2;
                y22_r    <= K22_BIAS32 + y22_c0 + y22_c1 + y22_c2;
                y23_r    <= K23_BIAS32 + y23_c0 + y23_c1 + y23_c2;
                y24_r    <= K24_BIAS32 + y24_c0 + y24_c1 + y24_c2;
                y25_r    <= K25_BIAS32 + y25_c0 + y25_c1 + y25_c2;
                y26_r    <= K26_BIAS32 + y26_c0 + y26_c1 + y26_c2;
                y27_r    <= K27_BIAS32 + y27_c0 + y27_c1 + y27_c2;
                y28_r    <= K28_BIAS32 + y28_c0 + y28_c1 + y28_c2;
                y29_r    <= K29_BIAS32 + y29_c0 + y29_c1 + y29_c2;
                y30_r    <= K30_BIAS32 + y30_c0 + y30_c1 + y30_c2;
                y31_r    <= K31_BIAS32 + y31_c0 + y31_c1 + y31_c2;
                valid_r <= valid_dot;
            end if;
        end if;
    end process final_sum;

    valid_out <= valid_r;
    y0 <= y0_r;
    y1 <= y1_r;
    y2 <= y2_r;
    y3 <= y3_r;
    y4 <= y4_r;
    y5 <= y5_r;
    y6 <= y6_r;
    y7 <= y7_r;
    y8 <= y8_r;
    y9 <= y9_r;
    y10 <= y10_r;
    y11 <= y11_r;
    y12 <= y12_r;
    y13 <= y13_r;
    y14 <= y14_r;
    y15 <= y15_r;
    y16 <= y16_r;
    y17 <= y17_r;
    y18 <= y18_r;
    y19 <= y19_r;
    y20 <= y20_r;
    y21 <= y21_r;
    y22 <= y22_r;
    y23 <= y23_r;
    y24 <= y24_r;
    y25 <= y25_r;
    y26 <= y26_r;
    y27 <= y27_r;
    y28 <= y28_r;
    y29 <= y29_r;
    y30 <= y30_r;
    y31 <= y31_r;

end architecture rtl;
