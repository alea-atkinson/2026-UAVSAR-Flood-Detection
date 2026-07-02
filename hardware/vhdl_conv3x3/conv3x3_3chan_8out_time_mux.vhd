-- conv3x3_3chan_8out_time_mux.vhd
-- Time-multiplexed 8-output scheduler for ONE 3-channel 3x3 spatial window.
--
-- This is NOT a full streaming convolution engine and NOT full U-Net
-- inference. It computes all EIGHT first-layer output-channel values
-- (kernels 0-7 of enc1.block.0.weight) for a SINGLE, already-captured
-- 3-channel 3x3 patch, by reusing ONE conv3x3_dot_time_mux instance
-- sequentially across the 24 (kernel, channel) combinations needed
-- (8 kernels x 3 input channels), instead of instantiating 24 parallel
-- dot-product units as stream_conv3x3_3chan_8out_cell.vhd does.
--
-- Motivation: this is the natural next step after conv3x3_dot_time_mux.vhd
-- (a single time-multiplexed dot product, driven by testbench-side
-- orchestration only). Here, the orchestration moves INTO hardware: a
-- small FSM schedules the 24 dot operations, muxes in the correct pixel
-- channel and kernel weights for each one, accumulates each kernel's three
-- per-channel results, and reports y0..y7 with a single done pulse.
--
-- ---- Scheduling order ----------------------------------------------------
-- step = kernel_idx * 3 + channel_idx, for step = 0 .. 23:
--   step  0: kernel 0, channel 0        step 12: kernel 4, channel 0
--   step  1: kernel 0, channel 1        step 13: kernel 4, channel 1
--   step  2: kernel 0, channel 2 -> y0  step 14: kernel 4, channel 2 -> y4
--   step  3: kernel 1, channel 0        step 15: kernel 5, channel 0
--   step  4: kernel 1, channel 1        step 16: kernel 5, channel 1
--   step  5: kernel 1, channel 2 -> y1  step 17: kernel 5, channel 2 -> y5
--   step  6: kernel 2, channel 0        step 18: kernel 6, channel 0
--   step  7: kernel 2, channel 1        step 19: kernel 6, channel 1
--   step  8: kernel 2, channel 2 -> y2  step 20: kernel 6, channel 2 -> y6
--   step  9: kernel 3, channel 0        step 21: kernel 7, channel 0
--   step 10: kernel 3, channel 1        step 22: kernel 7, channel 1
--   step 11: kernel 3, channel 2 -> y3  step 23: kernel 7, channel 2 -> y7
--
-- After the third channel of each kernel, that kernel's three per-channel
-- dot-product results are summed together with the kernel's bias (0 for
-- every kernel here; Conv2d has no direct bias, BatchNorm is NOT folded)
-- and latched into y{kernel_idx}.
--
-- ---- Interface -----------------------------------------------------------
-- Start/done handshake, one shot per patch:
--   - While idle, asserting `start` for one cycle begins scheduling all 24
--     dot operations against the currently-presented 3x3x3 patch
--     (c0_p0..c0_p8, c1_p0..c1_p8, c2_p0..c2_p8). The patch must remain
--     stable for the entire run (it is NOT re-latched per step; only the
--     conv3x3_dot_time_mux submodule latches its own p/w inputs on each
--     of its own internal start pulses).
--   - `done` pulses for exactly one cycle when y0..y7 are all valid.
--   - `start` is ignored while busy.
--
-- ---- Weight source ---------------------------------------------------
-- Kernels 0-7 (INT8 weights, symmetric per-tensor quantization) are taken
-- directly from hardware/vhdl_conv3x3/first_layer_kernels0_to7_pkg.vhd, the
-- same package used by stream_conv3x3_3chan_8out_cell.vhd. Weights are not
-- exposed as ports; this module hardwires all 8 kernels internally.
--
-- ---- Cycle count -----------------------------------------------------
-- conv3x3_dot_time_mux takes 10 clock cycles per dot product (see
-- conv3x3_dot_time_mux.vhd and time_mux_dot_summary.md). This scheduler's
-- FSM makes the decision to redispatch the next dot operation on the SAME
-- edge it observes the previous operation's done pulse, but `dot_start` is
-- a REGISTERED signal, so that decision only becomes visible to the shared
-- conv3x3_dot_time_mux instance on the FOLLOWING edge -- an unavoidable
-- one-cycle registration gap between every pair of consecutive dot
-- operations (this is true of any registered start signal, not a bug or an
-- oversight specific to this FSM). Measured by
-- tb_conv3x3_3chan_8out_time_mux.vhd (which counts clock edges between
-- start and done directly): 265 cycles total = 24 dot operations x 10
-- cycles (240) + 24 one-cycle registration gaps (one before the first dot
-- operation's dispatch becomes visible, and one between each subsequent
-- pair). See time_mux_8out_summary.md for the full breakdown.
--
-- ---- Scope ----------------------------------------------------------------
-- This module computes ONE 3-channel 3x3 spatial window's 8 output values.
-- It does NOT slide a window across an image, does NOT stream pixels, does
-- NOT implement padding, activation, or BatchNorm, does NOT cover all 32
-- first-layer output channels, and has NOT been run on real FPGA hardware.
-- It is a scheduler PROTOTYPE demonstrating that a single time-multiplexed
-- MAC lane can be reused across multiple kernels and channels under
-- hardware control, not a claim of measured speedup or a complete
-- time-multiplexed convolution accelerator.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_kernels0_to7_pkg.all;

entity conv3x3_3chan_8out_time_mux is
    port (
        clk   : in  std_logic;
        rst   : in  std_logic;             -- synchronous reset, active high
        start : in  std_logic;             -- pulse (while idle) to begin scheduling

        -- One 3-channel 3x3 patch (row-major: p0 p1 p2 / p3 p4 p5 / p6 p7 p8),
        -- held stable for the duration of one run.
        c0_p0 : in signed(7 downto 0);  c0_p1 : in signed(7 downto 0);  c0_p2 : in signed(7 downto 0);
        c0_p3 : in signed(7 downto 0);  c0_p4 : in signed(7 downto 0);  c0_p5 : in signed(7 downto 0);
        c0_p6 : in signed(7 downto 0);  c0_p7 : in signed(7 downto 0);  c0_p8 : in signed(7 downto 0);

        c1_p0 : in signed(7 downto 0);  c1_p1 : in signed(7 downto 0);  c1_p2 : in signed(7 downto 0);
        c1_p3 : in signed(7 downto 0);  c1_p4 : in signed(7 downto 0);  c1_p5 : in signed(7 downto 0);
        c1_p6 : in signed(7 downto 0);  c1_p7 : in signed(7 downto 0);  c1_p8 : in signed(7 downto 0);

        c2_p0 : in signed(7 downto 0);  c2_p1 : in signed(7 downto 0);  c2_p2 : in signed(7 downto 0);
        c2_p3 : in signed(7 downto 0);  c2_p4 : in signed(7 downto 0);  c2_p5 : in signed(7 downto 0);
        c2_p6 : in signed(7 downto 0);  c2_p7 : in signed(7 downto 0);  c2_p8 : in signed(7 downto 0);

        busy : out std_logic;              -- '1' while scheduling is in progress
        done : out std_logic;              -- one-cycle pulse when y0..y7 are all valid
        y0 : out signed(31 downto 0);
        y1 : out signed(31 downto 0);
        y2 : out signed(31 downto 0);
        y3 : out signed(31 downto 0);
        y4 : out signed(31 downto 0);
        y5 : out signed(31 downto 0);
        y6 : out signed(31 downto 0);
        y7 : out signed(31 downto 0)
    );
end entity conv3x3_3chan_8out_time_mux;

architecture rtl of conv3x3_3chan_8out_time_mux is

    -- ----------------------------------------------------------------
    -- Convert the package's plain-integer kernel constants to
    -- signed(7 downto 0) arrays, once, at elaboration time (same helper
    -- as used by stream_conv3x3_3chan_8out_cell.vhd).
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

    -- ----------------------------------------------------------------
    -- Weight LUT: 24 entries, one per (kernel, channel) pair, indexed by
    -- step = kernel_idx*3 + channel_idx. Built once at elaboration time.
    -- ----------------------------------------------------------------
    type weight_lut_t is array (0 to 23) of signed8_kernel_t;

    function build_weight_lut return weight_lut_t is
        variable lut : weight_lut_t;
    begin
        lut(0)  := to_signed_kernel(KERNEL0_CH0_W);
        lut(1)  := to_signed_kernel(KERNEL0_CH1_W);
        lut(2)  := to_signed_kernel(KERNEL0_CH2_W);
        lut(3)  := to_signed_kernel(KERNEL1_CH0_W);
        lut(4)  := to_signed_kernel(KERNEL1_CH1_W);
        lut(5)  := to_signed_kernel(KERNEL1_CH2_W);
        lut(6)  := to_signed_kernel(KERNEL2_CH0_W);
        lut(7)  := to_signed_kernel(KERNEL2_CH1_W);
        lut(8)  := to_signed_kernel(KERNEL2_CH2_W);
        lut(9)  := to_signed_kernel(KERNEL3_CH0_W);
        lut(10) := to_signed_kernel(KERNEL3_CH1_W);
        lut(11) := to_signed_kernel(KERNEL3_CH2_W);
        lut(12) := to_signed_kernel(KERNEL4_CH0_W);
        lut(13) := to_signed_kernel(KERNEL4_CH1_W);
        lut(14) := to_signed_kernel(KERNEL4_CH2_W);
        lut(15) := to_signed_kernel(KERNEL5_CH0_W);
        lut(16) := to_signed_kernel(KERNEL5_CH1_W);
        lut(17) := to_signed_kernel(KERNEL5_CH2_W);
        lut(18) := to_signed_kernel(KERNEL6_CH0_W);
        lut(19) := to_signed_kernel(KERNEL6_CH1_W);
        lut(20) := to_signed_kernel(KERNEL6_CH2_W);
        lut(21) := to_signed_kernel(KERNEL7_CH0_W);
        lut(22) := to_signed_kernel(KERNEL7_CH1_W);
        lut(23) := to_signed_kernel(KERNEL7_CH2_W);
        return lut;
    end function build_weight_lut;

    constant WEIGHT_LUT : weight_lut_t := build_weight_lut;

    -- ----------------------------------------------------------------
    -- Bias LUT: one INT32 bias per kernel, added once after summing that
    -- kernel's three per-channel dot products (all 0 here; Conv2d has no
    -- direct bias, BatchNorm is NOT folded).
    -- ----------------------------------------------------------------
    type bias_lut_t is array (0 to 7) of signed(31 downto 0);

    function build_bias_lut return bias_lut_t is
        variable lut : bias_lut_t;
    begin
        lut(0) := to_signed(KERNEL0_BIAS, 32);
        lut(1) := to_signed(KERNEL1_BIAS, 32);
        lut(2) := to_signed(KERNEL2_BIAS, 32);
        lut(3) := to_signed(KERNEL3_BIAS, 32);
        lut(4) := to_signed(KERNEL4_BIAS, 32);
        lut(5) := to_signed(KERNEL5_BIAS, 32);
        lut(6) := to_signed(KERNEL6_BIAS, 32);
        lut(7) := to_signed(KERNEL7_BIAS, 32);
        return lut;
    end function build_bias_lut;

    constant BIAS_LUT : bias_lut_t := build_bias_lut;

    -- ----------------------------------------------------------------
    -- Per-channel pixel windows, packed into arrays for the mux below.
    -- ----------------------------------------------------------------
    signal ch0_arr, ch1_arr, ch2_arr : signed8_kernel_t;

    -- ----------------------------------------------------------------
    -- Scheduler FSM
    -- ----------------------------------------------------------------
    type state_t is (S_IDLE, S_RUN);
    signal state : state_t := S_IDLE;

    signal step        : integer range 0 to 23 := 0;
    signal channel_idx : integer range 0 to 2  := 0;   -- step mod 3
    signal kernel_idx  : integer range 0 to 7  := 0;   -- step / 3

    signal partial_sum : signed(31 downto 0) := (others => '0');

    type y_arr_t is array (0 to 7) of signed(31 downto 0);
    signal y_regs : y_arr_t := (others => (others => '0'));

    signal done_r : std_logic := '0';

    -- ----------------------------------------------------------------
    -- Single shared conv3x3_dot_time_mux instance and its interconnect.
    -- ----------------------------------------------------------------
    signal dot_start : std_logic := '0';
    signal dot_busy  : std_logic;
    signal dot_done  : std_logic;
    signal dot_y     : signed(31 downto 0);

    signal dot_p_arr : signed8_kernel_t;
    signal dot_w_arr : signed8_kernel_t;

    signal dot_p0, dot_p1, dot_p2, dot_p3, dot_p4, dot_p5, dot_p6, dot_p7, dot_p8 : signed(7 downto 0);
    signal dot_w0, dot_w1, dot_w2, dot_w3, dot_w4, dot_w5, dot_w6, dot_w7, dot_w8 : signed(7 downto 0);

    constant ZERO32 : signed(31 downto 0) := (others => '0');

begin

    -- ================================================================
    -- Pack input ports into arrays and derive step -> (kernel, channel).
    -- ================================================================
    ch0_arr(0) <= c0_p0; ch0_arr(1) <= c0_p1; ch0_arr(2) <= c0_p2;
    ch0_arr(3) <= c0_p3; ch0_arr(4) <= c0_p4; ch0_arr(5) <= c0_p5;
    ch0_arr(6) <= c0_p6; ch0_arr(7) <= c0_p7; ch0_arr(8) <= c0_p8;

    ch1_arr(0) <= c1_p0; ch1_arr(1) <= c1_p1; ch1_arr(2) <= c1_p2;
    ch1_arr(3) <= c1_p3; ch1_arr(4) <= c1_p4; ch1_arr(5) <= c1_p5;
    ch1_arr(6) <= c1_p6; ch1_arr(7) <= c1_p7; ch1_arr(8) <= c1_p8;

    ch2_arr(0) <= c2_p0; ch2_arr(1) <= c2_p1; ch2_arr(2) <= c2_p2;
    ch2_arr(3) <= c2_p3; ch2_arr(4) <= c2_p4; ch2_arr(5) <= c2_p5;
    ch2_arr(6) <= c2_p6; ch2_arr(7) <= c2_p7; ch2_arr(8) <= c2_p8;

    channel_idx <= step mod 3;
    kernel_idx  <= step / 3;

    -- ================================================================
    -- Combinational muxes: select this step's pixel channel and weight
    -- set for the shared conv3x3_dot_time_mux instance.
    -- ================================================================
    dot_p_arr <= ch0_arr when channel_idx = 0 else
                 ch1_arr when channel_idx = 1 else
                 ch2_arr;

    dot_p0 <= dot_p_arr(0); dot_p1 <= dot_p_arr(1); dot_p2 <= dot_p_arr(2);
    dot_p3 <= dot_p_arr(3); dot_p4 <= dot_p_arr(4); dot_p5 <= dot_p_arr(5);
    dot_p6 <= dot_p_arr(6); dot_p7 <= dot_p_arr(7); dot_p8 <= dot_p_arr(8);

    dot_w_arr <= WEIGHT_LUT(step);

    dot_w0 <= dot_w_arr(0); dot_w1 <= dot_w_arr(1); dot_w2 <= dot_w_arr(2);
    dot_w3 <= dot_w_arr(3); dot_w4 <= dot_w_arr(4); dot_w5 <= dot_w_arr(5);
    dot_w6 <= dot_w_arr(6); dot_w7 <= dot_w_arr(7); dot_w8 <= dot_w_arr(8);

    -- ================================================================
    -- The ONE reusable time-multiplexed dot-product lane, shared across
    -- all 24 (kernel, channel) operations.
    -- ================================================================
    dot_inst : entity work.conv3x3_dot_time_mux
        port map (
            clk   => clk,
            rst   => rst,
            start => dot_start,
            p0 => dot_p0, p1 => dot_p1, p2 => dot_p2,
            p3 => dot_p3, p4 => dot_p4, p5 => dot_p5,
            p6 => dot_p6, p7 => dot_p7, p8 => dot_p8,
            w0 => dot_w0, w1 => dot_w1, w2 => dot_w2,
            w3 => dot_w3, w4 => dot_w4, w5 => dot_w5,
            w6 => dot_w6, w7 => dot_w7, w8 => dot_w8,
            bias  => ZERO32,   -- per-channel bias always 0; kernel bias added below
            busy  => dot_busy,
            done  => dot_done,
            y     => dot_y
        );

    -- ================================================================
    -- Scheduler FSM: decides to redispatch the next dot operation on the
    -- same edge the previous one's done is observed (that decision becomes
    -- visible to conv3x3_dot_time_mux one cycle later, since dot_start is
    -- registered -- see the "Cycle count" header comment), accumulates
    -- each kernel's 3 channel results, adds that kernel's bias once every
    -- 3rd (channel_idx = 2) completion, and asserts the top-level done
    -- pulse after the 24th (step = 23) completion.
    -- ================================================================
    main : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state       <= S_IDLE;
                step        <= 0;
                partial_sum <= (others => '0');
                y_regs      <= (others => (others => '0'));
                done_r      <= '0';
                dot_start   <= '0';

            else
                done_r <= '0';

                case state is
                    when S_IDLE =>
                        dot_start <= '0';
                        if start = '1' then
                            step        <= 0;
                            partial_sum <= (others => '0');
                            dot_start   <= '1';
                            state       <= S_RUN;
                        end if;

                    when S_RUN =>
                        -- De-assert the one-cycle start pulse issued last edge.
                        if dot_start = '1' then
                            dot_start <= '0';
                        end if;

                        if dot_done = '1' then
                            if channel_idx = 2 then
                                y_regs(kernel_idx) <= partial_sum + resize(dot_y, 32) + BIAS_LUT(kernel_idx);
                                partial_sum        <= (others => '0');
                            else
                                partial_sum <= partial_sum + resize(dot_y, 32);
                            end if;

                            if step = 23 then
                                done_r <= '1';
                                state  <= S_IDLE;
                            else
                                step      <= step + 1;
                                dot_start <= '1';
                            end if;
                        end if;
                end case;
            end if;
        end if;
    end process main;

    busy <= '0' when state = S_IDLE else '1';
    done <= done_r;

    y0 <= y_regs(0);
    y1 <= y_regs(1);
    y2 <= y_regs(2);
    y3 <= y_regs(3);
    y4 <= y_regs(4);
    y5 <= y_regs(5);
    y6 <= y_regs(6);
    y7 <= y_regs(7);

end architecture rtl;
