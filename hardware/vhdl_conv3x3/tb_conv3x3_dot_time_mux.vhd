-- tb_conv3x3_dot_time_mux.vhd
-- Self-checking testbench for conv3x3_dot_time_mux.
--
-- conv3x3_dot_time_mux computes ONE input channel's 3x3 dot product at a
-- time (one MAC lane, 9 cycles). The existing package constants
-- KERNEL{k}_EXPECTED hold the FULL 3-input-channel accumulated output (the
-- same convention used by every other testbench in this directory, e.g.
-- tb_stream_conv3x3_3chan_8out_cell.vhd), so this testbench runs the DUT
-- three times per kernel -- once per input channel, with that channel's
-- weights and pixel window -- and sums the three time-multiplexed results
-- (plus the kernel's bias, which is 0 for every kernel here) to reproduce
-- KERNEL{k}_EXPECTED(0), the golden value for output position (row 0, col 0)
-- i.e. the top-left 3x3 window of the canonical 5x5 toy image.
--
-- ---- Canonical input patch (top-left 3x3 window only) --------------------
-- Matching every other testbench in this directory:
--   Channel 0: 1..25 row-major -> top-left 3x3 window = 1,2,3,6,7,8,11,12,13
--   Channel 1: 2 x channel 0   -> 2,4,6,12,14,16,22,24,26
--   Channel 2: -1 x channel 0  -> -1,-2,-3,-6,-7,-8,-11,-12,-13
--
-- ---- What this validates -----------------------------------------------
-- 1. Kernels 0, 1, and 7's full (3-channel) dot product at the top-left
--    position, computed by running the time-multiplexed single-channel DUT
--    three times per kernel and summing in the testbench, matches the SAME
--    golden INT32 values (KERNEL{0,1,7}_EXPECTED(0)) used to verify the
--    fully parallel 4-output and 8-output prototypes.
-- 2. The measured cycle count from asserting `start` to observing `done`
--    is exactly 10 cycles for every run, matching the module's documented
--    latency (1 load cycle + 9 tap-processing cycles).
--
-- This does NOT validate full quantized U-Net inference (BatchNorm is not
-- applied), does NOT cover the remaining 24 of 32 first-layer output
-- channels, and has NOT been run on real FPGA hardware. It also does NOT
-- test a time-multiplexed multi-kernel scheduler -- only a single reusable
-- dot-product primitive run sequentially by testbench-side orchestration.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_kernels0_to7_pkg.all;

entity tb_conv3x3_dot_time_mux is
end entity tb_conv3x3_dot_time_mux;

architecture sim of tb_conv3x3_dot_time_mux is

    constant CLK_PERIOD : time := 10 ns;

    signal clk   : std_logic := '0';
    signal rst   : std_logic := '1';
    signal start : std_logic := '0';

    signal p0, p1, p2, p3, p4, p5, p6, p7, p8 : signed(7 downto 0) := (others => '0');
    signal w0, w1, w2, w3, w4, w5, w6, w7, w8 : signed(7 downto 0) := (others => '0');
    signal s_bias : signed(31 downto 0) := (others => '0');

    signal busy : std_logic;
    signal done : std_logic;
    signal s_y  : signed(31 downto 0);

    -- Top-left 3x3 window of the canonical 5x5x3 toy input, per channel.
    -- Reuses int8_kernel_t (a 9-element integer array, range -128..127)
    -- from the package purely as a convenient 9-element container type --
    -- these are pixel values, not kernel weights.
    constant PIX_C0 : int8_kernel_t := (1, 2, 3, 6, 7, 8, 11, 12, 13);
    constant PIX_C1 : int8_kernel_t := (2, 4, 6, 12, 14, 16, 22, 24, 26);
    constant PIX_C2 : int8_kernel_t := (-1, -2, -3, -6, -7, -8, -11, -12, -13);

    constant EXPECTED_CYCLES : integer := 10;

begin

    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.conv3x3_dot_time_mux
        port map (
            clk   => clk,
            rst   => rst,
            start => start,
            p0 => p0, p1 => p1, p2 => p2,
            p3 => p3, p4 => p4, p5 => p5,
            p6 => p6, p7 => p7, p8 => p8,
            w0 => w0, w1 => w1, w2 => w2,
            w3 => w3, w4 => w4, w5 => w5,
            w6 => w6, w7 => w7, w8 => w8,
            bias => s_bias,
            busy => busy,
            done => done,
            y    => s_y
        );

    stim : process

        -- Runs the time-multiplexed DUT once on one channel's pixels/weights
        -- (zero bias -- the kernel's own bias is added once after summing
        -- all three channels, mirroring the parallel cells' convention).
        -- Returns the INT32 result and the measured start-to-done cycle count.
        procedure run_one_channel(
            ch_w        : in  int8_kernel_t;
            ch_p        : in  int8_kernel_t;
            result      : out signed(31 downto 0);
            cycle_count : out integer
        ) is
            variable cycles : integer := 0;
        begin
            p0 <= to_signed(ch_p(0), 8); p1 <= to_signed(ch_p(1), 8); p2 <= to_signed(ch_p(2), 8);
            p3 <= to_signed(ch_p(3), 8); p4 <= to_signed(ch_p(4), 8); p5 <= to_signed(ch_p(5), 8);
            p6 <= to_signed(ch_p(6), 8); p7 <= to_signed(ch_p(7), 8); p8 <= to_signed(ch_p(8), 8);

            w0 <= to_signed(ch_w(0), 8); w1 <= to_signed(ch_w(1), 8); w2 <= to_signed(ch_w(2), 8);
            w3 <= to_signed(ch_w(3), 8); w4 <= to_signed(ch_w(4), 8); w5 <= to_signed(ch_w(5), 8);
            w6 <= to_signed(ch_w(6), 8); w7 <= to_signed(ch_w(7), 8); w8 <= to_signed(ch_w(8), 8);

            s_bias <= (others => '0');

            start <= '1';
            wait until rising_edge(clk);
            wait for 1 ns;
            start <= '0';
            cycles := 1;

            while done /= '1' loop
                wait until rising_edge(clk);
                wait for 1 ns;
                cycles := cycles + 1;
            end loop;

            result      := s_y;
            cycle_count := cycles;
        end procedure run_one_channel;

        -- Runs all three input channels for one kernel, sums them with the
        -- kernel's bias, and checks against KERNEL{k}_EXPECTED(0) -- the
        -- top-left (row 0, col 0) golden output shared with every other
        -- prototype in this directory.
        procedure check_kernel(
            kernel_id : integer;
            ch0_w     : int8_kernel_t;
            ch1_w     : int8_kernel_t;
            ch2_w     : int8_kernel_t;
            kbias     : integer;
            expected0 : integer
        ) is
            variable y_ch0, y_ch1, y_ch2 : signed(31 downto 0);
            variable c_ch0, c_ch1, c_ch2 : integer;
            variable total : integer;
        begin
            run_one_channel(ch0_w, PIX_C0, y_ch0, c_ch0);
            run_one_channel(ch1_w, PIX_C1, y_ch1, c_ch1);
            run_one_channel(ch2_w, PIX_C2, y_ch2, c_ch2);

            total := to_integer(y_ch0) + to_integer(y_ch1) + to_integer(y_ch2) + kbias;

            report "Kernel " & integer'image(kernel_id) &
                   ": channel dot products y_ch0=" & integer'image(to_integer(y_ch0)) &
                   " y_ch1=" & integer'image(to_integer(y_ch1)) &
                   " y_ch2=" & integer'image(to_integer(y_ch2)) &
                   "  cycles(ch0/ch1/ch2)=" & integer'image(c_ch0) & "/" &
                   integer'image(c_ch1) & "/" & integer'image(c_ch2);

            assert c_ch0 = EXPECTED_CYCLES and c_ch1 = EXPECTED_CYCLES and c_ch2 = EXPECTED_CYCLES
                report "FAIL kernel " & integer'image(kernel_id) &
                       ": expected " & integer'image(EXPECTED_CYCLES) &
                       " cycles per dot product, observed " &
                       integer'image(c_ch0) & "/" & integer'image(c_ch1) & "/" & integer'image(c_ch2)
                severity failure;

            assert total = expected0
                report "FAIL kernel " & integer'image(kernel_id) &
                       " (top-left position): summed y = " & integer'image(total) &
                       "  expected " & integer'image(expected0)
                severity failure;

            report "PASS kernel " & integer'image(kernel_id) &
                   " (top-left position): summed y = " & integer'image(total) &
                   "  (matches KERNEL" & integer'image(kernel_id) & "_EXPECTED(0) = " &
                   integer'image(expected0) & "), " &
                   integer'image(EXPECTED_CYCLES) & " cycles per channel dot product";
        end procedure check_kernel;

    begin
        -- ---- Reset (3 clocks) ----------------------------------------
        rst   <= '1';
        start <= '0';
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst <= '0';
        wait until rising_edge(clk);
        wait for 1 ns;

        -- ---- Kernel 0 ---------------------------------------------------
        check_kernel(0, KERNEL0_CH0_W, KERNEL0_CH1_W, KERNEL0_CH2_W,
                     KERNEL0_BIAS, KERNEL0_EXPECTED(0));

        -- ---- Kernel 1 ---------------------------------------------------
        check_kernel(1, KERNEL1_CH0_W, KERNEL1_CH1_W, KERNEL1_CH2_W,
                     KERNEL1_BIAS, KERNEL1_EXPECTED(0));

        -- ---- Kernel 7 ---------------------------------------------------
        check_kernel(7, KERNEL7_CH0_W, KERNEL7_CH1_W, KERNEL7_CH2_W,
                     KERNEL7_BIAS, KERNEL7_EXPECTED(0));

        report "=== All conv3x3_dot_time_mux tests PASSED ===" &
               "  (3 kernels x 3 channel dot products = 9 / 9 checks, " &
               "each " & integer'image(EXPECTED_CYCLES) & " cycles, " &
               "top-left position matches Python golden vectors)" severity note;
        report "    NOTE: single-lane time-multiplexed DOT-PRODUCT PROTOTYPE only." &
               " Not a full time-multiplexed convolution engine, not board-tested." severity note;

        wait;
    end process stim;

end architecture sim;
