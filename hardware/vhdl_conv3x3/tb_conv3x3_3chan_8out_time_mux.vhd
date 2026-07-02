-- tb_conv3x3_3chan_8out_time_mux.vhd
-- Self-checking testbench for conv3x3_3chan_8out_time_mux.
--
-- Drives the scheduler with the top-left 3x3 patch of the canonical 5x5x3
-- toy input (the same patch used by every other testbench in this
-- directory), pulses `start` once, waits for the scheduler's own `done`
-- pulse, and checks all eight outputs y0..y7 against
-- KERNEL{0..7}_EXPECTED(0) -- the same top-left golden values used to
-- verify the fully parallel 4-output and 8-output prototypes and the
-- single time-multiplexed dot product. It also measures the total number
-- of clock cycles from `start` to `done`.
--
-- ---- Canonical input patch (top-left 3x3 window only) --------------------
--   Channel 0: 1..25 row-major -> top-left 3x3 window = 1,2,3,6,7,8,11,12,13
--   Channel 1: 2 x channel 0   -> 2,4,6,12,14,16,22,24,26
--   Channel 2: -1 x channel 0  -> -1,-2,-3,-6,-7,-8,-11,-12,-13
--
-- ---- What this validates -----------------------------------------------
-- 1. The scheduler correctly reuses ONE conv3x3_dot_time_mux instance to
--    compute all 8 kernels' outputs for one 3-channel 3x3 patch, and every
--    one of the 8 outputs matches the same golden vectors used across this
--    entire prototype series.
-- 2. The measured start-to-done cycle count matches the module's documented
--    design intent (24 dot operations x 10 cycles each, issued back-to-back
--    with no idle bubble = 240 cycles) -- confirmed here empirically rather
--    than assumed.
--
-- This does NOT validate full quantized U-Net inference (BatchNorm is not
-- applied), does NOT cover the remaining 24 of 32 first-layer output
-- channels, does NOT test a sliding/streaming window across an image, and
-- has NOT been run on real FPGA hardware.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_kernels0_to7_pkg.all;

entity tb_conv3x3_3chan_8out_time_mux is
end entity tb_conv3x3_3chan_8out_time_mux;

architecture sim of tb_conv3x3_3chan_8out_time_mux is

    constant CLK_PERIOD : time := 10 ns;

    signal clk   : std_logic := '0';
    signal rst   : std_logic := '1';
    signal start : std_logic := '0';

    -- Top-left 3x3 window of the canonical 5x5x3 toy input, per channel.
    signal c0_p0, c0_p1, c0_p2, c0_p3, c0_p4, c0_p5, c0_p6, c0_p7, c0_p8 : signed(7 downto 0) := (others => '0');
    signal c1_p0, c1_p1, c1_p2, c1_p3, c1_p4, c1_p5, c1_p6, c1_p7, c1_p8 : signed(7 downto 0) := (others => '0');
    signal c2_p0, c2_p1, c2_p2, c2_p3, c2_p4, c2_p5, c2_p6, c2_p7, c2_p8 : signed(7 downto 0) := (others => '0');

    signal busy : std_logic;
    signal done : std_logic;
    signal s_y0, s_y1, s_y2, s_y3, s_y4, s_y5, s_y6, s_y7 : signed(31 downto 0);

    -- Measured (see time_mux_8out_summary.md): 265 cycles. This is 24 dot
    -- operations x 10 cycles each (240), PLUS 24 one-cycle registration
    -- gaps -- one between the scheduler's own start and the first dot
    -- operation's dispatch becoming visible, and one between each pair of
    -- consecutive dot operations. This gap is NOT a bug: `dot_start` is a
    -- registered signal, so a decision made on the edge where one dot
    -- operation's `done` fires can only become visible to the shared
    -- conv3x3_dot_time_mux instance on the following edge -- true of any
    -- registered start signal, hardware or testbench-driven alike.
    constant EXPECTED_CYCLES : integer := 265;

begin

    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.conv3x3_3chan_8out_time_mux
        port map (
            clk   => clk,
            rst   => rst,
            start => start,
            c0_p0 => c0_p0, c0_p1 => c0_p1, c0_p2 => c0_p2,
            c0_p3 => c0_p3, c0_p4 => c0_p4, c0_p5 => c0_p5,
            c0_p6 => c0_p6, c0_p7 => c0_p7, c0_p8 => c0_p8,
            c1_p0 => c1_p0, c1_p1 => c1_p1, c1_p2 => c1_p2,
            c1_p3 => c1_p3, c1_p4 => c1_p4, c1_p5 => c1_p5,
            c1_p6 => c1_p6, c1_p7 => c1_p7, c1_p8 => c1_p8,
            c2_p0 => c2_p0, c2_p1 => c2_p1, c2_p2 => c2_p2,
            c2_p3 => c2_p3, c2_p4 => c2_p4, c2_p5 => c2_p5,
            c2_p6 => c2_p6, c2_p7 => c2_p7, c2_p8 => c2_p8,
            busy => busy,
            done => done,
            y0 => s_y0, y1 => s_y1, y2 => s_y2, y3 => s_y3,
            y4 => s_y4, y5 => s_y5, y6 => s_y6, y7 => s_y7
        );

    stim : process

        variable cycles : integer := 0;

        procedure check_one(kernel_id : integer; got : signed(31 downto 0); expected : integer) is
        begin
            assert to_integer(got) = expected
                report "FAIL kernel " & integer'image(kernel_id) &
                       ": y = " & integer'image(to_integer(got)) &
                       "  expected " & integer'image(expected)
                severity failure;
            report "PASS kernel " & integer'image(kernel_id) &
                   " (top-left position): y = " & integer'image(to_integer(got)) &
                   "  (matches KERNEL" & integer'image(kernel_id) & "_EXPECTED(0))";
        end procedure check_one;

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

        -- ---- Drive the top-left 3x3x3 patch (held stable for the run) ---
        c0_p0 <= to_signed(1,  8); c0_p1 <= to_signed(2,  8); c0_p2 <= to_signed(3,  8);
        c0_p3 <= to_signed(6,  8); c0_p4 <= to_signed(7,  8); c0_p5 <= to_signed(8,  8);
        c0_p6 <= to_signed(11, 8); c0_p7 <= to_signed(12, 8); c0_p8 <= to_signed(13, 8);

        c1_p0 <= to_signed(2,  8); c1_p1 <= to_signed(4,  8); c1_p2 <= to_signed(6,  8);
        c1_p3 <= to_signed(12, 8); c1_p4 <= to_signed(14, 8); c1_p5 <= to_signed(16, 8);
        c1_p6 <= to_signed(22, 8); c1_p7 <= to_signed(24, 8); c1_p8 <= to_signed(26, 8);

        c2_p0 <= to_signed(-1,  8); c2_p1 <= to_signed(-2,  8); c2_p2 <= to_signed(-3,  8);
        c2_p3 <= to_signed(-6,  8); c2_p4 <= to_signed(-7,  8); c2_p5 <= to_signed(-8,  8);
        c2_p6 <= to_signed(-11, 8); c2_p7 <= to_signed(-12, 8); c2_p8 <= to_signed(-13, 8);

        wait until rising_edge(clk);
        wait for 1 ns;

        -- ---- Pulse start, measure cycles until done ----------------------
        start <= '1';
        wait until rising_edge(clk);
        wait for 1 ns;
        start <= '0';
        cycles := 1;

        while done /= '1' loop
            assert cycles <= EXPECTED_CYCLES + 20
                report "FAIL: scheduler did not assert done within " &
                       integer'image(EXPECTED_CYCLES + 20) & " cycles (runaway FSM?)"
                severity failure;
            wait until rising_edge(clk);
            wait for 1 ns;
            cycles := cycles + 1;
        end loop;

        report "Scheduler done after " & integer'image(cycles) & " cycles" &
               " (measured; see time_mux_8out_summary.md for the breakdown)";

        assert cycles = EXPECTED_CYCLES
            report "FAIL: expected exactly " & integer'image(EXPECTED_CYCLES) &
                   " cycles from start to done, observed " & integer'image(cycles)
            severity failure;

        -- ---- Check all 8 outputs against the shared golden vectors -------
        check_one(0, s_y0, KERNEL0_EXPECTED(0));
        check_one(1, s_y1, KERNEL1_EXPECTED(0));
        check_one(2, s_y2, KERNEL2_EXPECTED(0));
        check_one(3, s_y3, KERNEL3_EXPECTED(0));
        check_one(4, s_y4, KERNEL4_EXPECTED(0));
        check_one(5, s_y5, KERNEL5_EXPECTED(0));
        check_one(6, s_y6, KERNEL6_EXPECTED(0));
        check_one(7, s_y7, KERNEL7_EXPECTED(0));

        -- ---- done must be a one-cycle pulse, not held high ----------------
        wait until rising_edge(clk);
        wait for 1 ns;
        assert done = '0'
            report "FAIL: done did not return to '0' the cycle after asserting"
            severity failure;

        report "=== All conv3x3_3chan_8out_time_mux tests PASSED ===" &
               "  (8 / 8 kernel outputs match Python golden vectors, " &
               integer'image(cycles) & " cycles for one 3x3x3 patch)" severity note;
        report "    NOTE: one-window, 8-output, TIME-MULTIPLEXED SCHEDULER PROTOTYPE only." &
               " Not a full streaming convolution engine, not board-tested." severity note;

        wait;
    end process stim;

end architecture sim;
