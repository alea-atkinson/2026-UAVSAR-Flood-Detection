-- tb_stream_conv3x3_3chan_8out_time_mux.vhd
-- Self-checking testbench for stream_conv3x3_3chan_8out_time_mux.
--
-- This is a STREAMING TIME-MULTIPLEXED PROTOTYPE testbench, not full U-Net
-- inference and not a board demo. It streams the canonical 5x5x3 toy image
-- into the DUT exactly as tb_stream_conv3x3_3chan_8out_cell.vhd streams it
-- into the fully parallel 8-output cell, then waits (with no back-pressure
-- or overlap) for the DUT's internal one-window scheduler to sequentially
-- process all 9 buffered windows, checking each window's 8 kernel outputs
-- as they appear.
--
-- ---- Input patch --------------------------------------------------------
-- Canonical 5x5x3 toy patch matching every other testbench in this directory:
--   Channel 0: 1..25 row-major
--   Channel 1: 2 x channel 0  (2..50)
--   Channel 2: -1 x channel 0  (-1..-25)
--
-- ---- Expected INT32 outputs (3x3 valid convolution, row-major) ----------
-- Golden vectors sourced from first_layer_kernels0_to7_pkg's
-- KERNEL{0..7}_EXPECTED constants -- the SAME golden values used by
-- tb_stream_conv3x3_3chan_8out_cell.vhd (fully parallel) and
-- tb_conv3x3_3chan_8out_time_mux.vhd (one-window scheduler).
--
-- ---- Timing behavior (two non-overlapped phases) -------------------------
-- Phase 1 (capture): pixels stream in at 1/clock for ~25 clocks, exactly
-- like the parallel cell's testbench. No valid_out pulses occur during
-- this phase (by design -- the DUT only starts phase 2 once all 9 windows
-- are captured).
-- Phase 2 (process): the DUT's internal scheduler processes the 9 buffered
-- windows ONE AT A TIME (no overlap), taking roughly 265 cycles per window
-- (see time_mux_8out_summary.md for that per-window figure). This
-- testbench does NOT assert an exact precomputed total cycle count for the
-- whole streaming run -- it measures and reports the actual total directly,
-- with only a generous safety bound to catch a runaway FSM.
--
-- ---- What this validates -----------------------------------------------
-- This testbench confirms that stream_conv3x3_3chan_8out_time_mux, built by
-- wrapping the UNMODIFIED window3x3_stream generators and the UNMODIFIED
-- conv3x3_3chan_8out_time_mux scheduler in a simple two-phase
-- capture-then-process FSM, reproduces the same 9x8=72 INT32 values used to
-- verify the fully parallel 8-output prototype. It does NOT validate full
-- quantized U-Net inference (BatchNorm is not applied), does NOT cover the
-- remaining 24 of 32 first-layer output channels, does NOT overlap window
-- capture with compute, does NOT implement backpressure into
-- window3x3_stream, and has NOT been run on real FPGA hardware.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_kernels0_to7_pkg.all;

entity tb_stream_conv3x3_3chan_8out_time_mux is
end entity tb_stream_conv3x3_3chan_8out_time_mux;

architecture sim of tb_stream_conv3x3_3chan_8out_time_mux is

    constant CLK_PERIOD : time     := 10 ns;
    constant IMG_W      : positive := 5;
    constant NUM_WINDOWS : integer := 9;   -- (IMG_W - 2) * (IMG_W - 2)

    -- A generous sanity bound only, NOT an asserted exact cycle count:
    -- ~25 cycles to stream the image + 9 windows x (up to ~300 cycles each,
    -- padded well above the ~265-cycle-per-window figure measured for the
    -- one-window scheduler) to catch a genuinely runaway FSM, not to pin an
    -- exact timing model.
    constant MAX_CYCLES : integer := 25 + NUM_WINDOWS * 300;

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    signal valid_out : std_logic;
    signal all_done  : std_logic;
    signal s_y0, s_y1, s_y2, s_y3, s_y4, s_y5, s_y6, s_y7 : signed(31 downto 0);

begin

    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.stream_conv3x3_3chan_8out_time_mux
        generic map (IMG_WIDTH => IMG_W)
        port map (
            clk      => clk,
            rst      => rst,
            valid_in => valid_in,
            pixel_c0 => pixel_c0,
            pixel_c1 => pixel_c1,
            pixel_c2 => pixel_c2,
            valid_out => valid_out,
            y0 => s_y0, y1 => s_y1, y2 => s_y2, y3 => s_y3,
            y4 => s_y4, y5 => s_y5, y6 => s_y6, y7 => s_y7,
            all_done => all_done
        );

    stim : process

        variable out_count   : integer := 0;
        variable total_cycles : integer := 0;
        variable all_done_seen_at : integer := -1;

        procedure check_output(out_num : integer) is
        begin
            assert to_integer(s_y0) = KERNEL0_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) & " kernel 0: y0 = " &
                       integer'image(to_integer(s_y0)) & "  expected " &
                       integer'image(KERNEL0_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y1) = KERNEL1_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) & " kernel 1: y1 = " &
                       integer'image(to_integer(s_y1)) & "  expected " &
                       integer'image(KERNEL1_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y2) = KERNEL2_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) & " kernel 2: y2 = " &
                       integer'image(to_integer(s_y2)) & "  expected " &
                       integer'image(KERNEL2_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y3) = KERNEL3_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) & " kernel 3: y3 = " &
                       integer'image(to_integer(s_y3)) & "  expected " &
                       integer'image(KERNEL3_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y4) = KERNEL4_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) & " kernel 4: y4 = " &
                       integer'image(to_integer(s_y4)) & "  expected " &
                       integer'image(KERNEL4_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y5) = KERNEL5_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) & " kernel 5: y5 = " &
                       integer'image(to_integer(s_y5)) & "  expected " &
                       integer'image(KERNEL5_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y6) = KERNEL6_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) & " kernel 6: y6 = " &
                       integer'image(to_integer(s_y6)) & "  expected " &
                       integer'image(KERNEL6_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y7) = KERNEL7_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) & " kernel 7: y7 = " &
                       integer'image(to_integer(s_y7)) & "  expected " &
                       integer'image(KERNEL7_EXPECTED(out_num - 1))
                severity failure;

            report "PASS window " & integer'image(out_num) &
                   ": y0=" & integer'image(to_integer(s_y0)) &
                   " y1=" & integer'image(to_integer(s_y1)) &
                   " y2=" & integer'image(to_integer(s_y2)) &
                   " y3=" & integer'image(to_integer(s_y3)) &
                   " y4=" & integer'image(to_integer(s_y4)) &
                   " y5=" & integer'image(to_integer(s_y5)) &
                   " y6=" & integer'image(to_integer(s_y6)) &
                   " y7=" & integer'image(to_integer(s_y7)) &
                   " (@ cycle " & integer'image(total_cycles) & ")";
        end procedure check_output;

    begin
        -- ---- Reset (3 clocks) ----------------------------------------
        rst      <= '1';
        valid_in <= '0';
        pixel_c0 <= (others => '0');
        pixel_c1 <= (others => '0');
        pixel_c2 <= (others => '0');
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst  <= '0';
        wait until rising_edge(clk);
        wait for 1 ns;

        -- ---- PHASE 1: stream the 5x5 image (one pixel triplet/clock) ---
        -- Channel 0: pixel i  (1..25)
        -- Channel 1: 2*i      (2..50, fits INT8)
        -- Channel 2: -i       (-1..-25, fits INT8)
        --
        -- No valid_out pulses are expected during this phase -- the DUT's
        -- capture phase only hands off to its processing phase once all 9
        -- windows have been buffered.
        valid_in <= '1';
        for i in 1 to 25 loop
            pixel_c0 <= to_signed(i,    8);
            pixel_c1 <= to_signed(2*i,  8);
            pixel_c2 <= to_signed(-i,   8);
            wait until rising_edge(clk);
            wait for 1 ns;
            total_cycles := total_cycles + 1;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_output(out_count);
            end if;
        end loop;
        valid_in <= '0';

        -- ---- PHASE 2: wait for the DUT to sequentially process all 9 --
        -- buffered windows through its internal one-window scheduler.
        while out_count < NUM_WINDOWS loop
            assert total_cycles <= MAX_CYCLES
                report "FAIL: did not observe all " & integer'image(NUM_WINDOWS) &
                       " windows within " & integer'image(MAX_CYCLES) &
                       " cycles (runaway FSM?)"
                severity failure;

            wait until rising_edge(clk);
            wait for 1 ns;
            total_cycles := total_cycles + 1;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_output(out_count);
            end if;

            if all_done = '1' then
                all_done_seen_at := out_count;
            end if;
        end loop;

        -- ---- Final checks --------------------------------------------
        assert out_count = NUM_WINDOWS
            report "FAIL: expected exactly " & integer'image(NUM_WINDOWS) &
                   " valid windows, observed " & integer'image(out_count)
            severity failure;

        assert all_done_seen_at = NUM_WINDOWS
            report "FAIL: all_done did not pulse on the same cycle as the " &
                   integer'image(NUM_WINDOWS) & "th window's valid_out " &
                   "(all_done seen at window count " & integer'image(all_done_seen_at) & ")"
            severity failure;

        report "Total cycles for streaming capture + sequential processing: " &
               integer'image(total_cycles) & " (measured, not asserted exact)";

        report "=== All stream_conv3x3_3chan_8out_time_mux tests PASSED ===" &
               "  (" & integer'image(NUM_WINDOWS) & " windows x 8 kernels = " &
               integer'image(NUM_WINDOWS * 8) & " / " & integer'image(NUM_WINDOWS * 8) &
               " outputs match Python golden vectors)" severity note;
        report "    NOTE: STREAMING TIME-MULTIPLEXED PROTOTYPE only (capture-then-process," &
               " no overlap, no backpressure into window3x3_stream)." &
               " Not full U-Net inference, not board-tested, BatchNorm not folded." severity note;

        wait;
    end process stim;

end architecture sim;
