-- tb_first_layer_2win_8out_folded_arithmetic_core.vhd
-- Self-checking, REAL-UAVSAR-ACTIVATION testbench for the NEW spatial-
-- parallelism arithmetic core (first_layer_2win_8out_folded_arithmetic_core.vhd):
-- drives PAIRS of already-flattened, already-extracted 27-value windows
-- (one pair per cycle -- lane 0 and lane 1) and checks BOTH lanes' 8
-- Q.20 fixed-point folded Conv-BN-ReLU outputs against the EXISTING,
-- independently-generated, already-GHDL-verified per-kernel real-tile
-- Q.20 golden vectors.
--
-- ---- Why this exists -------------------------------------------------------
-- first_layer_16out_folded_arithmetic_core.vhd's own testbench (via its
-- integrated streaming front end) checks ONE spatial window per cycle,
-- 16 output channels wide. This testbench checks the SAME real 6x6
-- UAVSAR-tile-derived block's 16 valid 3x3 windows (the SAME block used
-- throughout this repo's real-tile verification work), but grouped into
-- 8 PAIRS and driven 2 windows per cycle -- the arithmetic-core-only
-- check of the spatial-parallelism trade (8 output channels/window x 2
-- windows/cycle, vs. 16 output channels/window x 1 window/cycle).
--
-- ---- What is reused, unmodified ---------------------------------------
--   real_tile_stimulus_pkg.vhd                             (existing, unmodified -- REAL 6x6 block)
--   first_conv_bn_relu_kernel{0..7}_real_tile_q20_pkg.vhd   (existing, unmodified -- golden vectors,
--                                                             the SAME per-kernel Q.20 packages already
--                                                             used by the 16-output and 32-output
--                                                             resource-shared/streaming real-tile
--                                                             verifications)
--
-- ---- Windows: pre-extracted directly from the real 6x6 block, NOT streamed --
-- This testbench does NOT use window3x3_stream or
-- window3x3_stream_3chan_flattened (no streaming front end exists for
-- this 2-lane core yet -- that is explicitly future work). Instead, all
-- 16 valid 3x3 windows of the real 6x6 block (row-major output positions
-- (row_out, col_out), row_out/col_out in 0..3) are extracted directly, in
-- VHDL, from real_tile_stimulus_pkg.vhd's REAL_TILE_CH0/1/2 constant
-- arrays -- the SAME flattening order (channel-major, then row-major
-- within each 3x3 channel plane) used by every other real-tile testbench
-- in this repo, and the SAME row-major window-index order that
-- window3x3_stream itself produces, so window index widx = row_out*4 +
-- col_out matches K{k}_RT_BN_RELU_EXPECTED_FX(widx) directly.
--
-- ---- Pairing: window pairs = (widx=2p, widx=2p+1) for p = 0..7 ------------
-- Lane 0 gets the EVEN-indexed window of each pair, lane 1 gets the
-- ODD-indexed window -- i.e. consecutive row-major window positions are
-- processed together, one pair per clock.
--
-- ---- Expected outputs: REUSED, not freshly computed -----------------------
-- K{k}_RT_BN_RELU_EXPECTED_FX(widx), k=0..7, widx=0..15 -- the SAME
-- constants already GHDL-verified by the direct-parallel single-kernel,
-- 16-output streaming, and 32-output resource-shared real-tile Q.20
-- designs. Total checked: 8 pairs x 2 lanes x 8 kernels = 128 outputs.
--
-- ---- What this validates -----------------------------------------------
-- That the NEW 2-window x 8-output spatial-parallelism arithmetic core
-- produces BIT-IDENTICAL Q.20 fixed-point results (for channels 0-7) to
-- every other already-verified architecture in this repo, for BOTH
-- spatial lanes simultaneously, when driven by REAL UAVSAR-derived pixel
-- data. It does NOT validate any streaming front end (none exists for
-- this core), full-tile streaming, full U-Net inference, board timing, or
-- measured speedup.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.real_tile_stimulus_pkg.all;
use work.first_conv_bn_relu_kernel0_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel1_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel2_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel3_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel4_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel5_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel6_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel7_real_tile_q20_pkg.all;

entity tb_first_layer_2win_8out_folded_arithmetic_core is
end entity tb_first_layer_2win_8out_folded_arithmetic_core;

architecture sim of tb_first_layer_2win_8out_folded_arithmetic_core is

    constant CLK_PERIOD   : time     := 10 ns;
    constant IMG_W        : positive := BLOCK_SIZE;              -- 6, from real_tile_stimulus_pkg
    constant WINDOWS_SIDE : positive := IMG_W - 2;                -- 4 (valid windows per row/col)
    constant NUM_WINDOWS  : positive := WINDOWS_SIDE * WINDOWS_SIDE;  -- 16
    constant NUM_PAIRS    : positive := NUM_WINDOWS / 2;           -- 8

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal w0_c0_p0, w0_c0_p1, w0_c0_p2, w0_c0_p3, w0_c0_p4, w0_c0_p5, w0_c0_p6, w0_c0_p7, w0_c0_p8 : signed(7 downto 0) := (others => '0');
    signal w0_c1_p0, w0_c1_p1, w0_c1_p2, w0_c1_p3, w0_c1_p4, w0_c1_p5, w0_c1_p6, w0_c1_p7, w0_c1_p8 : signed(7 downto 0) := (others => '0');
    signal w0_c2_p0, w0_c2_p1, w0_c2_p2, w0_c2_p3, w0_c2_p4, w0_c2_p5, w0_c2_p6, w0_c2_p7, w0_c2_p8 : signed(7 downto 0) := (others => '0');

    signal w1_c0_p0, w1_c0_p1, w1_c0_p2, w1_c0_p3, w1_c0_p4, w1_c0_p5, w1_c0_p6, w1_c0_p7, w1_c0_p8 : signed(7 downto 0) := (others => '0');
    signal w1_c1_p0, w1_c1_p1, w1_c1_p2, w1_c1_p3, w1_c1_p4, w1_c1_p5, w1_c1_p6, w1_c1_p7, w1_c1_p8 : signed(7 downto 0) := (others => '0');
    signal w1_c2_p0, w1_c2_p1, w1_c2_p2, w1_c2_p3, w1_c2_p4, w1_c2_p5, w1_c2_p6, w1_c2_p7, w1_c2_p8 : signed(7 downto 0) := (others => '0');

    signal valid_out : std_logic;
    signal s_y0_0, s_y0_1, s_y0_2, s_y0_3, s_y0_4, s_y0_5, s_y0_6, s_y0_7 : signed(47 downto 0);
    signal s_y1_0, s_y1_1, s_y1_2, s_y1_3, s_y1_4, s_y1_5, s_y1_6, s_y1_7 : signed(47 downto 0);

    -- One flattened 27-value window (channel-major, then row-major within
    -- each 3x3 channel plane) -- SAME order used throughout this repo.
    type window27_t is array (0 to 26) of integer range -128 to 127;

    -- Extracts the valid 3x3 window at output position (row_out, col_out)
    -- (0-based, 0..WINDOWS_SIDE-1 each) directly from the real 6x6 block's
    -- flat REAL_TILE_CH0/1/2 arrays -- NOT via window3x3_stream (no
    -- streaming front end exists for this core).
    impure function extract_window(row_out : integer; col_out : integer) return window27_t is
        variable w   : window27_t;
        variable idx : integer := 0;
    begin
        for r in row_out to row_out + 2 loop
            for c in col_out to col_out + 2 loop
                w(idx) := REAL_TILE_CH0(r * IMG_W + c);
                idx := idx + 1;
            end loop;
        end loop;
        for r in row_out to row_out + 2 loop
            for c in col_out to col_out + 2 loop
                w(idx) := REAL_TILE_CH1(r * IMG_W + c);
                idx := idx + 1;
            end loop;
        end loop;
        for r in row_out to row_out + 2 loop
            for c in col_out to col_out + 2 loop
                w(idx) := REAL_TILE_CH2(r * IMG_W + c);
                idx := idx + 1;
            end loop;
        end loop;
        return w;
    end function extract_window;

begin

    clk <= not clk after CLK_PERIOD / 2;

    -- ================================================================
    -- DUT: the NEW 2-window x 8-output spatial-parallelism arithmetic
    -- core.
    -- ================================================================
    dut : entity work.first_layer_2win_8out_folded_arithmetic_core
        port map (
            clk => clk, rst => rst, valid_in => valid_in,
            w0_c0_p0 => w0_c0_p0, w0_c0_p1 => w0_c0_p1, w0_c0_p2 => w0_c0_p2,
            w0_c0_p3 => w0_c0_p3, w0_c0_p4 => w0_c0_p4, w0_c0_p5 => w0_c0_p5,
            w0_c0_p6 => w0_c0_p6, w0_c0_p7 => w0_c0_p7, w0_c0_p8 => w0_c0_p8,
            w0_c1_p0 => w0_c1_p0, w0_c1_p1 => w0_c1_p1, w0_c1_p2 => w0_c1_p2,
            w0_c1_p3 => w0_c1_p3, w0_c1_p4 => w0_c1_p4, w0_c1_p5 => w0_c1_p5,
            w0_c1_p6 => w0_c1_p6, w0_c1_p7 => w0_c1_p7, w0_c1_p8 => w0_c1_p8,
            w0_c2_p0 => w0_c2_p0, w0_c2_p1 => w0_c2_p1, w0_c2_p2 => w0_c2_p2,
            w0_c2_p3 => w0_c2_p3, w0_c2_p4 => w0_c2_p4, w0_c2_p5 => w0_c2_p5,
            w0_c2_p6 => w0_c2_p6, w0_c2_p7 => w0_c2_p7, w0_c2_p8 => w0_c2_p8,
            w1_c0_p0 => w1_c0_p0, w1_c0_p1 => w1_c0_p1, w1_c0_p2 => w1_c0_p2,
            w1_c0_p3 => w1_c0_p3, w1_c0_p4 => w1_c0_p4, w1_c0_p5 => w1_c0_p5,
            w1_c0_p6 => w1_c0_p6, w1_c0_p7 => w1_c0_p7, w1_c0_p8 => w1_c0_p8,
            w1_c1_p0 => w1_c1_p0, w1_c1_p1 => w1_c1_p1, w1_c1_p2 => w1_c1_p2,
            w1_c1_p3 => w1_c1_p3, w1_c1_p4 => w1_c1_p4, w1_c1_p5 => w1_c1_p5,
            w1_c1_p6 => w1_c1_p6, w1_c1_p7 => w1_c1_p7, w1_c1_p8 => w1_c1_p8,
            w1_c2_p0 => w1_c2_p0, w1_c2_p1 => w1_c2_p1, w1_c2_p2 => w1_c2_p2,
            w1_c2_p3 => w1_c2_p3, w1_c2_p4 => w1_c2_p4, w1_c2_p5 => w1_c2_p5,
            w1_c2_p6 => w1_c2_p6, w1_c2_p7 => w1_c2_p7, w1_c2_p8 => w1_c2_p8,
            valid_out => valid_out,
            y0_0 => s_y0_0, y0_1 => s_y0_1, y0_2 => s_y0_2, y0_3 => s_y0_3,
            y0_4 => s_y0_4, y0_5 => s_y0_5, y0_6 => s_y0_6, y0_7 => s_y0_7,
            y1_0 => s_y1_0, y1_1 => s_y1_1, y1_2 => s_y1_2, y1_3 => s_y1_3,
            y1_4 => s_y1_4, y1_5 => s_y1_5, y1_6 => s_y1_6, y1_7 => s_y1_7
        );

    stim : process

        variable out_count : integer := 0;   -- pair count (0..NUM_PAIRS-1)

        procedure drive_pair(w0 : window27_t; w1 : window27_t) is
        begin
            w0_c0_p0 <= to_signed(w0(0), 8);  w0_c0_p1 <= to_signed(w0(1), 8);  w0_c0_p2 <= to_signed(w0(2), 8);
            w0_c0_p3 <= to_signed(w0(3), 8);  w0_c0_p4 <= to_signed(w0(4), 8);  w0_c0_p5 <= to_signed(w0(5), 8);
            w0_c0_p6 <= to_signed(w0(6), 8);  w0_c0_p7 <= to_signed(w0(7), 8);  w0_c0_p8 <= to_signed(w0(8), 8);
            w0_c1_p0 <= to_signed(w0(9), 8);  w0_c1_p1 <= to_signed(w0(10), 8); w0_c1_p2 <= to_signed(w0(11), 8);
            w0_c1_p3 <= to_signed(w0(12), 8); w0_c1_p4 <= to_signed(w0(13), 8); w0_c1_p5 <= to_signed(w0(14), 8);
            w0_c1_p6 <= to_signed(w0(15), 8); w0_c1_p7 <= to_signed(w0(16), 8); w0_c1_p8 <= to_signed(w0(17), 8);
            w0_c2_p0 <= to_signed(w0(18), 8); w0_c2_p1 <= to_signed(w0(19), 8); w0_c2_p2 <= to_signed(w0(20), 8);
            w0_c2_p3 <= to_signed(w0(21), 8); w0_c2_p4 <= to_signed(w0(22), 8); w0_c2_p5 <= to_signed(w0(23), 8);
            w0_c2_p6 <= to_signed(w0(24), 8); w0_c2_p7 <= to_signed(w0(25), 8); w0_c2_p8 <= to_signed(w0(26), 8);

            w1_c0_p0 <= to_signed(w1(0), 8);  w1_c0_p1 <= to_signed(w1(1), 8);  w1_c0_p2 <= to_signed(w1(2), 8);
            w1_c0_p3 <= to_signed(w1(3), 8);  w1_c0_p4 <= to_signed(w1(4), 8);  w1_c0_p5 <= to_signed(w1(5), 8);
            w1_c0_p6 <= to_signed(w1(6), 8);  w1_c0_p7 <= to_signed(w1(7), 8);  w1_c0_p8 <= to_signed(w1(8), 8);
            w1_c1_p0 <= to_signed(w1(9), 8);  w1_c1_p1 <= to_signed(w1(10), 8); w1_c1_p2 <= to_signed(w1(11), 8);
            w1_c1_p3 <= to_signed(w1(12), 8); w1_c1_p4 <= to_signed(w1(13), 8); w1_c1_p5 <= to_signed(w1(14), 8);
            w1_c1_p6 <= to_signed(w1(15), 8); w1_c1_p7 <= to_signed(w1(16), 8); w1_c1_p8 <= to_signed(w1(17), 8);
            w1_c2_p0 <= to_signed(w1(18), 8); w1_c2_p1 <= to_signed(w1(19), 8); w1_c2_p2 <= to_signed(w1(20), 8);
            w1_c2_p3 <= to_signed(w1(21), 8); w1_c2_p4 <= to_signed(w1(22), 8); w1_c2_p5 <= to_signed(w1(23), 8);
            w1_c2_p6 <= to_signed(w1(24), 8); w1_c2_p7 <= to_signed(w1(25), 8); w1_c2_p8 <= to_signed(w1(26), 8);
        end procedure drive_pair;

        procedure check_one(lane : integer; kernel_id : integer; got : signed(47 downto 0);
                             expected : integer; widx : integer) is
        begin
            assert to_integer(got) = expected
                report "FAIL lane " & integer'image(lane) &
                       " window " & integer'image(widx) &
                       " kernel " & integer'image(kernel_id) &
                       ": y = " & integer'image(to_integer(got)) &
                       "  expected " & integer'image(expected)
                severity failure;
        end procedure check_one;

        procedure check_pair(pair_num : integer) is
            -- pair_num is 1-based pair count; widx0/widx1 are the 0-based
            -- row-major window indices for lane 0 (even) / lane 1 (odd).
            variable widx0 : integer := (pair_num - 1) * 2;
            variable widx1 : integer := (pair_num - 1) * 2 + 1;
        begin
            -- Lane 0 (even-indexed window), channels 0-7
            check_one(0, 0, s_y0_0, K0_RT_BN_RELU_EXPECTED_FX(widx0), widx0);
            check_one(0, 1, s_y0_1, K1_RT_BN_RELU_EXPECTED_FX(widx0), widx0);
            check_one(0, 2, s_y0_2, K2_RT_BN_RELU_EXPECTED_FX(widx0), widx0);
            check_one(0, 3, s_y0_3, K3_RT_BN_RELU_EXPECTED_FX(widx0), widx0);
            check_one(0, 4, s_y0_4, K4_RT_BN_RELU_EXPECTED_FX(widx0), widx0);
            check_one(0, 5, s_y0_5, K5_RT_BN_RELU_EXPECTED_FX(widx0), widx0);
            check_one(0, 6, s_y0_6, K6_RT_BN_RELU_EXPECTED_FX(widx0), widx0);
            check_one(0, 7, s_y0_7, K7_RT_BN_RELU_EXPECTED_FX(widx0), widx0);

            -- Lane 1 (odd-indexed window), channels 0-7
            check_one(1, 0, s_y1_0, K0_RT_BN_RELU_EXPECTED_FX(widx1), widx1);
            check_one(1, 1, s_y1_1, K1_RT_BN_RELU_EXPECTED_FX(widx1), widx1);
            check_one(1, 2, s_y1_2, K2_RT_BN_RELU_EXPECTED_FX(widx1), widx1);
            check_one(1, 3, s_y1_3, K3_RT_BN_RELU_EXPECTED_FX(widx1), widx1);
            check_one(1, 4, s_y1_4, K4_RT_BN_RELU_EXPECTED_FX(widx1), widx1);
            check_one(1, 5, s_y1_5, K5_RT_BN_RELU_EXPECTED_FX(widx1), widx1);
            check_one(1, 6, s_y1_6, K6_RT_BN_RELU_EXPECTED_FX(widx1), widx1);
            check_one(1, 7, s_y1_7, K7_RT_BN_RELU_EXPECTED_FX(widx1), widx1);

            report "PASS pair " & integer'image(pair_num) &
                   " (widx0=" & integer'image(widx0) & " widx1=" & integer'image(widx1) & ")" &
                   " lane0_y0=" & integer'image(to_integer(s_y0_0)) &
                   " lane1_y0=" & integer'image(to_integer(s_y1_0));
        end procedure check_pair;

    begin
        -- ---- Reset (3 clocks) ----------------------------------------
        rst      <= '1';
        valid_in <= '0';
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst <= '0';
        wait until rising_edge(clk);
        wait for 1 ns;

        -- ---- Present all NUM_PAIRS window pairs, one pair per clock ---
        valid_in <= '1';
        for p in 0 to NUM_PAIRS - 1 loop
            drive_pair(
                extract_window((2 * p) / WINDOWS_SIDE, (2 * p) mod WINDOWS_SIDE),
                extract_window((2 * p + 1) / WINDOWS_SIDE, (2 * p + 1) mod WINDOWS_SIDE)
            );
            wait until rising_edge(clk);
            wait for 1 ns;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_pair(out_count);
            end if;
        end loop;

        -- ---- Drain: enough clocks to flush the 6-cycle pipeline -------
        valid_in <= '0';
        for d in 1 to 8 loop
            wait until rising_edge(clk);
            wait for 1 ns;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_pair(out_count);
            end if;
        end loop;

        -- ---- Final checks --------------------------------------------
        assert out_count = NUM_PAIRS
            report "FAIL: expected exactly " & integer'image(NUM_PAIRS) &
                   " valid output pairs, observed " & integer'image(out_count)
            severity failure;

        report "=== All tb_first_layer_2win_8out_folded_arithmetic_core tests PASSED ===" &
               "  (" & integer'image(NUM_PAIRS) & " pairs x 2 lanes x 8 kernels = " &
               integer'image(NUM_PAIRS * 2 * 8) & " / " & integer'image(NUM_PAIRS * 2 * 8) &
               " outputs match the SAME real-tile Q.20 golden vectors already used by the" &
               " direct-parallel single-kernel, 16-output streaming, and 32-output" &
               " resource-shared verifications)" severity note;
        report "    NOTE: 2-window x 8-output (of 32) folded Conv-BN-ReLU SPATIAL-PARALLELISM" &
               " arithmetic-core ARTIFACT, pre-extracted windows only (no streaming front end" &
               " exists for this core yet), driven by REAL UAVSAR-tile-derived activations." &
               " Not full-tile streaming, not full U-Net inference, not board-tested, not a" &
               " measured speedup/power claim." severity note;

        wait;
    end process stim;

end architecture sim;
