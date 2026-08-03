-- tb_first_layer_16out_folded_arithmetic_core_streaming_front_end.vhd
-- REAL-UAVSAR-ACTIVATION self-checking testbench for the NEW, integrated
-- first_layer_16out_folded_arithmetic_core_streaming_front_end (streaming
-- front end + 16-output folded Conv-BN-ReLU arithmetic core).
--
-- ---- Why this exists -------------------------------------------------------
-- The 32-output integrated skeleton (commit 77a71e35) was GHDL-verified
-- only indirectly: the front end alone
-- (tb_window3x3_stream_3chan_flattened.vhd) and the arithmetic core alone
-- (tb_first_layer_32out_folded_arithmetic_core.vhd, single flattened
-- window per cycle, not streamed) were each checked separately -- there was
-- no end-to-end streamed-pixels-in test of the combined design. This
-- testbench closes that gap for the NEW 16-output artifact: it streams the
-- SAME real 6x6 UAVSAR-tile-derived block already used elsewhere in this
-- repo (real_tile_stimulus_pkg.vhd) through the combined, integrated
-- streaming front end + 16-output arithmetic core, and checks all 16
-- output channels for all 16 valid windows against the EXISTING,
-- independently-generated, already-GHDL-verified per-kernel real-tile
-- Q.20 golden vectors -- a genuine end-to-end, real-sensor-data check, not
-- a fresh independent computation.
--
-- ---- What is reused, unmodified ---------------------------------------
--   real_tile_stimulus_pkg.vhd                              (existing, unmodified -- REAL 6x6 block)
--   first_conv_bn_relu_kernel{0..15}_real_tile_q20_pkg.vhd   (existing, unmodified -- golden vectors,
--                                                              the SAME per-kernel Q.20 packages already
--                                                              used by the 32-output resource-shared
--                                                              real-tile Q.20 verification)
--
-- ---- What is NEW ------------------------------------------------------------
--   This testbench itself, driving the NEW
--   first_layer_16out_folded_arithmetic_core_streaming_front_end DUT.
--
-- ---- Input: REAL 6x6 UAVSAR-tile-derived sub-block (NOT a toy patch) ----
-- IMG_WIDTH=6, one pixel triplet per clock, giving (6-2)x(6-2) = 16 valid
-- 3x3 windows, all 16 checked, row-major -- the SAME window order
-- window3x3_stream (and therefore window3x3_stream_3chan_flattened) always
-- produces, and the SAME order K{k}_RT_BN_RELU_EXPECTED_FX(0..15) already
-- indexes.
--
-- ---- Expected outputs: REUSED, not freshly computed -----------------------
-- K{k}_RT_BN_RELU_EXPECTED_FX(widx), k=0..15, widx=0..15 -- the SAME
-- constants already GHDL-verified by the direct-parallel single-kernel
-- real-tile Q.20 designs and by the 32-output resource-shared real-tile
-- Q.20 testbench. Total checked: 16 windows x 16 kernels = 256 outputs.
--
-- ---- What this validates -----------------------------------------------
-- That the EXISTING, unmodified streaming front end, chained into the NEW
-- 16-output folded arithmetic core, produces BIT-IDENTICAL Q.20
-- fixed-point results (for channels 0-15) to every other already-verified
-- architecture in this repo, when driven by REAL UAVSAR-derived pixel data
-- streamed one triplet per clock. It does NOT validate full-tile
-- streaming at realistic (128/256) widths, full U-Net inference, board
-- timing, board power, or measured speedup.

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
use work.first_conv_bn_relu_kernel8_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel9_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel10_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel11_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel12_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel13_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel14_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel15_real_tile_q20_pkg.all;

entity tb_first_layer_16out_folded_arithmetic_core_streaming_front_end is
end entity tb_first_layer_16out_folded_arithmetic_core_streaming_front_end;

architecture sim of tb_first_layer_16out_folded_arithmetic_core_streaming_front_end is

    constant CLK_PERIOD  : time     := 10 ns;
    constant IMG_W       : positive := BLOCK_SIZE;  -- 6, from real_tile_stimulus_pkg
    constant NUM_WINDOWS : positive := (IMG_W - 2) * (IMG_W - 2);  -- 16

    -- Generous sanity bound only, NOT an asserted exact cycle count:
    -- IMG_W*IMG_W pixels to stream in, plus a wide margin for the 6-cycle
    -- arithmetic-core pipeline to drain after the last pixel.
    constant MAX_CYCLES : integer := (IMG_W * IMG_W) + 50;

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    signal valid_out : std_logic;
    signal s_y0, s_y1, s_y2, s_y3, s_y4, s_y5, s_y6, s_y7,
           s_y8, s_y9, s_y10, s_y11, s_y12, s_y13, s_y14, s_y15 : signed(47 downto 0);

begin

    clk <= not clk after CLK_PERIOD / 2;

    -- ================================================================
    -- DUT: the NEW integrated streaming front end + 16-output folded
    -- arithmetic core.
    -- ================================================================
    dut : entity work.first_layer_16out_folded_arithmetic_core_streaming_front_end
        generic map (IMG_WIDTH => IMG_W)
        port map (
            clk      => clk,
            rst      => rst,
            valid_in => valid_in,
            pixel_c0 => pixel_c0,
            pixel_c1 => pixel_c1,
            pixel_c2 => pixel_c2,
            valid_out => valid_out,
            y0 => s_y0, y1 => s_y1, y2 => s_y2, y3 => s_y3, y4 => s_y4,
            y5 => s_y5, y6 => s_y6, y7 => s_y7, y8 => s_y8, y9 => s_y9,
            y10 => s_y10, y11 => s_y11, y12 => s_y12, y13 => s_y13, y14 => s_y14,
            y15 => s_y15
        );

    stim : process

        variable out_count    : integer := 0;   -- window count (0..15)
        variable total_cycles : integer := 0;

        procedure check_one(kernel_id : integer; got : signed(47 downto 0);
                             expected : integer; widx : integer) is
        begin
            assert to_integer(got) = expected
                report "FAIL window " & integer'image(widx) &
                       " kernel " & integer'image(kernel_id) &
                       ": y = " & integer'image(to_integer(got)) &
                       "  expected " & integer'image(expected)
                severity failure;
        end procedure check_one;

        procedure check_output(window_num : integer) is
            -- window_num is 1-based window count; widx is the 0-based
            -- row-major index into each kernel's 16-entry expected array.
            variable widx : integer := window_num - 1;
        begin
            check_one(0, s_y0, K0_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(1, s_y1, K1_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(2, s_y2, K2_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(3, s_y3, K3_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(4, s_y4, K4_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(5, s_y5, K5_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(6, s_y6, K6_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(7, s_y7, K7_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(8, s_y8, K8_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(9, s_y9, K9_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(10, s_y10, K10_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(11, s_y11, K11_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(12, s_y12, K12_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(13, s_y13, K13_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(14, s_y14, K14_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(15, s_y15, K15_RT_BN_RELU_EXPECTED_FX(widx), widx);

            report "PASS window " & integer'image(window_num) &
                   " y0=" & integer'image(to_integer(s_y0)) &
                   " y1=" & integer'image(to_integer(s_y1)) &
                   " y15=" & integer'image(to_integer(s_y15)) &
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

        -- ---- PHASE 1: stream the REAL 6x6 UAVSAR-tile-derived block ---
        -- (one pixel triplet per clock, row-major, from real_tile_stimulus_pkg)
        valid_in <= '1';
        for i in 0 to (IMG_W * IMG_W) - 1 loop
            pixel_c0 <= to_signed(REAL_TILE_CH0(i), 8);
            pixel_c1 <= to_signed(REAL_TILE_CH1(i), 8);
            pixel_c2 <= to_signed(REAL_TILE_CH2(i), 8);
            wait until rising_edge(clk);
            wait for 1 ns;
            total_cycles := total_cycles + 1;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_output(out_count);
            end if;
        end loop;
        valid_in <= '0';

        -- ---- PHASE 2: drain the arithmetic core's 6-cycle pipeline ----
        while out_count < NUM_WINDOWS loop
            assert total_cycles <= MAX_CYCLES
                report "FAIL: did not observe all " & integer'image(NUM_WINDOWS) &
                       " windows within " & integer'image(MAX_CYCLES) &
                       " cycles (runaway pipeline?)"
                severity failure;

            wait until rising_edge(clk);
            wait for 1 ns;
            total_cycles := total_cycles + 1;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_output(out_count);
            end if;
        end loop;

        -- ---- Final checks --------------------------------------------
        assert out_count = NUM_WINDOWS
            report "FAIL: expected exactly " & integer'image(NUM_WINDOWS) &
                   " valid windows, observed " & integer'image(out_count)
            severity failure;

        report "Total cycles for streaming capture + pipeline drain: " &
               integer'image(total_cycles) & " (measured, not asserted exact)";

        report "=== All tb_first_layer_16out_folded_arithmetic_core_streaming_front_end tests PASSED ===" &
               "  (" & integer'image(NUM_WINDOWS) & " REAL windows x 16 kernels = " &
               integer'image(NUM_WINDOWS * 16) & " / " & integer'image(NUM_WINDOWS * 16) &
               " outputs match the SAME real-tile Q.20 golden vectors already used by the" &
               " direct-parallel single-kernel and 32-output resource-shared verifications)" severity note;
        report "    NOTE: 16-output (of 32) folded Conv-BN-ReLU streaming front-end + arithmetic-core" &
               " ARTIFACT, driven by REAL UAVSAR-tile-derived activations. Not full-tile streaming" &
               " at realistic widths, not full U-Net inference, not board-tested, not a measured" &
               " speedup/power claim." severity note;

        wait;
    end process stim;

end architecture sim;
