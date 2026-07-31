-- tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20.vhd
-- REAL-UAVSAR-ACTIVATION validation testbench for the EXISTING, UNMODIFIED
-- 2-LANE resource-shared (time-multiplexed) folded Conv-BN-ReLU design
-- (stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd).
--
-- ---- Why this exists -------------------------------------------------------
-- Every prior GHDL/Vivado result for the resource-shared family (1-lane and
-- 2-lane) used the canonical SYNTHETIC 5x5x3 toy patch (channel 0 = 1..25,
-- channel 1 = 2x, channel 2 = -1x) -- never real UAVSAR sensor-derived
-- data. Meanwhile, the direct-parallel folded arithmetic core (commit
-- 313558e4) and the all-32-kernel single-pipeline verification
-- (real_tile_all32_kernels_q20_verification_summary.md) have already been
-- checked against a REAL 6x6 UAVSAR-tile-derived sub-block. This
-- testbench closes that gap for the resource-shared family: it feeds the
-- SAME real 6x6 sub-block through the EXISTING, UNMODIFIED 2-lane
-- resource-shared entity and checks all 32 outputs, for all 16 valid
-- windows, against the SAME already-GHDL-verified real-tile Q.20 golden
-- values the direct-parallel single-kernel-pipeline verification already
-- used -- i.e. a genuine cross-architecture consistency check, not a
-- fresh independent computation.
--
-- ---- What is reused, unmodified ---------------------------------------
--   window3x3_stream.vhd                    (existing, unmodified)
--   conv3x3_dot_time_mux.vhd                (existing, unmodified)
--   conv3x3_3chan_16out_bn_relu_time_mux.vhd (existing, unmodified)
--   stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd (existing, unmodified)
--   real_tile_stimulus_pkg.vhd               (existing, unmodified -- REAL
--                                              6x6 UAVSAR-tile-derived block)
--   first_conv_bn_relu_kernel{0..31}_real_tile_q20_pkg.vhd (existing,
--       unmodified -- the SAME 32 golden-vector packages the all-32-kernel
--       single-pipeline real-tile Q.20 verification already used)
--
-- ---- What is NEW ------------------------------------------------------------
--   resource_shared_32out_real_tile_q20_vectors_pkg.vhd -- a package
--     declared under the SAME NAME the 2-lane design's dependencies
--     already reference (`first_layer_32out_bn_relu_resource_shared_pkg`),
--     populated with REAL-TILE Q.20 weight/scale/bias constants instead of
--     the existing file's Q.16 toy-pattern constants. This package is
--     analyzed into an ISOLATED GHDL work library (see the run script's
--     --workdir flag) so it NEVER overwrites or interferes with the
--     existing toy-pattern package/testbenches compiled in the default
--     work library in this same directory. See that package's header for
--     the full explanation of why a same-named package in an isolated
--     library was chosen over editing the existing file.
--   This testbench itself.
--
-- ---- Input: REAL 6x6 UAVSAR-tile-derived sub-block (NOT the toy patch) ----
-- Streamed exactly like every other real-tile testbench in this
-- directory: IMG_WIDTH=6, one pixel triplet per clock, giving
-- (6-2)x(6-2) = 16 valid 3x3 windows, all 16 checked (NUM_WINDOWS=16 for
-- IMG_WIDTH=6, computed by the DUT itself -- not hardcoded here).
--
-- ---- Expected outputs: REUSED, not freshly computed -----------------------
-- K{k}_RT_BN_RELU_EXPECTED_FX(widx), k=0..31, widx=0..15 -- the SAME
-- constants already GHDL-verified against the direct-parallel
-- single-kernel-pipeline real-tile Q.20 design family. Total checked:
-- 16 windows x 32 kernels = 512 outputs.
--
-- ---- What this validates -----------------------------------------------
-- That the EXISTING, UNMODIFIED 2-lane resource-shared time-multiplexed
-- datapath -- a fundamentally different scheduling architecture from the
-- direct-parallel folded arithmetic core (sequential per-lane dot-product
-- reuse vs. 96-way fully parallel dot products) -- produces BIT-IDENTICAL
-- Q.20 fixed-point results to the direct-parallel family, when driven by
-- the SAME real UAVSAR-derived window data and the SAME folded
-- weight/scale/bias constants. It does NOT validate full-tile streaming,
-- full U-Net inference, board timing, board power, or measured speedup.

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
use work.first_conv_bn_relu_kernel16_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel17_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel18_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel19_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel20_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel21_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel22_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel23_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel24_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel25_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel26_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel27_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel28_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel29_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel30_real_tile_q20_pkg.all;
use work.first_conv_bn_relu_kernel31_real_tile_q20_pkg.all;

entity tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20 is
end entity tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20;

architecture sim of tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20 is

    constant CLK_PERIOD : time     := 10 ns;
    constant IMG_W       : positive := BLOCK_SIZE;  -- 6, from real_tile_stimulus_pkg
    constant NUM_WINDOWS : positive := (IMG_W - 2) * (IMG_W - 2);  -- 16

    -- Generous sanity bound only, NOT an asserted exact cycle count.
    constant MAX_CYCLES : integer := (IMG_W * IMG_W) + NUM_WINDOWS * 3500;

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    signal valid_out : std_logic;
    signal all_done  : std_logic;
    signal s_y0, s_y1, s_y2, s_y3, s_y4, s_y5, s_y6, s_y7, s_y8, s_y9,
           s_y10, s_y11, s_y12, s_y13, s_y14, s_y15, s_y16, s_y17, s_y18, s_y19,
           s_y20, s_y21, s_y22, s_y23, s_y24, s_y25, s_y26, s_y27, s_y28, s_y29,
           s_y30, s_y31 : signed(47 downto 0);

begin

    clk <= not clk after CLK_PERIOD / 2;

    -- ================================================================
    -- DUT: the EXISTING, UNMODIFIED 2-lane resource-shared design.
    -- Its dependencies (conv3x3_3chan_16out_bn_relu_time_mux, etc.)
    -- resolve `first_layer_32out_bn_relu_resource_shared_pkg` to THIS
    -- run's real-tile Q.20 package (analyzed into an isolated work
    -- library -- see the run script), not the existing toy Q.16 one.
    -- ================================================================
    dut : entity work.stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux
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
            y15 => s_y15, y16 => s_y16, y17 => s_y17, y18 => s_y18, y19 => s_y19,
            y20 => s_y20, y21 => s_y21, y22 => s_y22, y23 => s_y23, y24 => s_y24,
            y25 => s_y25, y26 => s_y26, y27 => s_y27, y28 => s_y28, y29 => s_y29,
            y30 => s_y30, y31 => s_y31,
            all_done => all_done
        );

    stim : process

        variable out_count       : integer := 0;   -- window count (0..15)
        variable total_cycles    : integer := 0;
        variable all_done_seen_at : integer := -1;

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
            check_one(16, s_y16, K16_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(17, s_y17, K17_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(18, s_y18, K18_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(19, s_y19, K19_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(20, s_y20, K20_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(21, s_y21, K21_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(22, s_y22, K22_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(23, s_y23, K23_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(24, s_y24, K24_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(25, s_y25, K25_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(26, s_y26, K26_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(27, s_y27, K27_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(28, s_y28, K28_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(29, s_y29, K29_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(30, s_y30, K30_RT_BN_RELU_EXPECTED_FX(widx), widx);
            check_one(31, s_y31, K31_RT_BN_RELU_EXPECTED_FX(widx), widx);

            report "PASS window " & integer'image(window_num) &
                   " y0=" & integer'image(to_integer(s_y0)) &
                   " y1=" & integer'image(to_integer(s_y1)) &
                   " y31=" & integer'image(to_integer(s_y31)) &
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

        -- ---- PHASE 2: wait for the DUT to sequentially process all 16 --
        -- buffered windows, both lanes running in parallel per window.
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

        report "=== All tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20 tests PASSED ===" &
               "  (" & integer'image(NUM_WINDOWS) & " REAL windows x 32 kernels = " &
               integer'image(NUM_WINDOWS * 32) & " / " & integer'image(NUM_WINDOWS * 32) &
               " outputs match the SAME real-tile Q.20 golden vectors already used by the" &
               " direct-parallel single-kernel-pipeline verification)" severity note;
        report "    NOTE: 2-LANE RESOURCE-SHARED (time-multiplexed) folded Conv-BN-ReLU" &
               " design, driven by REAL UAVSAR-tile-derived activations for the first time." &
               " Not full-tile streaming, not full U-Net inference, not board-tested, not a" &
               " measured speedup/power claim." severity note;

        wait;
    end process stim;

end architecture sim;
