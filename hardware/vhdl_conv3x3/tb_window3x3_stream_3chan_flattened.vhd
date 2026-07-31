-- tb_window3x3_stream_3chan_flattened.vhd
-- Self-checking, END-TO-END testbench for
-- first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd --
-- verifies that streaming a REAL row-major pixel image through the NEW
-- 3-channel front end (window3x3_stream_3chan_flattened.vhd, itself built
-- from 3 existing, unmodified window3x3_stream instances) into the
-- EXISTING, UNMODIFIED direct-parallel folded 32-output arithmetic core
-- produces the SAME real-tile Q.20 golden outputs already verified twice
-- elsewhere in this repo (the direct-parallel single-shot testbench and
-- the resource-shared real-tile testbench).
--
-- ---- Input: REAL 6x6 UAVSAR-tile-derived sub-block (NOT synthetic) --------
-- Streamed row-major, one pixel triplet per clock, from the EXISTING,
-- UNMODIFIED real_tile_stimulus_pkg.vhd (the SAME 6x6 block used by every
-- other real-tile testbench in this directory). IMG_WIDTH=6 gives
-- (6-2)x(6-2) = 16 valid 3x3 windows.
--
-- ---- Expected outputs: REUSED, not freshly computed -----------------------
-- K{k}_RT_BN_RELU_EXPECTED_FX(widx), k=0..31, widx=0..15 -- the SAME
-- constants already GHDL-verified against both the direct-parallel
-- single-shot arithmetic core and the resource-shared time-multiplexed
-- design. Total checked: 16 windows x 32 kernels = 512 outputs.
--
-- ---- What this validates -----------------------------------------------
-- That a real streaming pixel front end CAN correctly feed the
-- direct-parallel arithmetic core's exact port interface, end to end, and
-- measures (does not merely assume) the combined fill latency and
-- per-window throughput once primed. It does NOT claim this combined
-- design has been synthesized, does NOT claim board timing, and does NOT
-- claim full 256x256 tile-scale board behavior.

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

entity tb_window3x3_stream_3chan_flattened is
end entity tb_window3x3_stream_3chan_flattened;

architecture sim of tb_window3x3_stream_3chan_flattened is

    constant CLK_PERIOD  : time     := 10 ns;
    constant IMG_W       : positive := BLOCK_SIZE;               -- 6
    constant NUM_WINDOWS : positive := (IMG_W - 2) * (IMG_W - 2);  -- 16
    constant MAX_CYCLES  : integer := (IMG_W * IMG_W) + 50;  -- generous drain margin

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    signal valid_out : std_logic;
    signal s_y0, s_y1, s_y2, s_y3, s_y4, s_y5, s_y6, s_y7, s_y8, s_y9,
           s_y10, s_y11, s_y12, s_y13, s_y14, s_y15, s_y16, s_y17, s_y18, s_y19,
           s_y20, s_y21, s_y22, s_y23, s_y24, s_y25, s_y26, s_y27, s_y28, s_y29,
           s_y30, s_y31 : signed(47 downto 0);

begin

    clk <= not clk after CLK_PERIOD / 2;

    -- ================================================================
    -- DUT: the integrated front-end + arithmetic-core skeleton.
    -- ================================================================
    dut : entity work.first_layer_32out_folded_arithmetic_core_streaming_front_end
        generic map (IMG_WIDTH => IMG_W)
        port map (
            clk => clk, rst => rst, valid_in => valid_in,
            pixel_c0 => pixel_c0, pixel_c1 => pixel_c1, pixel_c2 => pixel_c2,
            valid_out => valid_out,
            y0 => s_y0, y1 => s_y1, y2 => s_y2, y3 => s_y3, y4 => s_y4,
            y5 => s_y5, y6 => s_y6, y7 => s_y7, y8 => s_y8, y9 => s_y9,
            y10 => s_y10, y11 => s_y11, y12 => s_y12, y13 => s_y13, y14 => s_y14,
            y15 => s_y15, y16 => s_y16, y17 => s_y17, y18 => s_y18, y19 => s_y19,
            y20 => s_y20, y21 => s_y21, y22 => s_y22, y23 => s_y23, y24 => s_y24,
            y25 => s_y25, y26 => s_y26, y27 => s_y27, y28 => s_y28, y29 => s_y29,
            y30 => s_y30, y31 => s_y31
        );

    stim : process

        variable out_count      : integer := 0;   -- window count (0..15)
        variable total_cycles   : integer := 0;
        variable pixels_sent    : integer := 0;
        variable first_valid_at : integer := -1;   -- cycle of first valid_out pulse
        variable last_valid_at  : integer := -1;   -- cycle of last valid_out pulse

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
                   " (@ cycle " & integer'image(total_cycles) &
                   ", pixels_sent=" & integer'image(pixels_sent) & ")";
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

        -- ---- Stream the REAL 6x6 UAVSAR-tile-derived block, one pixel --
        -- triplet per clock (row-major, from real_tile_stimulus_pkg).
        valid_in <= '1';
        for i in 0 to (IMG_W * IMG_W) - 1 loop
            pixel_c0 <= to_signed(REAL_TILE_CH0(i), 8);
            pixel_c1 <= to_signed(REAL_TILE_CH1(i), 8);
            pixel_c2 <= to_signed(REAL_TILE_CH2(i), 8);
            wait until rising_edge(clk);
            wait for 1 ns;
            total_cycles := total_cycles + 1;
            pixels_sent  := pixels_sent + 1;

            if valid_out = '1' then
                out_count := out_count + 1;
                if first_valid_at = -1 then
                    first_valid_at := total_cycles;
                end if;
                last_valid_at := total_cycles;
                check_output(out_count);
            end if;
        end loop;
        valid_in <= '0';

        -- ---- Drain: flush the front end's + arithmetic core's combined --
        -- pipeline for the last few in-flight windows.
        while out_count < NUM_WINDOWS loop
            assert total_cycles <= MAX_CYCLES
                report "FAIL: did not observe all " & integer'image(NUM_WINDOWS) &
                       " windows within " & integer'image(MAX_CYCLES) &
                       " cycles (runaway or stalled pipeline?)"
                severity failure;

            wait until rising_edge(clk);
            wait for 1 ns;
            total_cycles := total_cycles + 1;

            if valid_out = '1' then
                out_count := out_count + 1;
                last_valid_at := total_cycles;
                check_output(out_count);
            end if;
        end loop;

        -- ---- Final checks --------------------------------------------
        assert out_count = NUM_WINDOWS
            report "FAIL: expected exactly " & integer'image(NUM_WINDOWS) &
                   " valid windows, observed " & integer'image(out_count)
            severity failure;

        report "MEASURED: first valid_out at cycle " & integer'image(first_valid_at) &
               " (fill latency); last valid_out at cycle " & integer'image(last_valid_at) &
               "; total pixels streamed = " & integer'image(IMG_W * IMG_W) &
               "; total cycles (stream + drain) = " & integer'image(total_cycles)
               severity note;

        report "=== All tb_window3x3_stream_3chan_flattened (integrated front-end + core) tests PASSED ===" &
               "  (" & integer'image(NUM_WINDOWS) & " REAL windows x 32 kernels = " &
               integer'image(NUM_WINDOWS * 32) & " / " & integer'image(NUM_WINDOWS * 32) &
               " outputs match the SAME real-tile Q.20 golden vectors already used by the" &
               " direct-parallel single-shot and resource-shared verifications)" severity note;
        report "    NOTE: Streaming FRONT-END + EXISTING arithmetic-core feasibility skeleton." &
               " Not full-tile board behavior, not full U-Net inference, not board-tested, not a" &
               " measured speedup/power claim." severity note;

        wait;
    end process stim;

end architecture sim;
