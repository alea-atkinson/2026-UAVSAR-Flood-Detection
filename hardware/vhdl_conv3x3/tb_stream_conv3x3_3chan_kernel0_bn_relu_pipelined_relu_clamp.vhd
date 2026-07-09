-- tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vhd
-- *** SYNTHETIC RELU CLAMP UNIT TEST -- NOT a real UAVSAR tile, NOT real model inference. ***
--
-- Self-checking testbench for
-- stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.
--
-- Drives an ALL-ZERO synthetic 5x5x3 input patch (NOT the canonical toy
-- patch used by every other testbench in this repo) and checks all 9
-- valid output positions against the EXACT Q.16 fixed-point integer
-- golden values from first_conv_bn_relu_kernel0_relu_clamp_pkg (generated
-- by scripts/generate_bn_relu_fixed_point_vectors.py --kernel-id 0
-- --input-mode synthetic_relu_clamp). This compares fixed-point integers
-- to fixed-point integers -- NOT a float-vs-fixed-point comparison.
--
-- Purpose: kernel 0's and kernel 2's prior toy-patch tests both happened
-- to produce all-positive biased_fx values, so relu_fx == biased_fx for
-- every output and the ReLU clamp (negative) branch was never
-- numerically exercised. This synthetic all-zero input deterministically
-- forces raw_conv_int32 = 0 for every position (regardless of the
-- kernel's weights), so biased_fx = BIAS_FX = -936 (< 0) for all 9
-- positions -- exercising the clamp branch for every single output.
--
-- ---- Source of weights/constants -----------------------------------------
-- Checkpoint : models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
-- Tensor key : enc1.block.0.weight (kernel 0), folded with enc1.block.1
--              (BatchNorm2d, eval-mode running stats) -- SAME weights and
--              SAME SCALE_FX=135/BIAS_FX=-936 as kernel 0's other tests.
--
-- ---- Input patch (SYNTHETIC -- NOT a real UAVSAR tile) -------------------
-- All-zero 5x5x3 patch: pixel_c0 = pixel_c1 = pixel_c2 = 0 for all 25
-- streamed positions.
--
-- ---- Timing (6-cycle latency, IDENTICAL structure to the kernel0/kernel2
-- pipelined testbenches -- window-validity timing depends only on
-- IMG_WIDTH and cycle count, not on pixel values) ---------------------------
--   Outputs 1-3: main loop i=19, 20, 21
--   Outputs 4-5: main loop i=24, 25
--   Outputs 6-9: drain d=1, 4, 5, 6 (d=2, d=3 produce no output)
--
-- ---- What this validates -----------------------------------------------
-- Confirms stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp
-- produces, in ONE streaming pass, the same 9 Q.16 fixed-point values
-- (all zero) that scripts/generate_bn_relu_fixed_point_vectors.py
-- --kernel-id 0 --input-mode synthetic_relu_clamp computes -- i.e. that
-- the VHDL ReLU clamp (negative) branch correctly zeroes a negative
-- biased_fx value, using the SAME two-stage pipeline structure verified
-- for kernel 0's and kernel 2's toy-input tests. It does NOT represent a
-- real UAVSAR tile, does NOT validate the other 31 output channels, does
-- NOT implement padding, and has NOT been run on real FPGA hardware.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_conv_bn_relu_kernel0_relu_clamp_pkg.all;

entity tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp is
end entity tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp;

architecture sim of tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp is

    constant CLK_PERIOD : time     := 10 ns;
    constant IMG_W      : positive := 5;

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    -- Synthetic all-zero pixel stream (NOT the canonical toy patch)
    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    signal valid_out    : std_logic;
    signal y_bn_relu_fx : signed(47 downto 0);

begin

    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp
        generic map (IMG_WIDTH => IMG_W)
        port map (
            clk      => clk,
            rst      => rst,
            valid_in => valid_in,
            pixel_c0 => pixel_c0,
            pixel_c1 => pixel_c1,
            pixel_c2 => pixel_c2,
            valid_out    => valid_out,
            y_bn_relu_fx => y_bn_relu_fx
        );

    stim : process

        variable out_count     : integer := 0;
        variable clamped_count : integer := 0;

        procedure check_output(out_num : integer) is
            variable expected : integer;
            variable got      : integer;
        begin
            expected := K0_BN_RELU_EXPECTED_FX(out_num - 1);
            got      := to_integer(y_bn_relu_fx);
            assert got = expected
                report "FAIL output " & integer'image(out_num) &
                       ": y_bn_relu_fx = " & integer'image(got) &
                       "  expected " & integer'image(expected)
                severity failure;
            if got = 0 then
                clamped_count := clamped_count + 1;
            end if;
            report "PASS output " & integer'image(out_num) &
                   ": y_bn_relu_fx = " & integer'image(got) &
                   "  (Q." & integer'image(FRAC_BITS) & " fixed-point, ReLU-clamped, kernel 0, SYNTHETIC input)";
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

        -- ---- Stream 5x5 SYNTHETIC ALL-ZERO image (one pixel triplet
        -- per clock) -- 6-cycle cell latency (4-cycle raw conv + 2-cycle
        -- pipelined BN+ReLU): first output visible at i=19. Pixel values
        -- are irrelevant to window-validity timing (all zero here), only
        -- their count and cadence matter.

        valid_in <= '1';
        for i in 1 to 25 loop
            pixel_c0 <= to_signed(0, 8);
            pixel_c1 <= to_signed(0, 8);
            pixel_c2 <= to_signed(0, 8);
            wait until rising_edge(clk);
            wait for 1 ns;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_output(out_count);
            end if;
        end loop;

        -- ---- Drain: 6 clocks to flush the two pipelined BN+ReLU stages --
        valid_in <= '0';
        for d in 1 to 6 loop
            wait until rising_edge(clk);
            wait for 1 ns;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_output(out_count);
            end if;
        end loop;

        -- ---- Final checks --------------------------------------------
        assert out_count = 9
            report "FAIL: expected exactly 9 valid output positions, observed " &
                   integer'image(out_count)
            severity failure;

        assert clamped_count >= 1
            report "FAIL: expected at least 1 ReLU-clamped (zero) output, observed " &
                   integer'image(clamped_count)
            severity failure;

        report "=== ReLU clamp count: " & integer'image(clamped_count) &
               " / " & integer'image(out_count) & " outputs clamped to zero ===" severity note;

        report "=== All stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp tests PASSED ===" &
               "  (9 / 9 outputs match Python Q.16 fixed-point golden vectors, ReLU clamp branch exercised)"
               severity note;
        report "    Kernel 0 folded Conv-BN-ReLU (PIPELINED, SYNTHETIC ReLU clamp test), enc1.block.0/1 from" &
               " alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt" severity note;
        report "    NOTE: SYNTHETIC unit test input -- NOT a real UAVSAR tile, NOT real model inference." &
               " kernel-0-only proof-of-concept. Not all 32 channels, not board-tested," &
               " not a measured speedup/power claim." severity note;

        wait;
    end process stim;

end architecture sim;
