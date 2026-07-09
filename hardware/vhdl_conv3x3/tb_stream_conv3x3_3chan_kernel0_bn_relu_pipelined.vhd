-- tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd
-- Self-checking testbench for stream_conv3x3_3chan_kernel0_bn_relu_pipelined.
--
-- Drives the canonical 5x5x3 toy patch (same stimulus as every other
-- testbench in this repo) and checks all 9 valid output positions against
-- the SAME EXACT Q.16 fixed-point integer golden values from
-- first_conv_bn_relu_kernel0_pkg used by the unpipelined design's
-- testbench (tb_stream_conv3x3_3chan_kernel0_bn_relu.vhd) -- the pipelined
-- design must produce IDENTICAL final output values; only the timing of
-- when valid_out pulses changes (one cycle later per output, from the
-- extra pipeline register).
--
-- ---- Source of weights/constants -----------------------------------------
-- Checkpoint : models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
-- Tensor key : enc1.block.0.weight (kernel 0), folded with enc1.block.1
--              (BatchNorm2d, eval-mode running stats)
-- Fixed pt.  : Q.16 (signed, 16 fractional bits); SCALE_FX, BIAS_FX from
--              first_conv_bn_relu_kernel0_pkg (UNCHANGED from the
--              unpipelined design -- no new golden arithmetic)
--
-- ---- Input patch --------------------------------------------------------
-- Canonical 5x5x3 toy patch, identical to every prior testbench:
--   Channel 0: 1..25 row-major
--   Channel 1: 2 x channel 0  (2..50)
--   Channel 2: -1 x channel 0  (-1..-25)
--
-- ---- Timing (6-cycle latency: 4 from stream_conv3x3_3chan_cell + 2 for
-- the now-split multiply / bias-add+ReLU stages -- ONE MORE cycle than
-- the unpipelined design's 5-cycle latency) --------------------------------
-- A window (r,c) (r,c in 0..2) completes on main-loop pixel index
-- 5r+c+13, and becomes visible on valid_out exactly 6 cycles later:
--   Outputs 1-3 (windows (0,0),(0,1),(0,2)): main loop i=19, 20, 21
--   Outputs 4-5 (windows (1,0),(1,1)):       main loop i=24, 25
--   Outputs 6-9 (windows (1,2),(2,0),(2,1),(2,2)): drain d=1, 4, 5, 6
--     (d=2, d=3 produce no output; 6-clock drain needed, one more than
--     the unpipelined design's 5-clock drain)
--
-- ---- What this validates -----------------------------------------------
-- Confirms stream_conv3x3_3chan_kernel0_bn_relu_pipelined produces, in ONE
-- streaming pass, the SAME 9 Q.16 fixed-point values as the unpipelined
-- design and the same Python golden vectors -- only latency changed. It
-- does NOT validate the other 31 output channels, does NOT implement
-- padding, and has NOT been run on real FPGA hardware -- this is a
-- simulation-only functional check of the kernel-0 folded Conv-BN-ReLU
-- pipelined proof-of-concept.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_conv_bn_relu_kernel0_pkg.all;

entity tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined is
end entity tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined;

architecture sim of tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined is

    constant CLK_PERIOD : time     := 10 ns;
    constant IMG_W      : positive := 5;

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    signal valid_out    : std_logic;
    signal y_bn_relu_fx : signed(47 downto 0);

begin

    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.stream_conv3x3_3chan_kernel0_bn_relu_pipelined
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

        variable out_count : integer := 0;

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
            report "PASS output " & integer'image(out_num) &
                   ": y_bn_relu_fx = " & integer'image(got) &
                   "  (Q." & integer'image(FRAC_BITS) & " fixed-point, ReLU applied, pipelined)";
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

        -- ---- Stream 5x5 image (one pixel triplet per clock) ----------
        -- 6-cycle cell latency (4-cycle raw conv + 2-cycle pipelined
        -- BN+ReLU): first output visible at i=19 (one cycle later than
        -- the unpipelined design's first output at i=18).

        valid_in <= '1';
        for i in 1 to 25 loop
            pixel_c0 <= to_signed(i,    8);
            pixel_c1 <= to_signed(2*i,  8);
            pixel_c2 <= to_signed(-i,   8);
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

        report "=== All stream_conv3x3_3chan_kernel0_bn_relu_pipelined tests PASSED ===" &
               "  (9 / 9 outputs match Python Q.16 fixed-point golden vectors)"
               severity note;
        report "    Kernel 0 folded Conv-BN-ReLU (PIPELINED), enc1.block.0/1 from" &
               " alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt" severity note;
        report "    NOTE: kernel-0-only proof-of-concept. Not all 32 channels," &
               " not board-tested, not a measured speedup/power claim." severity note;

        wait;
    end process stim;

end architecture sim;
