-- tb_stream_conv3x3_3chan_kernels0_to3.vhd
-- Self-checking testbench for stream_conv3x3_3chan_cell using REAL model
-- weights for first-layer output channels (kernels) 0, 1, 2, 3.
--
-- Generalization of tb_stream_conv3x3_3chan_realweights.vhd (kernel 0 only)
-- to multiple output channels. A single DUT instance is reused sequentially:
-- for each kernel, the testbench resets the pipeline, loads that kernel's
-- INT8 weights (from first_layer_kernels0_to3_pkg), streams the same
-- canonical 5x5x3 toy input patch, and checks the 9 resulting INT32 outputs
-- against the Python golden vectors baked into the package.
--
-- This is a VERIFICATION testbench only -- it does not describe a new
-- synthesized multi-kernel accelerator. It confirms that the same
-- stream_conv3x3_3chan_cell integer datapath produces correct results when
-- reloaded with different trained-model kernels, one at a time.
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
-- Tensor key : enc1.block.0.weight  (shape [32, 3, 3, 3])
-- Output chs : 0, 1, 2, 3
-- Generated  : scripts/export_first_layer_multi_kernel_vhdl_vectors.py
-- Vectors    : hardware/vhdl_conv3x3/test_vectors/multi_kernel_first_layer/
-- Constants  : hardware/vhdl_conv3x3/first_layer_kernels0_to3_pkg.vhd
--
-- ---- Quantization -------------------------------------------------------
-- Scheme : symmetric per-tensor INT8 for weights (scale computed per kernel)
-- The first Conv2d in this U-Net has NO direct bias (Conv2d -> BatchNorm;
-- BN params are NOT folded here). Therefore bias_int32 = 0 for every kernel.
--
-- ---- Input patch (same for every kernel) ---------------------------------
-- Canonical 5x5x3 toy patch matching tb_stream_conv3x3_3chan_cell.vhd:
--   Channel 0: 1..25 row-major
--   Channel 1: 2 x channel 0  (2..50)
--   Channel 2: -1 x channel 0  (-1..-25)
--
-- ---- Timing (per kernel run, 4-cycle cell latency) -----------------------
-- Identical per-run timing to tb_stream_conv3x3_3chan_cell.vhd:
--   Outputs 1-3: main loop i=17, 18, 19
--   Outputs 4-6: main loop i=22, 23, 24
--   Outputs 7-9: drain d=2, d=3, d=4  (4-clock drain; d=1 is empty)
-- The pipeline is reset (3 clocks) before each kernel's run so no state
-- leaks between kernels.
--
-- ---- What this validates -----------------------------------------------
-- This testbench confirms that the VHDL stream_conv3x3_3chan_cell integer
-- datapath produces the SAME INT32 values as the Python golden-vector script
-- for FOUR different trained-model kernels driven sequentially through one
-- DUT instance. It does NOT validate full quantized U-Net inference
-- (BatchNorm is not applied), and it does NOT cover all 32 first-layer
-- output channels.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_kernels0_to3_pkg.all;

entity tb_stream_conv3x3_3chan_kernels0_to3 is
end entity tb_stream_conv3x3_3chan_kernels0_to3;

architecture sim of tb_stream_conv3x3_3chan_kernels0_to3 is

    constant CLK_PERIOD : time     := 10 ns;
    constant IMG_W      : positive := 5;

    -- Testbench-driven signals
    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    -- Weight ports; reloaded between kernel runs
    signal c0_w0, c0_w1, c0_w2, c0_w3, c0_w4, c0_w5, c0_w6, c0_w7, c0_w8 : signed(7 downto 0) := (others => '0');
    signal c1_w0, c1_w1, c1_w2, c1_w3, c1_w4, c1_w5, c1_w6, c1_w7, c1_w8 : signed(7 downto 0) := (others => '0');
    signal c2_w0, c2_w1, c2_w2, c2_w3, c2_w4, c2_w5, c2_w6, c2_w7, c2_w8 : signed(7 downto 0) := (others => '0');

    signal s_bias : signed(31 downto 0) := to_signed(0, 32);

    -- DUT outputs
    signal valid_out : std_logic;
    signal s_y       : signed(31 downto 0);

begin

    -- ----------------------------------------------------------------
    -- Free-running clock
    -- ----------------------------------------------------------------
    clk <= not clk after CLK_PERIOD / 2;

    -- ----------------------------------------------------------------
    -- DUT: stream_conv3x3_3chan_cell (single instance, reused per kernel)
    -- ----------------------------------------------------------------
    dut : entity work.stream_conv3x3_3chan_cell
        generic map (IMG_WIDTH => IMG_W)
        port map (
            clk      => clk,
            rst      => rst,
            valid_in => valid_in,
            pixel_c0 => pixel_c0,
            pixel_c1 => pixel_c1,
            pixel_c2 => pixel_c2,
            c0_w0 => c0_w0,  c0_w1 => c0_w1,  c0_w2 => c0_w2,
            c0_w3 => c0_w3,  c0_w4 => c0_w4,  c0_w5 => c0_w5,
            c0_w6 => c0_w6,  c0_w7 => c0_w7,  c0_w8 => c0_w8,
            c1_w0 => c1_w0,  c1_w1 => c1_w1,  c1_w2 => c1_w2,
            c1_w3 => c1_w3,  c1_w4 => c1_w4,  c1_w5 => c1_w5,
            c1_w6 => c1_w6,  c1_w7 => c1_w7,  c1_w8 => c1_w8,
            c2_w0 => c2_w0,  c2_w1 => c2_w1,  c2_w2 => c2_w2,
            c2_w3 => c2_w3,  c2_w4 => c2_w4,  c2_w5 => c2_w5,
            c2_w6 => c2_w6,  c2_w7 => c2_w7,  c2_w8 => c2_w8,
            bias      => s_bias,
            valid_out => valid_out,
            y         => s_y
        );

    -- ----------------------------------------------------------------
    -- Stimulus and self-checking
    -- ----------------------------------------------------------------
    stim : process

        -- Cumulative pass/fail bookkeeping across all kernels
        variable total_out_count : integer := 0;

        -- Runs one kernel end-to-end: reset, load weights, stream the
        -- canonical 5x5x3 toy input, check all 9 outputs against EXPECTED.
        procedure run_kernel(
            kernel_id : in integer;
            ch0_w     : in int8_kernel_t;
            ch1_w     : in int8_kernel_t;
            ch2_w     : in int8_kernel_t;
            kbias     : in integer;
            expected  : in int32_outputs_t
        ) is
            variable out_count : integer := 0;
        begin
            -- ---- Reset (3 clocks), clear pipeline state from any prior kernel ----
            rst      <= '1';
            valid_in <= '0';
            pixel_c0 <= (others => '0');
            pixel_c1 <= (others => '0');
            pixel_c2 <= (others => '0');
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            rst <= '0';

            -- ---- Load this kernel's weights and bias ----
            c0_w0 <= to_signed(ch0_w(0), 8); c0_w1 <= to_signed(ch0_w(1), 8); c0_w2 <= to_signed(ch0_w(2), 8);
            c0_w3 <= to_signed(ch0_w(3), 8); c0_w4 <= to_signed(ch0_w(4), 8); c0_w5 <= to_signed(ch0_w(5), 8);
            c0_w6 <= to_signed(ch0_w(6), 8); c0_w7 <= to_signed(ch0_w(7), 8); c0_w8 <= to_signed(ch0_w(8), 8);

            c1_w0 <= to_signed(ch1_w(0), 8); c1_w1 <= to_signed(ch1_w(1), 8); c1_w2 <= to_signed(ch1_w(2), 8);
            c1_w3 <= to_signed(ch1_w(3), 8); c1_w4 <= to_signed(ch1_w(4), 8); c1_w5 <= to_signed(ch1_w(5), 8);
            c1_w6 <= to_signed(ch1_w(6), 8); c1_w7 <= to_signed(ch1_w(7), 8); c1_w8 <= to_signed(ch1_w(8), 8);

            c2_w0 <= to_signed(ch2_w(0), 8); c2_w1 <= to_signed(ch2_w(1), 8); c2_w2 <= to_signed(ch2_w(2), 8);
            c2_w3 <= to_signed(ch2_w(3), 8); c2_w4 <= to_signed(ch2_w(4), 8); c2_w5 <= to_signed(ch2_w(5), 8);
            c2_w6 <= to_signed(ch2_w(6), 8); c2_w7 <= to_signed(ch2_w(7), 8); c2_w8 <= to_signed(ch2_w(8), 8);

            s_bias <= to_signed(kbias, 32);

            wait until rising_edge(clk);
            wait for 1 ns;

            -- ---- Stream 5x5 image (one pixel triplet per clock) ----------
            valid_in <= '1';
            for i in 1 to 25 loop
                pixel_c0 <= to_signed(i,   8);
                pixel_c1 <= to_signed(2*i, 8);
                pixel_c2 <= to_signed(-i,  8);
                wait until rising_edge(clk);
                wait for 1 ns;

                if valid_out = '1' then
                    out_count := out_count + 1;
                    assert to_integer(s_y) = expected(out_count - 1)
                        report "FAIL kernel " & integer'image(kernel_id) &
                               " output " & integer'image(out_count) &
                               ": y = " & integer'image(to_integer(s_y)) &
                               "  expected " & integer'image(expected(out_count - 1))
                        severity failure;
                    report "PASS kernel " & integer'image(kernel_id) &
                           " output " & integer'image(out_count) &
                           ": y = " & integer'image(to_integer(s_y));
                end if;
            end loop;

            -- ---- Drain: 4 clocks to flush the last 3 windows -------------
            valid_in <= '0';
            for d in 1 to 4 loop
                wait until rising_edge(clk);
                wait for 1 ns;

                if valid_out = '1' then
                    out_count := out_count + 1;
                    assert to_integer(s_y) = expected(out_count - 1)
                        report "FAIL kernel " & integer'image(kernel_id) &
                               " output " & integer'image(out_count) &
                               ": y = " & integer'image(to_integer(s_y)) &
                               "  expected " & integer'image(expected(out_count - 1))
                        severity failure;
                    report "PASS kernel " & integer'image(kernel_id) &
                           " output " & integer'image(out_count) &
                           ": y = " & integer'image(to_integer(s_y));
                end if;
            end loop;

            assert out_count = 9
                report "FAIL kernel " & integer'image(kernel_id) &
                       ": expected exactly 9 valid outputs, observed " &
                       integer'image(out_count)
                severity failure;

            total_out_count := total_out_count + out_count;

            report "=== Kernel " & integer'image(kernel_id) &
                   " : all 9 outputs PASSED ===" severity note;
        end procedure run_kernel;

    begin
        run_kernel(0, KERNEL0_CH0_W, KERNEL0_CH1_W, KERNEL0_CH2_W, KERNEL0_BIAS, KERNEL0_EXPECTED);
        run_kernel(1, KERNEL1_CH0_W, KERNEL1_CH1_W, KERNEL1_CH2_W, KERNEL1_BIAS, KERNEL1_EXPECTED);
        run_kernel(2, KERNEL2_CH0_W, KERNEL2_CH1_W, KERNEL2_CH2_W, KERNEL2_BIAS, KERNEL2_EXPECTED);
        run_kernel(3, KERNEL3_CH0_W, KERNEL3_CH1_W, KERNEL3_CH2_W, KERNEL3_BIAS, KERNEL3_EXPECTED);

        assert total_out_count = 4 * 9
            report "FAIL: expected 36 total valid outputs across 4 kernels, observed " &
                   integer'image(total_out_count)
            severity failure;

        report "=== All stream_conv3x3_3chan_kernels0_to3 tests PASSED ===" &
               "  (4 kernels x 9 / 9 outputs match Python golden vectors)" severity note;
        report "    Weights: enc1.block.0.weight channels 0-3 from" &
               " alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt" severity note;

        wait;
    end process stim;

end architecture sim;
