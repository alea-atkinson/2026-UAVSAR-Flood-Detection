-- tb_stream_conv3x3_3chan_realweights.vhd
-- Self-checking testbench for stream_conv3x3_3chan_cell using REAL model weights.
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
-- Tensor key : enc1.block.0.weight  (shape [32, 3, 3, 3])
-- Output ch  : 0
-- Generated  : scripts/export_first_layer_conv3x3_vhdl_vectors.py
-- Vectors    : hardware/vhdl_conv3x3/test_vectors/first_layer_kernel0_int8.json
--
-- ---- Quantization -------------------------------------------------------
-- Scheme : symmetric per-tensor INT8 for weights
-- scale_w: max(|w_float|) / 127  =  0.17280057 / 127  =  0.00136063
-- w_int8 : round(w_float / scale_w), clipped to [-127, 127]
--
-- The first Conv2d in this U-Net has NO direct bias
-- (Conv2d → BatchNorm; BN params are NOT folded here).
-- Therefore bias_int32 = 0.
--
-- ---- INT8 kernel (output channel 0) ------------------------------------
--   Channel 0 (row-major, w0..w8):  110  117  -38  127  -28   33  -66   86  124
--   Channel 1 (row-major, w0..w8): -103  120   24  102   20   67  -19  108   20
--   Channel 2 (row-major, w0..w8):  -68   31  -72  -21  -59   92 -112  -66  -44
--
-- ---- Input patch --------------------------------------------------------
-- Canonical 5×5×3 toy patch matching tb_stream_conv3x3_3chan_cell.vhd:
--   Channel 0: 1..25 row-major
--   Channel 1: 2 × channel 0  (2..50)
--   Channel 2: -1 × channel 0  (-1..-25)
--
-- ---- Expected INT32 outputs (3×3 valid convolution, row-major) ----------
--   y[0,0]=11287  y[0,1]=12749  y[0,2]=14211
--   y[1,0]=18597  y[1,1]=20059  y[1,2]=21521
--   y[2,0]=25907  y[2,1]=27369  y[2,2]=28831
--
-- Python-verified:
--   y_float_ref[0,0] ≈ 15.40  (y_int32 × scale_w ≈ 15.36, max err ≈ 0.13)
--
-- ---- Timing (4-cycle cell latency) -------------------------------------
-- Identical timing to tb_stream_conv3x3_3chan_cell.vhd:
--   Outputs 1-3: main loop i=17, 18, 19
--   Outputs 4-6: main loop i=22, 23, 24
--   Outputs 7-9: drain d=2, d=3, d=4  (4-clock drain; d=1 is empty)
--
-- ---- What this validates -----------------------------------------------
-- This testbench confirms that the VHDL stream_conv3x3_3chan_cell integer
-- datapath produces the SAME INT32 values as the Python golden-vector script
-- when driven with real trained-model INT8 weights.  It does NOT validate
-- full quantized U-Net inference (BatchNorm is not applied).

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity tb_stream_conv3x3_3chan_realweights is
end entity tb_stream_conv3x3_3chan_realweights;

architecture sim of tb_stream_conv3x3_3chan_realweights is

    constant CLK_PERIOD : time     := 10 ns;
    constant IMG_W      : positive := 5;

    -- Testbench-driven signals
    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    -- ---- Real INT8 kernel weights (enc1.block.0.weight, output channel 0) ----
    --
    -- Channel 0  (w0..w8 = rows top→bottom, left→right):
    --   110  117  -38   |  127  -28   33   |  -66   86  124
    signal c0_w0 : signed(7 downto 0) := to_signed( 110, 8);
    signal c0_w1 : signed(7 downto 0) := to_signed( 117, 8);
    signal c0_w2 : signed(7 downto 0) := to_signed( -38, 8);
    signal c0_w3 : signed(7 downto 0) := to_signed( 127, 8);
    signal c0_w4 : signed(7 downto 0) := to_signed( -28, 8);
    signal c0_w5 : signed(7 downto 0) := to_signed(  33, 8);
    signal c0_w6 : signed(7 downto 0) := to_signed( -66, 8);
    signal c0_w7 : signed(7 downto 0) := to_signed(  86, 8);
    signal c0_w8 : signed(7 downto 0) := to_signed( 124, 8);

    -- Channel 1:
    --  -103  120   24   |  102   20   67   |  -19  108   20
    signal c1_w0 : signed(7 downto 0) := to_signed(-103, 8);
    signal c1_w1 : signed(7 downto 0) := to_signed( 120, 8);
    signal c1_w2 : signed(7 downto 0) := to_signed(  24, 8);
    signal c1_w3 : signed(7 downto 0) := to_signed( 102, 8);
    signal c1_w4 : signed(7 downto 0) := to_signed(  20, 8);
    signal c1_w5 : signed(7 downto 0) := to_signed(  67, 8);
    signal c1_w6 : signed(7 downto 0) := to_signed( -19, 8);
    signal c1_w7 : signed(7 downto 0) := to_signed( 108, 8);
    signal c1_w8 : signed(7 downto 0) := to_signed(  20, 8);

    -- Channel 2:
    --   -68   31  -72   |  -21  -59   92   | -112  -66  -44
    signal c2_w0 : signed(7 downto 0) := to_signed( -68, 8);
    signal c2_w1 : signed(7 downto 0) := to_signed(  31, 8);
    signal c2_w2 : signed(7 downto 0) := to_signed( -72, 8);
    signal c2_w3 : signed(7 downto 0) := to_signed( -21, 8);
    signal c2_w4 : signed(7 downto 0) := to_signed( -59, 8);
    signal c2_w5 : signed(7 downto 0) := to_signed(  92, 8);
    signal c2_w6 : signed(7 downto 0) := to_signed(-112, 8);
    signal c2_w7 : signed(7 downto 0) := to_signed( -66, 8);
    signal c2_w8 : signed(7 downto 0) := to_signed( -44, 8);

    -- bias_int32 = 0  (Conv2d has no direct bias; uses BatchNorm instead)
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
    -- DUT: stream_conv3x3_3chan_cell
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

        -- ---- Golden expected INT32 outputs (row-major, output_index 1..9) ----
        -- Source: hardware/vhdl_conv3x3/test_vectors/first_layer_kernel0_expected_outputs.csv
        -- Generated by: scripts/export_first_layer_conv3x3_vhdl_vectors.py
        --
        --   output_index  row  col   y_int32
        --   1             0    0     11287
        --   2             0    1     12749
        --   3             0    2     14211
        --   4             1    0     18597
        --   5             1    1     20059
        --   6             1    2     21521
        --   7             2    0     25907
        --   8             2    1     27369
        --   9             2    2     28831
        type expected_t is array (1 to 9) of integer;
        constant EXPECTED : expected_t := (
            11287, 12749, 14211,
            18597, 20059, 21521,
            25907, 27369, 28831
        );

        variable out_count : integer := 0;

        procedure check_output(out_num : integer) is
        begin
            assert to_integer(s_y) = EXPECTED(out_num)
                report "FAIL output " & integer'image(out_num) &
                       " (realweights): y = " & integer'image(to_integer(s_y)) &
                       "  expected " & integer'image(EXPECTED(out_num))
                severity failure;
            report "PASS output " & integer'image(out_num) &
                   " (stream_conv3x3_3chan_realweights): y = " &
                   integer'image(to_integer(s_y));
        end procedure;

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

        -- ---- Stream 5×5 image (one pixel triplet per clock) ----------
        -- Channel 0: pixel i  (1..25)
        -- Channel 1: 2*i      (2..50, fits INT8)
        -- Channel 2: -i       (-1..-25, fits INT8)
        --
        -- 4-cycle cell latency (window gen + 3-stage dot product + final sum):
        --   first output visible at i=17.
        --
        -- Outputs during main loop: i=17,18,19 (windows 1-3)
        --                           i=22,23,24 (windows 4-6)
        -- i=25: no output (windows 7-9 still in pipeline)

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

        -- ---- Drain: 4 clocks to flush windows 7, 8, 9 ---------------
        -- Windows 7-9 triggered by pixels 23, 24, 25 exit at
        -- drain clocks d=2, d=3, d=4.  d=1 is empty.

        valid_in <= '0';
        for d in 1 to 4 loop
            wait until rising_edge(clk);
            wait for 1 ns;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_output(out_count);
            end if;
        end loop;

        -- ---- Final checks --------------------------------------------
        assert out_count = 9
            report "FAIL: expected exactly 9 valid outputs, observed " &
                   integer'image(out_count)
            severity failure;

        report "=== All stream_conv3x3_3chan_realweights tests PASSED ===" &
               "  (9 / 9 outputs match Python golden vectors)" severity note;
        report "    Weights: enc1.block.0.weight channel 0 from" &
               " alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt" severity note;

        wait;
    end process stim;

end architecture sim;
