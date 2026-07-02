-- tb_stream_conv3x3_3chan_4out_cell.vhd
-- Self-checking testbench for stream_conv3x3_3chan_4out_cell.
--
-- This is the first testbench for an actual multi-output first-layer
-- convolution HARDWARE PROTOTYPE (not a sequential verification harness
-- like tb_stream_conv3x3_3chan_kernels0_to3.vhd, which reuses one
-- single-output DUT and reloads weights between runs). Here the DUT itself
-- computes all four kernel outputs in parallel from one shared 3x3 window
-- per input channel.
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
-- Tensor key : enc1.block.0.weight  (shape [32, 3, 3, 3])
-- Output chs : 0, 1, 2, 3 (baked into stream_conv3x3_3chan_4out_cell via
--              first_layer_kernels0_to3_pkg -- not loaded via ports here)
--
-- ---- Input patch --------------------------------------------------------
-- Canonical 5x5x3 toy patch matching tb_stream_conv3x3_3chan_cell.vhd and
-- tb_stream_conv3x3_3chan_kernels0_to3.vhd:
--   Channel 0: 1..25 row-major
--   Channel 1: 2 x channel 0  (2..50)
--   Channel 2: -1 x channel 0  (-1..-25)
--
-- ---- Expected INT32 outputs (3x3 valid convolution, row-major) ----------
-- Same golden vectors as tb_stream_conv3x3_3chan_kernels0_to3.vhd, sourced
-- from first_layer_kernels0_to3_pkg's KERNEL{0,1,2,3}_EXPECTED constants.
--
-- ---- Timing (4-cycle cell latency, single pass) --------------------------
--   Outputs 1-3: main loop i=17, 18, 19
--   Outputs 4-6: main loop i=22, 23, 24
--   Outputs 7-9: drain d=2, d=3, d=4  (4-clock drain; d=1 is empty)
-- Unlike tb_stream_conv3x3_3chan_kernels0_to3.vhd, there is only ONE pass
-- over the image: y0, y1, y2, y3 all become valid together on every
-- valid_out pulse, so all four kernels are checked per output position.
--
-- ---- What this validates -----------------------------------------------
-- This testbench confirms that stream_conv3x3_3chan_4out_cell produces, in
-- ONE streaming pass, the same 9x4=36 INT32 values that the sequential
-- kernels0_to3 flow and the original Python golden-vector script produce.
-- It does NOT validate full quantized U-Net inference (BatchNorm is not
-- applied), does NOT cover the remaining 28 of 32 first-layer output
-- channels, and has NOT been run on real FPGA hardware -- this is a
-- simulation-only functional check of the 4-output prototype.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_kernels0_to3_pkg.all;

entity tb_stream_conv3x3_3chan_4out_cell is
end entity tb_stream_conv3x3_3chan_4out_cell;

architecture sim of tb_stream_conv3x3_3chan_4out_cell is

    constant CLK_PERIOD : time     := 10 ns;
    constant IMG_W      : positive := 5;

    -- Testbench-driven signals
    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    -- DUT outputs
    signal valid_out : std_logic;
    signal s_y0, s_y1, s_y2, s_y3 : signed(31 downto 0);

begin

    -- ----------------------------------------------------------------
    -- Free-running clock
    -- ----------------------------------------------------------------
    clk <= not clk after CLK_PERIOD / 2;

    -- ----------------------------------------------------------------
    -- DUT: stream_conv3x3_3chan_4out_cell
    -- ----------------------------------------------------------------
    dut : entity work.stream_conv3x3_3chan_4out_cell
        generic map (IMG_WIDTH => IMG_W)
        port map (
            clk      => clk,
            rst      => rst,
            valid_in => valid_in,
            pixel_c0 => pixel_c0,
            pixel_c1 => pixel_c1,
            pixel_c2 => pixel_c2,
            valid_out => valid_out,
            y0 => s_y0,
            y1 => s_y1,
            y2 => s_y2,
            y3 => s_y3
        );

    -- ----------------------------------------------------------------
    -- Stimulus and self-checking
    -- ----------------------------------------------------------------
    stim : process

        variable out_count : integer := 0;

        procedure check_output(out_num : integer) is
        begin
            assert to_integer(s_y0) = KERNEL0_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) &
                       " kernel 0: y0 = " & integer'image(to_integer(s_y0)) &
                       "  expected " & integer'image(KERNEL0_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y1) = KERNEL1_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) &
                       " kernel 1: y1 = " & integer'image(to_integer(s_y1)) &
                       "  expected " & integer'image(KERNEL1_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y2) = KERNEL2_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) &
                       " kernel 2: y2 = " & integer'image(to_integer(s_y2)) &
                       "  expected " & integer'image(KERNEL2_EXPECTED(out_num - 1))
                severity failure;
            assert to_integer(s_y3) = KERNEL3_EXPECTED(out_num - 1)
                report "FAIL output " & integer'image(out_num) &
                       " kernel 3: y3 = " & integer'image(to_integer(s_y3)) &
                       "  expected " & integer'image(KERNEL3_EXPECTED(out_num - 1))
                severity failure;

            report "PASS output " & integer'image(out_num) &
                   ": y0=" & integer'image(to_integer(s_y0)) &
                   " y1=" & integer'image(to_integer(s_y1)) &
                   " y2=" & integer'image(to_integer(s_y2)) &
                   " y3=" & integer'image(to_integer(s_y3));
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
            report "FAIL: expected exactly 9 valid output positions, observed " &
                   integer'image(out_count)
            severity failure;

        report "=== All stream_conv3x3_3chan_4out_cell tests PASSED ===" &
               "  (9 positions x 4 kernels = 36 / 36 outputs match Python golden vectors)"
               severity note;
        report "    Kernels: enc1.block.0.weight channels 0-3 from" &
               " alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt" severity note;
        report "    NOTE: 4-output first-layer convolution PROTOTYPE only." &
               " Not full U-Net inference, not board-tested, BatchNorm not folded." severity note;

        wait;
    end process stim;

end architecture sim;
