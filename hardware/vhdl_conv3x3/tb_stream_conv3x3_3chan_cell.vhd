-- tb_stream_conv3x3_3chan_cell.vhd
-- Self-checking testbench for stream_conv3x3_3chan_cell.
--
-- Input image setup:
--   Channel 0: 5×5 image, pixel values 1..25 (row-major)
--   Channel 1: 2 × channel 0  (values 2..50)
--   Channel 2: -1 × channel 0 (values -1..-25)
--
-- Kernel for each channel:
--    1  0 -1
--    1  0 -1
--    1  0 -1
--
-- Bias: 10
--
-- Expected outputs (all 9 valid windows):
--   y_c0 (zero-bias dot, channel 0):  -6   (left_col - right_col = -2 per row × 3 rows)
--   y_c1 (zero-bias dot, channel 1): -12   (2 × y_c0)
--   y_c2 (zero-bias dot, channel 2):  +6   (-1 × y_c0)
--   y    (bias + y_c0 + y_c1 + y_c2): 10 + (-6) + (-12) + 6 = -2
--
-- ---- Timing -------------------------------------------------------------
-- The cell adds one registered summation stage after the 3-stage dot products.
-- Total cell latency: 4 clock cycles from triggering pixel.
--
--  Clock  pixel  win_valid  dp_stg1  dp_stg2  dp_stg3  final_sum  valid_out
--  -----  -----  ---------  -------  -------  -------  ---------  ---------
--  13     p13    '1'(w1)     —        —        —         —          '0'
--  14     p14    '1'(w2)    w1        —        —         —          '0'
--  15     p15    '1'(w3)    w2       w1        —         —          '0'
--  16     p16    '0'        w3       w2       w1         —          '0'
--  17     p17    '0'         —       w3       w2        w1(y=-2)    '1' ← output 1
--  18     p18    '1'(w4)     —        —       w3        w2(y=-2)    '1' ← output 2
--  19     p19    '1'(w5)    w4        —        —        w3(y=-2)    '1' ← output 3
--  20     p20    '1'(w6)    w5       w4        —         —          '0'
--  21     p21    '0'        w6       w5       w4         —          '0'
--  22     p22    '0'         —       w6       w5        w4(y=-2)    '1' ← output 4
--  23     p23    '1'(w7)     —        —       w6        w5(y=-2)    '1' ← output 5
--  24     p24    '1'(w8)    w7        —        —        w6(y=-2)    '1' ← output 6
--  25     p25    '1'(w9)    w8       w7        —         —          '0'
--  [end of pixel loop; drain starts]
--  26     drain1  '0'       w9       w8       w7         —          '0'
--  27     drain2  '0'        —       w9       w8        w7(y=-2)    '1' ← output 7
--  28     drain3  '0'        —        —       w9        w8(y=-2)    '1' ← output 8
--  29     drain4  '0'        —        —        —        w9(y=-2)    '1' ← output 9
--
-- Outputs 1-6: collected during the 25-pixel loop (at i=17,18,19,22,23,24).
-- Outputs 7-9: collected during the 4-clock drain (at d=2,3,4; d=1 is empty).

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity tb_stream_conv3x3_3chan_cell is
end entity tb_stream_conv3x3_3chan_cell;

architecture sim of tb_stream_conv3x3_3chan_cell is

    constant CLK_PERIOD : time     := 10 ns;
    constant IMG_W      : positive := 5;

    -- Testbench-driven signals
    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

    -- Kernel: [1,0,-1; 1,0,-1; 1,0,-1] — same for all three channels
    signal s_w0 : signed(7 downto 0) := to_signed( 1, 8);
    signal s_w1 : signed(7 downto 0) := to_signed( 0, 8);
    signal s_w2 : signed(7 downto 0) := to_signed(-1, 8);
    signal s_w3 : signed(7 downto 0) := to_signed( 1, 8);
    signal s_w4 : signed(7 downto 0) := to_signed( 0, 8);
    signal s_w5 : signed(7 downto 0) := to_signed(-1, 8);
    signal s_w6 : signed(7 downto 0) := to_signed( 1, 8);
    signal s_w7 : signed(7 downto 0) := to_signed( 0, 8);
    signal s_w8 : signed(7 downto 0) := to_signed(-1, 8);

    signal s_bias : signed(31 downto 0) := to_signed(10, 32);

    -- DUT outputs
    signal valid_out : std_logic;
    signal s_y       : signed(31 downto 0);

begin

    -- ----------------------------------------------------------------
    -- Free-running clock
    -- ----------------------------------------------------------------
    clk <= not clk after CLK_PERIOD / 2;

    -- ----------------------------------------------------------------
    -- DUT
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
            -- Channel 0 weights
            c0_w0 => s_w0,  c0_w1 => s_w1,  c0_w2 => s_w2,
            c0_w3 => s_w3,  c0_w4 => s_w4,  c0_w5 => s_w5,
            c0_w6 => s_w6,  c0_w7 => s_w7,  c0_w8 => s_w8,
            -- Channel 1 weights (same kernel)
            c1_w0 => s_w0,  c1_w1 => s_w1,  c1_w2 => s_w2,
            c1_w3 => s_w3,  c1_w4 => s_w4,  c1_w5 => s_w5,
            c1_w6 => s_w6,  c1_w7 => s_w7,  c1_w8 => s_w8,
            -- Channel 2 weights (same kernel)
            c2_w0 => s_w0,  c2_w1 => s_w1,  c2_w2 => s_w2,
            c2_w3 => s_w3,  c2_w4 => s_w4,  c2_w5 => s_w5,
            c2_w6 => s_w6,  c2_w7 => s_w7,  c2_w8 => s_w8,
            bias      => s_bias,
            valid_out => valid_out,
            y         => s_y
        );

    -- ----------------------------------------------------------------
    -- Stimulus and self-checking
    -- ----------------------------------------------------------------
    stim : process
        variable out_count : integer := 0;
        constant EXPECTED_Y : integer := -2;
        constant EXPECTED_N : integer := 9;

        procedure check_output(out_num : integer) is
        begin
            assert to_integer(s_y) = EXPECTED_Y
                report "FAIL output " & integer'image(out_num) &
                       " (3chan_cell): y = " & integer'image(to_integer(s_y)) &
                       "  (expected " & integer'image(EXPECTED_Y) & ")"
                severity failure;
            report "PASS output " & integer'image(out_num) &
                   " (stream_conv3x3_3chan_cell): y = " & integer'image(to_integer(s_y));
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
        -- 4-cycle cell latency: first output appears at i=17.
        -- Outputs in this loop: i=17,18,19 (windows 1-3) and i=22,23,24 (windows 4-6).
        -- i=25 produces no output (windows 7-9 still in pipeline).

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
        -- Windows 7-9 were triggered by pixels 23, 24, 25.
        -- With 4-cycle latency they exit the pipeline at clocks 27, 28, 29
        -- (drain clocks d=2, d=3, d=4).  Drain clock d=1 (clock 26) is empty.

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
        assert out_count = EXPECTED_N
            report "FAIL: expected exactly " & integer'image(EXPECTED_N) &
                   " valid outputs for 5x5 image, observed " &
                   integer'image(out_count)
            severity failure;

        report "=== All stream_conv3x3_3chan_cell tests PASSED ===  (" &
               integer'image(out_count) & " / " & integer'image(EXPECTED_N) &
               " outputs, all y = " & integer'image(EXPECTED_Y) &
               " = bias(10) + y_c0(-6) + y_c1(-12) + y_c2(+6))" severity note;

        wait;
    end process stim;

end architecture sim;
