-- tb_stream_conv3x3_cell.vhd
-- Self-checking testbench for stream_conv3x3_cell.
--
-- Input image (5 × 5, row-major, pixel values 1..25):
--    1  2  3  4  5
--    6  7  8  9 10
--   11 12 13 14 15
--   16 17 18 19 20
--   21 22 23 24 25
--
-- Kernel (Sobel-like vertical edge detector):
--    1  0 -1
--    1  0 -1
--    1  0 -1
--
-- Bias: 0
--
-- Expected y for all 9 valid windows: -6
--   Proof: every 3×3 window extracted from this image has consecutive-integer
--   columns, so left_col_sum - right_col_sum = (a + a+5 + a+10) - (a+2 + a+7 + a+12)
--   = 3*(−2) = -6 for any starting pixel a.
--
-- ---- Timing -----------------------------------------------------------
-- The cell has two submodule stages connected in series:
--
--   window3x3_stream (combinational from registered line buffers; 0 extra latency
--   but its registered output is read by the dot product on the FOLLOWING clock)
--       +
--   conv3x3_dot_pipelined (3-stage registered pipeline)
--
-- Net cell latency: 3 clock cycles from triggering pixel to visible output.
--
--   Clock  Input pixel  win_valid  Stage 1  Stage 2  Stage 3  cell valid_out
--   -----  -----------  ---------  -------  -------  -------  -------------
--   13     pixel 13     '1' (win1)  0        0        0         '0'
--   14     pixel 14     '1' (win2)  win1     0        0         '0'
--   15     pixel 15     '1' (win3)  win2     win1     0         '0'
--   16     pixel 16     '0'        win3     win2     win1       '1' ← output 1
--   17     pixel 17     '0'        0        win3     win2       '1' ← output 2
--   18     pixel 18     '1' (win4)  0        0        win3      '1' ← output 3
--   ...
--   21     pixel 21     '0'        win6     win5     win4       '1' ← output 4
--   22     pixel 22     '0'        0        win6     win5       '1' ← output 5
--   23     pixel 23     '1' (win7)  0        0        win6      '1' ← output 6
--   ...
--   26     (drain d=1)  '0'        win9     win8     win7       '1' ← output 7
--   27     (drain d=2)  '0'        0        win9     win8       '1' ← output 8
--   28     (drain d=3)  '0'        0        0        win9       '1' ← output 9
--
-- Outputs 7-9 arrive AFTER the 25-pixel loop ends.
-- A 3-clock drain phase collects them.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity tb_stream_conv3x3_cell is
end entity tb_stream_conv3x3_cell;

architecture sim of tb_stream_conv3x3_cell is

    constant CLK_PERIOD : time    := 10 ns;   -- 100 MHz
    constant IMG_W      : positive := 5;

    -- Testbench-driven signals
    signal clk       : std_logic := '0';
    signal rst       : std_logic := '1';
    signal valid_in  : std_logic := '0';
    signal pixel_in  : signed(7 downto 0) := (others => '0');

    -- Kernel (fixed for this testbench: vertical Sobel-like)
    signal s_w0, s_w3, s_w6 : signed(7 downto 0) := to_signed( 1, 8);   -- left  col
    signal s_w1, s_w4, s_w7 : signed(7 downto 0) := to_signed( 0, 8);   -- mid   col
    signal s_w2, s_w5, s_w8 : signed(7 downto 0) := to_signed(-1, 8);   -- right col
    signal s_bias            : signed(31 downto 0) := (others => '0');

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
    dut : entity work.stream_conv3x3_cell
        generic map (IMG_WIDTH => IMG_W)
        port map (
            clk       => clk,
            rst       => rst,
            valid_in  => valid_in,
            pixel_in  => pixel_in,
            w0 => s_w0,  w1 => s_w1,  w2 => s_w2,
            w3 => s_w3,  w4 => s_w4,  w5 => s_w5,
            w6 => s_w6,  w7 => s_w7,  w8 => s_w8,
            bias      => s_bias,
            valid_out => valid_out,
            y         => s_y
        );

    -- ----------------------------------------------------------------
    -- Stimulus and self-checking
    -- ----------------------------------------------------------------
    stim : process
        variable out_count : integer := 0;
        constant EXPECTED_Y  : integer := -6;
        constant EXPECTED_N  : integer := 9;

        -- Check one output and report
        procedure check_output(out_num : integer) is
        begin
            assert to_integer(s_y) = EXPECTED_Y
                report "FAIL output " & integer'image(out_num) &
                       ": y = " & integer'image(to_integer(s_y)) &
                       "  (expected " & integer'image(EXPECTED_Y) & ")"
                severity failure;
            report "PASS output " & integer'image(out_num) &
                   " (stream_conv3x3_cell): y = " & integer'image(to_integer(s_y));
        end procedure;

    begin
        -- ---- Reset (3 clocks with rst='1') -------------------------
        rst      <= '1';
        valid_in <= '0';
        pixel_in <= (others => '0');
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst  <= '0';
        wait until rising_edge(clk);
        wait for 1 ns;

        -- ---- Stream 5×5 image (pixels 1..25, one per clock) --------
        -- Registered outputs are sampled 1 ns after each rising edge.
        -- See the timing table in the header comment.
        --
        -- During this loop:
        --   outputs 1-3 appear at i = 16, 17, 18
        --   outputs 4-6 appear at i = 21, 22, 23
        --   i = 24, 25 produce valid_out = '0' (dot product still processing)

        valid_in <= '1';
        for i in 1 to 25 loop
            pixel_in <= to_signed(i, 8);
            wait until rising_edge(clk);
            wait for 1 ns;   -- let registered outputs settle

            if valid_out = '1' then
                out_count := out_count + 1;
                check_output(out_count);
            end if;
        end loop;

        -- ---- Drain (3 clocks to flush outputs 7, 8, 9) -------------
        -- Pixel 23 triggered window 7 (cell valid_out='1' at drain d=1).
        -- Pixel 24 triggered window 8 (cell valid_out='1' at drain d=2).
        -- Pixel 25 triggered window 9 (cell valid_out='1' at drain d=3).
        -- Three drain clocks are exactly sufficient to collect all three.

        valid_in <= '0';
        for d in 1 to 3 loop
            wait until rising_edge(clk);
            wait for 1 ns;

            if valid_out = '1' then
                out_count := out_count + 1;
                check_output(out_count);
            end if;
        end loop;

        -- ---- Final checks -------------------------------------------
        assert out_count = EXPECTED_N
            report "FAIL: expected exactly " & integer'image(EXPECTED_N) &
                   " valid outputs for 5x5 image, observed " &
                   integer'image(out_count)
            severity failure;

        report "=== All stream_conv3x3_cell tests PASSED ===  (" &
               integer'image(out_count) & " / " & integer'image(EXPECTED_N) &
               " outputs, all y = " & integer'image(EXPECTED_Y) & ")" severity note;

        wait;
    end process stim;

end architecture sim;
