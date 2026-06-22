-- tb_window3x3_stream.vhd
-- Self-checking testbench for window3x3_stream.
--
-- Input image (5 × 5, row-major, pixel values 1..25):
--    1  2  3  4  5
--    6  7  8  9 10
--   11 12 13 14 15
--   16 17 18 19 20
--   21 22 23 24 25
--
-- Expected valid output windows (valid convolution, no padding, 3 × 3 output):
--
--   Window 1 (row-out=0, col-out=0) — triggered by pixel 13:
--    1  2  3
--    6  7  8
--   11 12 13
--
--   Window 2 (row-out=0, col-out=1) — triggered by pixel 14:
--    2  3  4
--    7  8  9
--   12 13 14
--
--   Window 3 (row-out=0, col-out=2) — triggered by pixel 15:
--    3  4  5
--    8  9 10
--   13 14 15
--
--   Window 4 (row-out=1, col-out=0) — triggered by pixel 18:
--    6  7  8
--   11 12 13
--   16 17 18
--
--   Window 5 (row-out=1, col-out=1) — triggered by pixel 19:
--    7  8  9
--   12 13 14
--   17 18 19
--
--   Window 6 (row-out=1, col-out=2) — triggered by pixel 20:
--    8  9 10
--   13 14 15
--   18 19 20
--
--   Window 7 (row-out=2, col-out=0) — triggered by pixel 23:
--   11 12 13
--   16 17 18
--   21 22 23
--
--   Window 8 (row-out=2, col-out=1) — triggered by pixel 24:
--   12 13 14
--   17 18 19
--   22 23 24
--
--   Window 9 (row-out=2, col-out=2) — triggered by pixel 25:
--   13 14 15
--   18 19 20
--   23 24 25
--
-- Timing:
--   Inputs are driven BEFORE the rising clock edge and held stable across it.
--   Registered DUT outputs are sampled 1 ns after the rising edge, ensuring
--   all delta-cycle updates have settled.
--
-- Pixel streaming: valid_in held '1' for all 25 pixels (one per clock).
-- This testbench checks all 9 valid windows exactly.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity tb_window3x3_stream is
end entity tb_window3x3_stream;

architecture sim of tb_window3x3_stream is

    constant CLK_PERIOD : time := 10 ns;   -- 100 MHz
    constant IMG_W      : positive := 5;

    -- Testbench-driven signals
    signal clk       : std_logic := '0';
    signal rst       : std_logic := '1';
    signal valid_in  : std_logic := '0';
    signal pixel_in  : signed(7 downto 0) := (others => '0');

    -- DUT outputs
    signal valid_out : std_logic;
    signal s_p0, s_p1, s_p2,
           s_p3, s_p4, s_p5,
           s_p6, s_p7, s_p8 : signed(7 downto 0);

begin

    -- ----------------------------------------------------------------
    -- Free-running clock
    -- ----------------------------------------------------------------
    clk <= not clk after CLK_PERIOD / 2;

    -- ----------------------------------------------------------------
    -- DUT
    -- ----------------------------------------------------------------
    dut : entity work.window3x3_stream
        generic map (IMG_WIDTH => IMG_W)
        port map (
            clk       => clk,
            rst       => rst,
            valid_in  => valid_in,
            pixel_in  => pixel_in,
            valid_out => valid_out,
            p0 => s_p0, p1 => s_p1, p2 => s_p2,
            p3 => s_p3, p4 => s_p4, p5 => s_p5,
            p6 => s_p6, p7 => s_p7, p8 => s_p8
        );

    -- ----------------------------------------------------------------
    -- Stimulus and self-checking
    -- ----------------------------------------------------------------
    stim : process
        -- Count valid windows seen so far
        variable win_count : integer := 0;

        -- Report and check a single pixel of the window
        procedure assert_pixel(
            win_num    : integer;
            port_name  : string;
            got        : signed(7 downto 0);
            expected   : integer
        ) is
        begin
            assert to_integer(got) = expected
                report "FAIL window " & integer'image(win_num) &
                       "  " & port_name &
                       " = " & integer'image(to_integer(got)) &
                       "  (expected " & integer'image(expected) & ")"
                severity failure;
        end procedure;

        -- Check an entire 3×3 window and report PASS
        procedure check_window(
            win_num      : integer;
            e0, e1, e2   : integer;
            e3, e4, e5   : integer;
            e6, e7, e8   : integer
        ) is
        begin
            assert_pixel(win_num, "p0", s_p0, e0);
            assert_pixel(win_num, "p1", s_p1, e1);
            assert_pixel(win_num, "p2", s_p2, e2);
            assert_pixel(win_num, "p3", s_p3, e3);
            assert_pixel(win_num, "p4", s_p4, e4);
            assert_pixel(win_num, "p5", s_p5, e5);
            assert_pixel(win_num, "p6", s_p6, e6);
            assert_pixel(win_num, "p7", s_p7, e7);
            assert_pixel(win_num, "p8", s_p8, e8);
            report "PASS window " & integer'image(win_num) &
                   " (stream):  [" &
                   integer'image(e0) & "," & integer'image(e1) & "," & integer'image(e2) & ";  " &
                   integer'image(e3) & "," & integer'image(e4) & "," & integer'image(e5) & ";  " &
                   integer'image(e6) & "," & integer'image(e7) & "," & integer'image(e8) & "]";
        end procedure;

    begin
        -- ---- Reset -----------------------------------------------
        rst      <= '1';
        valid_in <= '0';
        pixel_in <= (others => '0');
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst      <= '0';
        wait until rising_edge(clk);
        wait for 1 ns;

        -- ---- Stream 5×5 image (pixels 1..25, one per clock) ------
        -- Registered outputs are sampled 1 ns after each rising edge.
        -- valid_out pulses high on the same clock as the pixel that
        -- completes the bottom-right of the window.
        --
        -- Timeline (row 2 = rows_done reaches 2, row 3 = rows_done = 3, etc.):
        --   pixel 11 (col 0, row 2): col < 2  → valid_out = '0'
        --   pixel 12 (col 1, row 2): col < 2  → valid_out = '0'
        --   pixel 13 (col 2, row 2): col >= 2 → valid_out = '1', window 1
        --   pixel 14 (col 3, row 2): col >= 2 → valid_out = '1', window 2
        --   pixel 15 (col 4, row 2): col >= 2 → valid_out = '1', window 3
        --   pixel 16 (col 0, row 3): col < 2  → valid_out = '0'
        --   pixel 17 (col 1, row 3): col < 2  → valid_out = '0'
        --   pixel 18 (col 2, row 3): col >= 2 → valid_out = '1', window 4
        --   pixel 19 (col 3, row 3): col >= 2 → valid_out = '1', window 5
        --   pixel 20 (col 4, row 3): col >= 2 → valid_out = '1', window 6
        --   pixel 21 (col 0, row 4): col < 2  → valid_out = '0'
        --   pixel 22 (col 1, row 4): col < 2  → valid_out = '0'
        --   pixel 23 (col 2, row 4): col >= 2 → valid_out = '1', window 7
        --   pixel 24 (col 3, row 4): col >= 2 → valid_out = '1', window 8
        --   pixel 25 (col 4, row 4): col >= 2 → valid_out = '1', window 9

        valid_in <= '1';
        for i in 1 to 25 loop
            pixel_in <= to_signed(i, 8);
            wait until rising_edge(clk);
            wait for 1 ns;   -- let registered outputs settle

            if valid_out = '1' then
                win_count := win_count + 1;

                case win_count is
                    when 1 =>
                        -- Row-out 0, col-out 0: rows 0-2 of input, cols 0-2
                        --   1  2  3
                        --   6  7  8
                        --  11 12 13
                        check_window(1,
                             1,  2,  3,
                             6,  7,  8,
                            11, 12, 13);

                    when 2 =>
                        -- Row-out 0, col-out 1: rows 0-2 of input, cols 1-3
                        --   2  3  4
                        --   7  8  9
                        --  12 13 14
                        check_window(2,
                             2,  3,  4,
                             7,  8,  9,
                            12, 13, 14);

                    when 3 =>
                        -- Row-out 0, col-out 2: rows 0-2 of input, cols 2-4
                        --   3  4  5
                        --   8  9 10
                        --  13 14 15
                        check_window(3,
                             3,  4,  5,
                             8,  9, 10,
                            13, 14, 15);

                    when 4 =>
                        -- Row-out 1, col-out 0: rows 1-3 of input, cols 0-2
                        --   6  7  8
                        --  11 12 13
                        --  16 17 18
                        check_window(4,
                             6,  7,  8,
                            11, 12, 13,
                            16, 17, 18);

                    when 5 =>
                        -- Row-out 1, col-out 1: rows 1-3 of input, cols 1-3
                        --   7  8  9
                        --  12 13 14
                        --  17 18 19
                        check_window(5,
                             7,  8,  9,
                            12, 13, 14,
                            17, 18, 19);

                    when 6 =>
                        -- Row-out 1, col-out 2: rows 1-3 of input, cols 2-4
                        --   8  9 10
                        --  13 14 15
                        --  18 19 20
                        check_window(6,
                             8,  9, 10,
                            13, 14, 15,
                            18, 19, 20);

                    when 7 =>
                        -- Row-out 2, col-out 0: rows 2-4 of input, cols 0-2
                        --  11 12 13
                        --  16 17 18
                        --  21 22 23
                        check_window(7,
                            11, 12, 13,
                            16, 17, 18,
                            21, 22, 23);

                    when 8 =>
                        -- Row-out 2, col-out 1: rows 2-4 of input, cols 1-3
                        --  12 13 14
                        --  17 18 19
                        --  22 23 24
                        check_window(8,
                            12, 13, 14,
                            17, 18, 19,
                            22, 23, 24);

                    when 9 =>
                        -- Row-out 2, col-out 2: rows 2-4 of input, cols 2-4
                        --  13 14 15
                        --  18 19 20
                        --  23 24 25
                        check_window(9,
                            13, 14, 15,
                            18, 19, 20,
                            23, 24, 25);

                    when others =>
                        null;  -- unreachable for a 5x5 image (only 9 windows)
                end case;
            end if;
        end loop;

        valid_in <= '0';

        -- ---- Drain -----------------------------------------------
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for 1 ns;

        -- ---- Final checks ----------------------------------------
        -- A 5×5 image with valid (unpadded) 3×3 convolution produces exactly 9 windows.
        assert win_count = 9
            report "FAIL: expected exactly 9 valid windows for 5x5 image, observed " &
                   integer'image(win_count)
            severity failure;

        report "=== All window3x3_stream tests PASSED ===  (" &
               integer'image(win_count) & " / 9 valid windows checked)" severity note;

        wait;
    end process stim;

end architecture sim;
