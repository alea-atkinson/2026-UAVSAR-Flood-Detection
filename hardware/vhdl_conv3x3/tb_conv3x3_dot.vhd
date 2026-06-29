-- tb_conv3x3_dot.vhd
-- Self-checking testbench for conv3x3_dot.
--
-- Test case (hand-verified):
--   Pixels  : 1 2 3 / 4 5 6 / 7 8 9  (row-major)
--   Weights : 1 0 -1 / 1 0 -1 / 1 0 -1
--   Bias    : 0
--
-- Expected dot product:
--   1*1 + 2*0 + 3*(-1) +
--   4*1 + 5*0 + 6*(-1) +
--   7*1 + 8*0 + 9*(-1)
--   = 1 + 0 - 3 + 4 + 0 - 6 + 7 + 0 - 9
--   = (1 + 4 + 7) - (3 + 6 + 9)
--   = 12 - 18 = -6
--
-- The testbench asserts y = -6 and prints PASS/FAIL to the simulator console.
-- Simulation stops naturally after the stimulus process reaches "wait".

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity tb_conv3x3_dot is
end entity tb_conv3x3_dot;

architecture sim of tb_conv3x3_dot is

    -- DUT signals
    signal p0, p1, p2,
           p3, p4, p5,
           p6, p7, p8 : signed(7 downto 0) := (others => '0');

    signal w0, w1, w2,
           w3, w4, w5,
           w6, w7, w8 : signed(7 downto 0) := (others => '0');

    signal bias : signed(31 downto 0) := (others => '0');
    signal y    : signed(31 downto 0);

    -- Expected result (compile-time constant; easy to update for new test cases)
    constant EXPECTED : integer := -6;

begin

    -- Instantiate DUT (direct entity instantiation, VHDL-93+)
    dut : entity work.conv3x3_dot
        port map (
            p0   => p0,   p1   => p1,   p2   => p2,
            p3   => p3,   p4   => p4,   p5   => p5,
            p6   => p6,   p7   => p7,   p8   => p8,
            w0   => w0,   w1   => w1,   w2   => w2,
            w3   => w3,   w4   => w4,   w5   => w5,
            w6   => w6,   w7   => w7,   w8   => w8,
            bias => bias,
            y    => y
        );

    -- Stimulus and checking
    stim : process
    begin
        -- ----------------------------------------------------------------
        -- Test 1: Sobel-like vertical kernel on a 3x3 ramp
        --   pixels  = [1,2,3; 4,5,6; 7,8,9]
        --   weights = [1,0,-1; 1,0,-1; 1,0,-1]
        --   bias    = 0
        --   expected y = -6
        -- ----------------------------------------------------------------
        p0 <= to_signed(1, 8);   p1 <= to_signed(2, 8);   p2 <= to_signed(3, 8);
        p3 <= to_signed(4, 8);   p4 <= to_signed(5, 8);   p5 <= to_signed(6, 8);
        p6 <= to_signed(7, 8);   p7 <= to_signed(8, 8);   p8 <= to_signed(9, 8);

        w0 <= to_signed( 1, 8);  w1 <= to_signed(0, 8);   w2 <= to_signed(-1, 8);
        w3 <= to_signed( 1, 8);  w4 <= to_signed(0, 8);   w5 <= to_signed(-1, 8);
        w6 <= to_signed( 1, 8);  w7 <= to_signed(0, 8);   w8 <= to_signed(-1, 8);

        bias <= to_signed(0, 32);

        -- Allow combinational logic to settle (1 delta cycle is sufficient
        -- in simulation; 10 ns gives margin and a readable waveform).
        wait for 10 ns;

        -- Self-checking assertion
        assert to_integer(y) = EXPECTED
            report "FAIL: conv3x3_dot test 1 - expected " &
                   integer'image(EXPECTED) & " but got " &
                   integer'image(to_integer(y))
            severity failure;

        report "PASS: conv3x3_dot test 1 - y = " &
               integer'image(to_integer(y)) &
               "  (expected " & integer'image(EXPECTED) & ")";

        -- ----------------------------------------------------------------
        -- Test 2: all-ones kernel, all-ones pixels, non-zero bias
        --   pixels  = [1,1,1; 1,1,1; 1,1,1]
        --   weights = [1,1,1; 1,1,1; 1,1,1]
        --   bias    = 10
        --   expected y = 10 + 9*1 = 19
        -- ----------------------------------------------------------------
        p0 <= to_signed(1, 8);   p1 <= to_signed(1, 8);   p2 <= to_signed(1, 8);
        p3 <= to_signed(1, 8);   p4 <= to_signed(1, 8);   p5 <= to_signed(1, 8);
        p6 <= to_signed(1, 8);   p7 <= to_signed(1, 8);   p8 <= to_signed(1, 8);

        w0 <= to_signed(1, 8);   w1 <= to_signed(1, 8);   w2 <= to_signed(1, 8);
        w3 <= to_signed(1, 8);   w4 <= to_signed(1, 8);   w5 <= to_signed(1, 8);
        w6 <= to_signed(1, 8);   w7 <= to_signed(1, 8);   w8 <= to_signed(1, 8);

        bias <= to_signed(10, 32);

        wait for 10 ns;

        assert to_integer(y) = 19
            report "FAIL: conv3x3_dot test 2 - expected 19 but got " &
                   integer'image(to_integer(y))
            severity failure;

        report "PASS: conv3x3_dot test 2 - y = " &
               integer'image(to_integer(y)) & "  (expected 19)";

        -- ----------------------------------------------------------------
        -- Test 3: negative weights exercise sign extension
        --   pixels  = [10,10,10; 10,10,10; 10,10,10]
        --   weights = [-1,-1,-1; -1,-1,-1; -1,-1,-1]
        --   bias    = 100
        --   expected y = 100 + 9*(10*-1) = 100 - 90 = 10
        -- ----------------------------------------------------------------
        p0 <= to_signed(10, 8);  p1 <= to_signed(10, 8);  p2 <= to_signed(10, 8);
        p3 <= to_signed(10, 8);  p4 <= to_signed(10, 8);  p5 <= to_signed(10, 8);
        p6 <= to_signed(10, 8);  p7 <= to_signed(10, 8);  p8 <= to_signed(10, 8);

        w0 <= to_signed(-1, 8);  w1 <= to_signed(-1, 8);  w2 <= to_signed(-1, 8);
        w3 <= to_signed(-1, 8);  w4 <= to_signed(-1, 8);  w5 <= to_signed(-1, 8);
        w6 <= to_signed(-1, 8);  w7 <= to_signed(-1, 8);  w8 <= to_signed(-1, 8);

        bias <= to_signed(100, 32);

        wait for 10 ns;

        assert to_integer(y) = 10
            report "FAIL: conv3x3_dot test 3 - expected 10 but got " &
                   integer'image(to_integer(y))
            severity failure;

        report "PASS: conv3x3_dot test 3 - y = " &
               integer'image(to_integer(y)) & "  (expected 10)";

        -- All tests passed
        report "=== All conv3x3_dot tests PASSED ==="
            severity note;

        wait;  -- Stop simulation
    end process stim;

end architecture sim;
