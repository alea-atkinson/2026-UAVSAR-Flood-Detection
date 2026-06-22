-- tb_conv3x3_dot_pipelined.vhd
-- Self-checking testbench for conv3x3_dot_pipelined.
--
-- Pipeline latency: 3 clock cycles (valid_in to valid_out).
--
-- Strategy:
--   For each test, assert valid_in for exactly ONE clock cycle.
--   Then loop on rising edges until valid_out goes high (3 edges later).
--   Read y at that point and assert against the expected value.
--   Let the pipeline drain (valid_out drops back to '0') before next test.
--
-- Same three test vectors as tb_conv3x3_dot.vhd for direct comparison:
--
--   Test 1: pixels 1..9, weights [1,0,-1; 1,0,-1; 1,0,-1], bias 0
--           Expected: 1+0-3 + 4+0-6 + 7+0-9 = -6
--
--   Test 2: pixels all 1, weights all 1, bias 10
--           Expected: 10 + 9*1 = 19
--
--   Test 3: pixels all 10, weights all -1, bias 100
--           Expected: 100 + 9*(10*-1) = 100-90 = 10

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity tb_conv3x3_dot_pipelined is
end entity tb_conv3x3_dot_pipelined;

architecture sim of tb_conv3x3_dot_pipelined is

    constant CLK_PERIOD      : time    := 10 ns;   -- 100 MHz
    constant PIPELINE_STAGES : natural := 3;        -- for documentation

    -- Clock and control
    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    -- DUT data inputs
    signal p0, p1, p2,
           p3, p4, p5,
           p6, p7, p8 : signed(7 downto 0) := (others => '0');

    signal w0, w1, w2,
           w3, w4, w5,
           w6, w7, w8 : signed(7 downto 0) := (others => '0');

    signal bias : signed(31 downto 0) := (others => '0');

    -- DUT outputs
    signal valid_out : std_logic;
    signal y         : signed(31 downto 0);

begin

    -- ----------------------------------------------------------------
    -- Free-running clock
    -- ----------------------------------------------------------------
    clk <= not clk after CLK_PERIOD / 2;

    -- ----------------------------------------------------------------
    -- DUT
    -- ----------------------------------------------------------------
    dut : entity work.conv3x3_dot_pipelined
        port map (
            clk      => clk,      rst      => rst,      valid_in => valid_in,
            p0 => p0, p1 => p1,   p2 => p2,
            p3 => p3, p4 => p4,   p5 => p5,
            p6 => p6, p7 => p7,   p8 => p8,
            w0 => w0, w1 => w1,   w2 => w2,
            w3 => w3, w4 => w4,   w5 => w5,
            w6 => w6, w7 => w7,   w8 => w8,
            bias      => bias,
            valid_out => valid_out,
            y         => y
        );

    -- ----------------------------------------------------------------
    -- Stimulus and self-checking
    -- ----------------------------------------------------------------
    stim : process

        -- Wait on successive rising edges until valid_out is seen.
        -- Works for any pipeline depth; exits on the 3rd edge for this design.
        procedure wait_for_result is
        begin
            loop
                wait until rising_edge(clk);
                exit when valid_out = '1';
            end loop;
        end procedure;

    begin
        -- ---- Reset sequence ----------------------------------------
        rst      <= '1';
        valid_in <= '0';
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst <= '0';
        wait until rising_edge(clk);   -- one idle cycle after reset

        -- ==============================================================
        -- Test 1: Sobel-like vertical kernel on a 3×3 ramp
        --   pixels  : [1,2,3; 4,5,6; 7,8,9]
        --   weights : [1,0,-1; 1,0,-1; 1,0,-1]
        --   bias    : 0
        --   expected: 1*1+2*0+3*(-1) + 4*1+5*0+6*(-1) + 7*1+8*0+9*(-1)
        --           = 1+0-3 + 4+0-6 + 7+0-9 = -6
        -- ==============================================================
        p0 <= to_signed(1, 8);  p1 <= to_signed(2, 8);  p2 <= to_signed(3, 8);
        p3 <= to_signed(4, 8);  p4 <= to_signed(5, 8);  p5 <= to_signed(6, 8);
        p6 <= to_signed(7, 8);  p7 <= to_signed(8, 8);  p8 <= to_signed(9, 8);

        w0 <= to_signed( 1, 8);  w1 <= to_signed(0, 8);  w2 <= to_signed(-1, 8);
        w3 <= to_signed( 1, 8);  w4 <= to_signed(0, 8);  w5 <= to_signed(-1, 8);
        w6 <= to_signed( 1, 8);  w7 <= to_signed(0, 8);  w8 <= to_signed(-1, 8);

        bias <= to_signed(0, 32);

        valid_in <= '1';
        wait until rising_edge(clk);   -- clock edge N: stage 1 latches inputs
        valid_in <= '0';               -- deassert; valid propagates through pipeline

        wait_for_result;               -- resumes at clock edge N+3

        assert to_integer(y) = -6
            report "FAIL test 1: expected -6, got " & integer'image(to_integer(y))
            severity failure;
        report "PASS test 1 (pipelined): y = " & integer'image(to_integer(y)) &
               "  (expected -6,  latency = " & integer'image(PIPELINE_STAGES) & " cycles)";

        -- Drain: let valid_out return to '0' before next test
        wait until rising_edge(clk);
        wait until rising_edge(clk);

        -- ==============================================================
        -- Test 2: all-ones pixels and weights with non-zero bias
        --   pixels  : all 1
        --   weights : all 1
        --   bias    : 10
        --   expected: 10 + 9*1 = 19
        -- ==============================================================
        p0 <= to_signed(1, 8);  p1 <= to_signed(1, 8);  p2 <= to_signed(1, 8);
        p3 <= to_signed(1, 8);  p4 <= to_signed(1, 8);  p5 <= to_signed(1, 8);
        p6 <= to_signed(1, 8);  p7 <= to_signed(1, 8);  p8 <= to_signed(1, 8);

        w0 <= to_signed(1, 8);  w1 <= to_signed(1, 8);  w2 <= to_signed(1, 8);
        w3 <= to_signed(1, 8);  w4 <= to_signed(1, 8);  w5 <= to_signed(1, 8);
        w6 <= to_signed(1, 8);  w7 <= to_signed(1, 8);  w8 <= to_signed(1, 8);

        bias <= to_signed(10, 32);

        valid_in <= '1';
        wait until rising_edge(clk);
        valid_in <= '0';

        wait_for_result;

        assert to_integer(y) = 19
            report "FAIL test 2: expected 19, got " & integer'image(to_integer(y))
            severity failure;
        report "PASS test 2 (pipelined): y = " & integer'image(to_integer(y)) &
               "  (expected 19)";

        wait until rising_edge(clk);
        wait until rising_edge(clk);

        -- ==============================================================
        -- Test 3: negative weights — exercises sign extension path
        --   pixels  : all 10
        --   weights : all -1
        --   bias    : 100
        --   expected: 100 + 9*(10*-1) = 100 - 90 = 10
        -- ==============================================================
        p0 <= to_signed(10, 8); p1 <= to_signed(10, 8); p2 <= to_signed(10, 8);
        p3 <= to_signed(10, 8); p4 <= to_signed(10, 8); p5 <= to_signed(10, 8);
        p6 <= to_signed(10, 8); p7 <= to_signed(10, 8); p8 <= to_signed(10, 8);

        w0 <= to_signed(-1, 8); w1 <= to_signed(-1, 8); w2 <= to_signed(-1, 8);
        w3 <= to_signed(-1, 8); w4 <= to_signed(-1, 8); w5 <= to_signed(-1, 8);
        w6 <= to_signed(-1, 8); w7 <= to_signed(-1, 8); w8 <= to_signed(-1, 8);

        bias <= to_signed(100, 32);

        valid_in <= '1';
        wait until rising_edge(clk);
        valid_in <= '0';

        wait_for_result;

        assert to_integer(y) = 10
            report "FAIL test 3: expected 10, got " & integer'image(to_integer(y))
            severity failure;
        report "PASS test 3 (pipelined): y = " & integer'image(to_integer(y)) &
               "  (expected 10)";

        -- ---- Done ---------------------------------------------------
        report "=== All conv3x3_dot_pipelined tests PASSED ===" severity note;
        wait;   -- stops simulation

    end process stim;

end architecture sim;
