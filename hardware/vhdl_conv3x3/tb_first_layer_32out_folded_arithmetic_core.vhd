-- tb_first_layer_32out_folded_arithmetic_core.vhd
-- Self-checking testbench for first_layer_32out_folded_arithmetic_core.
--
-- Drives 5 already-flattened 27-value test windows (one per clock, fully
-- pipelined -- 1 window/cycle throughput, matching the benchmark's
-- assumption) and checks all 32 Q.20 fixed-point folded Conv-BN-ReLU
-- outputs per window against the Python golden vectors in
-- tb_first_layer_32out_folded_arithmetic_core_vectors_pkg.vhd.
--
-- ---- Source of golden vectors ------------------------------------------
-- scripts/generate_first_layer_32out_folded_arithmetic_core_vectors.py
-- Window 0 (real_window_0) is cross-checked by that script against the
-- EXISTING, already GHDL-verified all-32-kernel real-tile Q.20 report's
-- expected output at position (row=0, col=0) -- see that script's header.
--
-- ---- Latency (6 cycles: 4 raw-conv + 2 BN-fold/ReLU) --------------------
-- Windows presented at cycles 1-5 (one per clock); first output (window 0)
-- appears at cycle 7; outputs continue at cycles 8, 9, 10, 11 for windows
-- 1-4. Checked during an 8-cycle drain loop after the last window.
--
-- ---- Scope ----------------------------------------------------------------
-- Simulation-only functional check of the arithmetic-core prototype. NOT
-- image streaming, NOT full U-Net, NOT board-tested, NOT a measured
-- hardware speedup or power claim.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.tb_first_layer_32out_folded_arithmetic_core_vectors_pkg.all;

entity tb_first_layer_32out_folded_arithmetic_core is
end entity tb_first_layer_32out_folded_arithmetic_core;

architecture sim of tb_first_layer_32out_folded_arithmetic_core is

    constant CLK_PERIOD : time := 10 ns;

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal c0_p0, c0_p1, c0_p2, c0_p3, c0_p4, c0_p5, c0_p6, c0_p7, c0_p8 : signed(7 downto 0) := (others => '0');
    signal c1_p0, c1_p1, c1_p2, c1_p3, c1_p4, c1_p5, c1_p6, c1_p7, c1_p8 : signed(7 downto 0) := (others => '0');
    signal c2_p0, c2_p1, c2_p2, c2_p3, c2_p4, c2_p5, c2_p6, c2_p7, c2_p8 : signed(7 downto 0) := (others => '0');

    signal valid_out : std_logic;
    signal s_y0, s_y1, s_y2, s_y3, s_y4, s_y5, s_y6, s_y7, s_y8, s_y9,
           s_y10, s_y11, s_y12, s_y13, s_y14, s_y15, s_y16, s_y17, s_y18, s_y19,
           s_y20, s_y21, s_y22, s_y23, s_y24, s_y25, s_y26, s_y27, s_y28, s_y29,
           s_y30, s_y31 : signed(47 downto 0);

begin

    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.first_layer_32out_folded_arithmetic_core
        port map (
            clk => clk, rst => rst, valid_in => valid_in,
            c0_p0 => c0_p0, c0_p1 => c0_p1, c0_p2 => c0_p2,
            c0_p3 => c0_p3, c0_p4 => c0_p4, c0_p5 => c0_p5,
            c0_p6 => c0_p6, c0_p7 => c0_p7, c0_p8 => c0_p8,
            c1_p0 => c1_p0, c1_p1 => c1_p1, c1_p2 => c1_p2,
            c1_p3 => c1_p3, c1_p4 => c1_p4, c1_p5 => c1_p5,
            c1_p6 => c1_p6, c1_p7 => c1_p7, c1_p8 => c1_p8,
            c2_p0 => c2_p0, c2_p1 => c2_p1, c2_p2 => c2_p2,
            c2_p3 => c2_p3, c2_p4 => c2_p4, c2_p5 => c2_p5,
            c2_p6 => c2_p6, c2_p7 => c2_p7, c2_p8 => c2_p8,
            valid_out => valid_out,
            y0 => s_y0, y1 => s_y1, y2 => s_y2, y3 => s_y3, y4 => s_y4,
            y5 => s_y5, y6 => s_y6, y7 => s_y7, y8 => s_y8, y9 => s_y9,
            y10 => s_y10, y11 => s_y11, y12 => s_y12, y13 => s_y13, y14 => s_y14,
            y15 => s_y15, y16 => s_y16, y17 => s_y17, y18 => s_y18, y19 => s_y19,
            y20 => s_y20, y21 => s_y21, y22 => s_y22, y23 => s_y23, y24 => s_y24,
            y25 => s_y25, y26 => s_y26, y27 => s_y27, y28 => s_y28, y29 => s_y29,
            y30 => s_y30, y31 => s_y31
        );

    stim : process

        variable out_count : integer := 0;

        procedure drive_window(w : window27_t) is
        begin
            c0_p0 <= to_signed(w(0), 8);  c0_p1 <= to_signed(w(1), 8);  c0_p2 <= to_signed(w(2), 8);
            c0_p3 <= to_signed(w(3), 8);  c0_p4 <= to_signed(w(4), 8);  c0_p5 <= to_signed(w(5), 8);
            c0_p6 <= to_signed(w(6), 8);  c0_p7 <= to_signed(w(7), 8);  c0_p8 <= to_signed(w(8), 8);
            c1_p0 <= to_signed(w(9), 8);  c1_p1 <= to_signed(w(10), 8); c1_p2 <= to_signed(w(11), 8);
            c1_p3 <= to_signed(w(12), 8); c1_p4 <= to_signed(w(13), 8); c1_p5 <= to_signed(w(14), 8);
            c1_p6 <= to_signed(w(15), 8); c1_p7 <= to_signed(w(16), 8); c1_p8 <= to_signed(w(17), 8);
            c2_p0 <= to_signed(w(18), 8); c2_p1 <= to_signed(w(19), 8); c2_p2 <= to_signed(w(20), 8);
            c2_p3 <= to_signed(w(21), 8); c2_p4 <= to_signed(w(22), 8); c2_p5 <= to_signed(w(23), 8);
            c2_p6 <= to_signed(w(24), 8); c2_p7 <= to_signed(w(25), 8); c2_p8 <= to_signed(w(26), 8);
        end procedure drive_window;

        procedure check_one(kernel_id : integer; got : signed(47 downto 0);
                             expected : integer; window_idx : integer) is
        begin
            assert to_integer(got) = expected
                report "FAIL window " & integer'image(window_idx) &
                       " kernel " & integer'image(kernel_id) &
                       ": y = " & integer'image(to_integer(got)) &
                       "  expected " & integer'image(expected)
                severity failure;
        end procedure check_one;

        procedure check_output(window_idx : integer) is
            variable exp : y32_t;
        begin
            exp := EXPECTED_Y(window_idx);
            check_one(0, s_y0, exp(0), window_idx);   check_one(1, s_y1, exp(1), window_idx);
            check_one(2, s_y2, exp(2), window_idx);   check_one(3, s_y3, exp(3), window_idx);
            check_one(4, s_y4, exp(4), window_idx);   check_one(5, s_y5, exp(5), window_idx);
            check_one(6, s_y6, exp(6), window_idx);   check_one(7, s_y7, exp(7), window_idx);
            check_one(8, s_y8, exp(8), window_idx);   check_one(9, s_y9, exp(9), window_idx);
            check_one(10, s_y10, exp(10), window_idx); check_one(11, s_y11, exp(11), window_idx);
            check_one(12, s_y12, exp(12), window_idx); check_one(13, s_y13, exp(13), window_idx);
            check_one(14, s_y14, exp(14), window_idx); check_one(15, s_y15, exp(15), window_idx);
            check_one(16, s_y16, exp(16), window_idx); check_one(17, s_y17, exp(17), window_idx);
            check_one(18, s_y18, exp(18), window_idx); check_one(19, s_y19, exp(19), window_idx);
            check_one(20, s_y20, exp(20), window_idx); check_one(21, s_y21, exp(21), window_idx);
            check_one(22, s_y22, exp(22), window_idx); check_one(23, s_y23, exp(23), window_idx);
            check_one(24, s_y24, exp(24), window_idx); check_one(25, s_y25, exp(25), window_idx);
            check_one(26, s_y26, exp(26), window_idx); check_one(27, s_y27, exp(27), window_idx);
            check_one(28, s_y28, exp(28), window_idx); check_one(29, s_y29, exp(29), window_idx);
            check_one(30, s_y30, exp(30), window_idx); check_one(31, s_y31, exp(31), window_idx);

            report "PASS window " & integer'image(window_idx) &
                   " y0=" & integer'image(to_integer(s_y0)) &
                   " y1=" & integer'image(to_integer(s_y1)) &
                   " y31=" & integer'image(to_integer(s_y31));
        end procedure check_output;

    begin
        -- ---- Reset (3 clocks) ----------------------------------------
        rst      <= '1';
        valid_in <= '0';
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst <= '0';
        wait until rising_edge(clk);
        wait for 1 ns;

        -- ---- Present all NUM_TEST_WINDOWS windows, one per clock ------
        valid_in <= '1';
        for i in 0 to NUM_TEST_WINDOWS - 1 loop
            drive_window(TEST_WINDOWS(i));
            wait until rising_edge(clk);
            wait for 1 ns;

            if valid_out = '1' then
                check_output(out_count);
                out_count := out_count + 1;
            end if;
        end loop;

        -- ---- Drain: enough clocks to flush the 6-cycle pipeline -------
        valid_in <= '0';
        for d in 1 to 8 loop
            wait until rising_edge(clk);
            wait for 1 ns;

            if valid_out = '1' then
                check_output(out_count);
                out_count := out_count + 1;
            end if;
        end loop;

        -- ---- Final checks --------------------------------------------
        assert out_count = NUM_TEST_WINDOWS
            report "FAIL: expected exactly " & integer'image(NUM_TEST_WINDOWS) &
                   " valid output windows, observed " & integer'image(out_count)
            severity failure;

        report "=== All first_layer_32out_folded_arithmetic_core tests PASSED ===" &
               "  (" & integer'image(NUM_TEST_WINDOWS) & " windows x 32 kernels = " &
               integer'image(NUM_TEST_WINDOWS * 32) & " / " &
               integer'image(NUM_TEST_WINDOWS * 32) & " outputs match Python golden vectors)"
               severity note;
        report "    NOTE: Single-shot (not streamed) arithmetic-core PROTOTYPE only." &
               " Not full U-Net inference, not board-tested, no measured speedup." severity note;

        wait;
    end process stim;

end architecture sim;
