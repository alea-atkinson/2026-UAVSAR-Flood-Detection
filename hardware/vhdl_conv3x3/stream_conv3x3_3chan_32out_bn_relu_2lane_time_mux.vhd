-- stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd
-- Streaming wrapper around TWO PARALLEL resource-shared 16-output
-- one-window schedulers (conv3x3_3chan_16out_bn_relu_time_mux), forming
-- the first multi-lane implementation described in
-- hardware/vhdl_conv3x3/multi_lane_resource_shared_conv_bn_relu_design_memo.md.
--
-- This is NOT full U-Net FPGA inference and NOT a board demo. It IS the
-- 2-LANE resource-shared architecture recommended as the next
-- implementation by the design memo (Section 9/12): lane 0 handles
-- kernels 0-15, lane 1 handles kernels 16-31, each lane has its own
-- time-multiplexed dot-product engine and its own shared Q.16 BN+ReLU
-- unit (via two separate conv3x3_3chan_16out_bn_relu_time_mux
-- instances), and BOTH LANES PROCESS THE SAME WINDOW IN PARALLEL rather
-- than sequentially -- unlike the single-lane 32-output design
-- (stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd, preserved
-- UNMODIFIED), which processes one shared 32-kernel engine sequentially
-- per window.
--
-- ---- Two-phase, non-overlapped design (same pattern as the 1-lane
-- design, unmodified in structure) ------------------------------------------
-- PHASE 1 (CAPTURE): identical to the 1-lane design -- three unmodified
-- window3x3_stream instances (one per input channel) produce valid 3x3
-- windows, captured into an internal buffer (indexed 0 .. NUM_WINDOWS-1).
--
-- PHASE 2 (PROCESS): once all NUM_WINDOWS windows are captured, this
-- module feeds them into BOTH lane schedulers ONE WINDOW AT A TIME, but
-- WITHIN a window, both lanes run CONCURRENTLY: `start` is pulsed to
-- both lanes on the same cycle, and this module's barrier logic waits
-- until BOTH lanes have signaled `done` for the current window (latching
-- each lane's outputs independently as they arrive, so the two lanes'
-- done pulses do not need to land on the exact same cycle) before
-- combining their 16+16 outputs into ONE 32-value window-major/
-- kernel-minor valid_out pulse and advancing to the next window.
--
-- ---- Output ordering (UNCHANGED from the 1-lane design) -------------------
-- WINDOW-MAJOR, KERNEL-MINOR, y0..y31 per window, IDENTICAL to
-- stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd's output order: lane
-- 0's local y0..y15 become this module's global y0..y15, lane 1's local
-- y0..y15 become this module's global y16..y31. This is a DELIBERATE
-- design choice (memo Section 4/7) so the EXISTING Python Q.16 golden
-- vectors (first_layer_32out_bn_relu_resource_shared_pkg's
-- EXPECTED_RELU_FX, unmodified) remain valid without regeneration.
--
-- ---- Limitation: buffered, not truly streaming end-to-end -----------------
-- Same as the 1-lane design: phase 2 is far slower per window than phase
-- 1, so all windows must be captured before processing begins.
--
-- ---- Weight/constant source / scope ---------------------------------------
-- Both lanes source kernels 0-31's folded INT8 weights and Q.16
-- SCALE_FX/BIAS_FX constants from the EXISTING, UNMODIFIED
-- first_layer_32out_bn_relu_resource_shared_pkg.vhd (via
-- conv3x3_3chan_16out_bn_relu_time_mux's KERNEL_OFFSET generic), not a
-- new package. This covers the COMPLETE first Conv2d layer's 32 output
-- channels. This is still NOT full U-Net FPGA inference, NOT a board
-- demo, and NOT a measured speedup claim -- see the design memo's
-- Limitations section (11) for the full list of scope boundaries.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux is
    generic (
        IMG_WIDTH : positive := 5       -- square image; must be >= 3
    );
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;        -- synchronous, active high
        valid_in : in  std_logic;

        -- Three input-channel pixel streams (row-major, one triplet per clock)
        pixel_c0 : in signed(7 downto 0);
        pixel_c1 : in signed(7 downto 0);
        pixel_c2 : in signed(7 downto 0);

        -- One-cycle pulse per completed window, in window order (row-major).
        valid_out : out std_logic;
        y0 : out signed(47 downto 0);
        y1 : out signed(47 downto 0);
        y2 : out signed(47 downto 0);
        y3 : out signed(47 downto 0);
        y4 : out signed(47 downto 0);
        y5 : out signed(47 downto 0);
        y6 : out signed(47 downto 0);
        y7 : out signed(47 downto 0);
        y8 : out signed(47 downto 0);
        y9 : out signed(47 downto 0);
        y10 : out signed(47 downto 0);
        y11 : out signed(47 downto 0);
        y12 : out signed(47 downto 0);
        y13 : out signed(47 downto 0);
        y14 : out signed(47 downto 0);
        y15 : out signed(47 downto 0);
        y16 : out signed(47 downto 0);
        y17 : out signed(47 downto 0);
        y18 : out signed(47 downto 0);
        y19 : out signed(47 downto 0);
        y20 : out signed(47 downto 0);
        y21 : out signed(47 downto 0);
        y22 : out signed(47 downto 0);
        y23 : out signed(47 downto 0);
        y24 : out signed(47 downto 0);
        y25 : out signed(47 downto 0);
        y26 : out signed(47 downto 0);
        y27 : out signed(47 downto 0);
        y28 : out signed(47 downto 0);
        y29 : out signed(47 downto 0);
        y30 : out signed(47 downto 0);
        y31 : out signed(47 downto 0);

        -- One-cycle pulse after the LAST window's outputs are valid.
        all_done : out std_logic
    );
end entity stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux;

architecture rtl of stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux is

    constant NUM_WINDOWS : positive := (IMG_WIDTH - 2) * (IMG_WIDTH - 2);

    -- ----------------------------------------------------------------
    -- Window generators -- ONE PER INPUT CHANNEL, unmodified reuse of
    -- window3x3_stream.vhd. Identical to the 1-lane design.
    -- ----------------------------------------------------------------
    signal win_valid_c0, win_valid_c1, win_valid_c2 : std_logic;

    signal c0_p0, c0_p1, c0_p2, c0_p3, c0_p4, c0_p5,
           c0_p6, c0_p7, c0_p8 : signed(7 downto 0);

    signal c1_p0, c1_p1, c1_p2, c1_p3, c1_p4, c1_p5,
           c1_p6, c1_p7, c1_p8 : signed(7 downto 0);

    signal c2_p0, c2_p1, c2_p2, c2_p3, c2_p4, c2_p5,
           c2_p6, c2_p7, c2_p8 : signed(7 downto 0);

    -- ----------------------------------------------------------------
    -- Phase 1 capture buffer: NUM_WINDOWS slots, 9 INT8 pixels per
    -- channel per slot. Identical to the 1-lane design.
    -- ----------------------------------------------------------------
    type win9_t     is array (0 to 8) of signed(7 downto 0);
    type win_buf_t  is array (0 to NUM_WINDOWS - 1) of win9_t;

    signal buf_c0, buf_c1, buf_c2 : win_buf_t;

    signal wr_ptr : integer range 0 to NUM_WINDOWS := 0;

    -- ----------------------------------------------------------------
    -- Phase 2 processing: TWO shared conv3x3_3chan_16out_bn_relu_time_mux
    -- instances (lane 0: kernels 0-15, lane 1: kernels 16-31), both fed
    -- the SAME buffered window in parallel.
    -- ----------------------------------------------------------------
    signal proc_idx : integer range 0 to NUM_WINDOWS - 1 := 0;

    signal sched_c0_p0, sched_c0_p1, sched_c0_p2, sched_c0_p3, sched_c0_p4,
           sched_c0_p5, sched_c0_p6, sched_c0_p7, sched_c0_p8 : signed(7 downto 0);
    signal sched_c1_p0, sched_c1_p1, sched_c1_p2, sched_c1_p3, sched_c1_p4,
           sched_c1_p5, sched_c1_p6, sched_c1_p7, sched_c1_p8 : signed(7 downto 0);
    signal sched_c2_p0, sched_c2_p1, sched_c2_p2, sched_c2_p3, sched_c2_p4,
           sched_c2_p5, sched_c2_p6, sched_c2_p7, sched_c2_p8 : signed(7 downto 0);

    signal lane0_start, lane1_start : std_logic := '0';
    signal lane0_busy, lane1_busy   : std_logic;
    signal lane0_done, lane1_done   : std_logic;

    signal lane0_y0, lane0_y1, lane0_y2, lane0_y3, lane0_y4, lane0_y5, lane0_y6, lane0_y7,
           lane0_y8, lane0_y9, lane0_y10, lane0_y11, lane0_y12, lane0_y13, lane0_y14, lane0_y15 : signed(47 downto 0);
    signal lane1_y0, lane1_y1, lane1_y2, lane1_y3, lane1_y4, lane1_y5, lane1_y6, lane1_y7,
           lane1_y8, lane1_y9, lane1_y10, lane1_y11, lane1_y12, lane1_y13, lane1_y14, lane1_y15 : signed(47 downto 0);

    -- Sticky per-window "this lane has reported done" latches -- allows
    -- the barrier below to wait correctly even if the two lanes'
    -- `done` pulses do not land on the exact same clock cycle.
    signal lane0_done_r, lane1_done_r : std_logic := '0';

    type y_arr_t is array (0 to 31) of signed(47 downto 0);
    signal y_out_regs : y_arr_t := (others => (others => '0'));

    signal valid_out_r : std_logic := '0';
    signal all_done_r  : std_logic := '0';

    type state_t is (S_CAPTURE, S_RUN, S_FINISHED);
    signal state : state_t := S_CAPTURE;

begin

    -- ================================================================
    -- PHASE 1: window generators (one per input channel), unmodified
    -- reuse of window3x3_stream.vhd. Identical to the 1-lane design.
    -- ================================================================
    win_gen_c0 : entity work.window3x3_stream
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk => clk,  rst => rst,  valid_in => valid_in,
            pixel_in  => pixel_c0,
            valid_out => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8
        );

    win_gen_c1 : entity work.window3x3_stream
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk => clk,  rst => rst,  valid_in => valid_in,
            pixel_in  => pixel_c1,
            valid_out => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8
        );

    win_gen_c2 : entity work.window3x3_stream
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk => clk,  rst => rst,  valid_in => valid_in,
            pixel_in  => pixel_c2,
            valid_out => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8
        );

    -- ================================================================
    -- PHASE 1: capture each valid window into the buffer. Identical to
    -- the 1-lane design.
    -- ================================================================
    capture : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                wr_ptr <= 0;
            elsif win_valid_c0 = '1' and wr_ptr < NUM_WINDOWS then
                buf_c0(wr_ptr) <= (c0_p0, c0_p1, c0_p2, c0_p3, c0_p4, c0_p5, c0_p6, c0_p7, c0_p8);
                buf_c1(wr_ptr) <= (c1_p0, c1_p1, c1_p2, c1_p3, c1_p4, c1_p5, c1_p6, c1_p7, c1_p8);
                buf_c2(wr_ptr) <= (c2_p0, c2_p1, c2_p2, c2_p3, c2_p4, c2_p5, c2_p6, c2_p7, c2_p8);
                wr_ptr <= wr_ptr + 1;
            end if;
        end if;
    end process capture;

    -- ================================================================
    -- Combinational mux: present the window currently being processed
    -- (proc_idx) to BOTH lane schedulers' pixel ports (same window,
    -- shared/fanned-out, not duplicated storage).
    -- ================================================================
    sched_c0_p0 <= buf_c0(proc_idx)(0); sched_c0_p1 <= buf_c0(proc_idx)(1); sched_c0_p2 <= buf_c0(proc_idx)(2);
    sched_c0_p3 <= buf_c0(proc_idx)(3); sched_c0_p4 <= buf_c0(proc_idx)(4); sched_c0_p5 <= buf_c0(proc_idx)(5);
    sched_c0_p6 <= buf_c0(proc_idx)(6); sched_c0_p7 <= buf_c0(proc_idx)(7); sched_c0_p8 <= buf_c0(proc_idx)(8);

    sched_c1_p0 <= buf_c1(proc_idx)(0); sched_c1_p1 <= buf_c1(proc_idx)(1); sched_c1_p2 <= buf_c1(proc_idx)(2);
    sched_c1_p3 <= buf_c1(proc_idx)(3); sched_c1_p4 <= buf_c1(proc_idx)(4); sched_c1_p5 <= buf_c1(proc_idx)(5);
    sched_c1_p6 <= buf_c1(proc_idx)(6); sched_c1_p7 <= buf_c1(proc_idx)(7); sched_c1_p8 <= buf_c1(proc_idx)(8);

    sched_c2_p0 <= buf_c2(proc_idx)(0); sched_c2_p1 <= buf_c2(proc_idx)(1); sched_c2_p2 <= buf_c2(proc_idx)(2);
    sched_c2_p3 <= buf_c2(proc_idx)(3); sched_c2_p4 <= buf_c2(proc_idx)(4); sched_c2_p5 <= buf_c2(proc_idx)(5);
    sched_c2_p6 <= buf_c2(proc_idx)(6); sched_c2_p7 <= buf_c2(proc_idx)(7); sched_c2_p8 <= buf_c2(proc_idx)(8);

    -- ================================================================
    -- Lane 0: kernels 0-15. Own dot-product engine + own shared BN+ReLU
    -- unit (inside conv3x3_3chan_16out_bn_relu_time_mux).
    -- ================================================================
    lane0_inst : entity work.conv3x3_3chan_16out_bn_relu_time_mux
        generic map (KERNEL_OFFSET => 0)
        port map (
            clk   => clk,
            rst   => rst,
            start => lane0_start,
            c0_p0 => sched_c0_p0, c0_p1 => sched_c0_p1, c0_p2 => sched_c0_p2,
            c0_p3 => sched_c0_p3, c0_p4 => sched_c0_p4, c0_p5 => sched_c0_p5,
            c0_p6 => sched_c0_p6, c0_p7 => sched_c0_p7, c0_p8 => sched_c0_p8,
            c1_p0 => sched_c1_p0, c1_p1 => sched_c1_p1, c1_p2 => sched_c1_p2,
            c1_p3 => sched_c1_p3, c1_p4 => sched_c1_p4, c1_p5 => sched_c1_p5,
            c1_p6 => sched_c1_p6, c1_p7 => sched_c1_p7, c1_p8 => sched_c1_p8,
            c2_p0 => sched_c2_p0, c2_p1 => sched_c2_p1, c2_p2 => sched_c2_p2,
            c2_p3 => sched_c2_p3, c2_p4 => sched_c2_p4, c2_p5 => sched_c2_p5,
            c2_p6 => sched_c2_p6, c2_p7 => sched_c2_p7, c2_p8 => sched_c2_p8,
            busy  => lane0_busy,
            done  => lane0_done,
            y0 => lane0_y0, y1 => lane0_y1, y2 => lane0_y2, y3 => lane0_y3,
            y4 => lane0_y4, y5 => lane0_y5, y6 => lane0_y6, y7 => lane0_y7,
            y8 => lane0_y8, y9 => lane0_y9, y10 => lane0_y10, y11 => lane0_y11,
            y12 => lane0_y12, y13 => lane0_y13, y14 => lane0_y14, y15 => lane0_y15
        );

    -- ================================================================
    -- Lane 1: kernels 16-31. Own dot-product engine + own shared BN+ReLU
    -- unit (inside conv3x3_3chan_16out_bn_relu_time_mux), running IN
    -- PARALLEL with lane 0 on the SAME window.
    -- ================================================================
    lane1_inst : entity work.conv3x3_3chan_16out_bn_relu_time_mux
        generic map (KERNEL_OFFSET => 16)
        port map (
            clk   => clk,
            rst   => rst,
            start => lane1_start,
            c0_p0 => sched_c0_p0, c0_p1 => sched_c0_p1, c0_p2 => sched_c0_p2,
            c0_p3 => sched_c0_p3, c0_p4 => sched_c0_p4, c0_p5 => sched_c0_p5,
            c0_p6 => sched_c0_p6, c0_p7 => sched_c0_p7, c0_p8 => sched_c0_p8,
            c1_p0 => sched_c1_p0, c1_p1 => sched_c1_p1, c1_p2 => sched_c1_p2,
            c1_p3 => sched_c1_p3, c1_p4 => sched_c1_p4, c1_p5 => sched_c1_p5,
            c1_p6 => sched_c1_p6, c1_p7 => sched_c1_p7, c1_p8 => sched_c1_p8,
            c2_p0 => sched_c2_p0, c2_p1 => sched_c2_p1, c2_p2 => sched_c2_p2,
            c2_p3 => sched_c2_p3, c2_p4 => sched_c2_p4, c2_p5 => sched_c2_p5,
            c2_p6 => sched_c2_p6, c2_p7 => sched_c2_p7, c2_p8 => sched_c2_p8,
            busy  => lane1_busy,
            done  => lane1_done,
            y0 => lane1_y0, y1 => lane1_y1, y2 => lane1_y2, y3 => lane1_y3,
            y4 => lane1_y4, y5 => lane1_y5, y6 => lane1_y6, y7 => lane1_y7,
            y8 => lane1_y8, y9 => lane1_y9, y10 => lane1_y10, y11 => lane1_y11,
            y12 => lane1_y12, y13 => lane1_y13, y14 => lane1_y14, y15 => lane1_y15
        );

    -- ================================================================
    -- PHASE 2 FSM: wait for capture to complete (wr_ptr = NUM_WINDOWS),
    -- then sequentially process each buffered window, dispatching BOTH
    -- lanes together and waiting for a BARRIER (both lanes done) before
    -- combining outputs into ONE window-major/kernel-minor valid_out
    -- pulse and advancing to the next window.
    -- ================================================================
    process_fsm : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state        <= S_CAPTURE;
                proc_idx     <= 0;
                lane0_start  <= '0';
                lane1_start  <= '0';
                lane0_done_r <= '0';
                lane1_done_r <= '0';
                valid_out_r  <= '0';
                all_done_r   <= '0';
                y_out_regs   <= (others => (others => '0'));

            else
                valid_out_r <= '0';
                all_done_r  <= '0';

                case state is
                    when S_CAPTURE =>
                        lane0_start <= '0';
                        lane1_start <= '0';
                        if wr_ptr = NUM_WINDOWS then
                            proc_idx     <= 0;
                            lane0_start  <= '1';
                            lane1_start  <= '1';
                            lane0_done_r <= '0';
                            lane1_done_r <= '0';
                            state        <= S_RUN;
                        end if;

                    when S_RUN =>
                        if lane0_start = '1' then
                            lane0_start <= '0';
                        end if;
                        if lane1_start = '1' then
                            lane1_start <= '0';
                        end if;

                        -- Latch each lane's outputs independently as soon as
                        -- that lane reports done (does not require both
                        -- lanes to finish on the same clock cycle).
                        if lane0_done = '1' then
                            y_out_regs(0)  <= lane0_y0;
                            y_out_regs(1)  <= lane0_y1;
                            y_out_regs(2)  <= lane0_y2;
                            y_out_regs(3)  <= lane0_y3;
                            y_out_regs(4)  <= lane0_y4;
                            y_out_regs(5)  <= lane0_y5;
                            y_out_regs(6)  <= lane0_y6;
                            y_out_regs(7)  <= lane0_y7;
                            y_out_regs(8)  <= lane0_y8;
                            y_out_regs(9)  <= lane0_y9;
                            y_out_regs(10) <= lane0_y10;
                            y_out_regs(11) <= lane0_y11;
                            y_out_regs(12) <= lane0_y12;
                            y_out_regs(13) <= lane0_y13;
                            y_out_regs(14) <= lane0_y14;
                            y_out_regs(15) <= lane0_y15;
                            lane0_done_r   <= '1';
                        end if;

                        if lane1_done = '1' then
                            y_out_regs(16) <= lane1_y0;
                            y_out_regs(17) <= lane1_y1;
                            y_out_regs(18) <= lane1_y2;
                            y_out_regs(19) <= lane1_y3;
                            y_out_regs(20) <= lane1_y4;
                            y_out_regs(21) <= lane1_y5;
                            y_out_regs(22) <= lane1_y6;
                            y_out_regs(23) <= lane1_y7;
                            y_out_regs(24) <= lane1_y8;
                            y_out_regs(25) <= lane1_y9;
                            y_out_regs(26) <= lane1_y10;
                            y_out_regs(27) <= lane1_y11;
                            y_out_regs(28) <= lane1_y12;
                            y_out_regs(29) <= lane1_y13;
                            y_out_regs(30) <= lane1_y14;
                            y_out_regs(31) <= lane1_y15;
                            lane1_done_r   <= '1';
                        end if;

                        -- Barrier: only advance once BOTH lanes have
                        -- reported done for this window (either just now
                        -- or on an earlier cycle, via the sticky latches).
                        if (lane0_done_r = '1' or lane0_done = '1') and
                           (lane1_done_r = '1' or lane1_done = '1') then
                            valid_out_r <= '1';

                            if proc_idx = NUM_WINDOWS - 1 then
                                all_done_r <= '1';
                                state      <= S_FINISHED;
                            else
                                proc_idx     <= proc_idx + 1;
                                lane0_start  <= '1';
                                lane1_start  <= '1';
                                lane0_done_r <= '0';
                                lane1_done_r <= '0';
                            end if;
                        end if;

                    when S_FINISHED =>
                        null;
                end case;
            end if;
        end if;
    end process process_fsm;

    valid_out <= valid_out_r;
    all_done  <= all_done_r;

    y0 <= y_out_regs(0);
    y1 <= y_out_regs(1);
    y2 <= y_out_regs(2);
    y3 <= y_out_regs(3);
    y4 <= y_out_regs(4);
    y5 <= y_out_regs(5);
    y6 <= y_out_regs(6);
    y7 <= y_out_regs(7);
    y8 <= y_out_regs(8);
    y9 <= y_out_regs(9);
    y10 <= y_out_regs(10);
    y11 <= y_out_regs(11);
    y12 <= y_out_regs(12);
    y13 <= y_out_regs(13);
    y14 <= y_out_regs(14);
    y15 <= y_out_regs(15);
    y16 <= y_out_regs(16);
    y17 <= y_out_regs(17);
    y18 <= y_out_regs(18);
    y19 <= y_out_regs(19);
    y20 <= y_out_regs(20);
    y21 <= y_out_regs(21);
    y22 <= y_out_regs(22);
    y23 <= y_out_regs(23);
    y24 <= y_out_regs(24);
    y25 <= y_out_regs(25);
    y26 <= y_out_regs(26);
    y27 <= y_out_regs(27);
    y28 <= y_out_regs(28);
    y29 <= y_out_regs(29);
    y30 <= y_out_regs(30);
    y31 <= y_out_regs(31);

end architecture rtl;
