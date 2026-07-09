-- stream_conv3x3_3chan_4out_bn_relu_time_mux.vhd
-- Streaming wrapper around the resource-shared one-window scheduler
-- conv3x3_3chan_4out_bn_relu_time_mux.
--
-- This is NOT full U-Net FPGA inference and NOT a board demo. It IS the
-- smallest serious implemented prototype of the resource-shared
-- architecture described in
-- hardware/vhdl_conv3x3/resource_shared_first_layer_conv_bn_relu_plan.md
-- (which targets the complete 32-output first-layer folded Conv-BN-ReLU
-- stage): this module covers 4 of 32 output channels, but exercises the
-- FULL resource-sharing pattern -- ONE dot-product engine and ONE Q.16
-- fixed-point BN+ReLU unit, each reused across all kernels and all
-- windows, rather than any per-kernel or per-window dedicated hardware.
--
-- ---- Two-phase, non-overlapped design (identical pattern to
-- stream_conv3x3_3chan_8out_time_mux.vhd) -----------------------------------
-- The window generators (window3x3_stream, unmodified, one per input
-- channel) produce one valid 3x3 window per clock once primed, with NO
-- back-pressure support. The shared conv3x3_3chan_4out_bn_relu_time_mux
-- scheduler takes many clocks to process ONE window (12 dot ops x 10
-- cycles + gaps + 4 x 2-cycle BN+ReLU applications). This module uses the
-- SAME simple two-phase approach already established for the raw 8-output
-- time-mux design:
--
--   PHASE 1 (CAPTURE): pixels stream in via valid_in/pixel_c0/pixel_c1/
--   pixel_c2. The three window3x3_stream instances produce valid 3x3
--   windows as usual; each is captured into an internal buffer (indexed
--   0 .. NUM_WINDOWS-1). For the canonical 5x5 image, this takes ~25
--   clocks and produces NUM_WINDOWS = (IMG_WIDTH-2)^2 = 9 buffered windows.
--
--   PHASE 2 (PROCESS): once all NUM_WINDOWS windows are captured, this
--   module autonomously feeds them into ONE
--   conv3x3_3chan_4out_bn_relu_time_mux instance, one window at a time:
--   present window i's 27 pixels, pulse the scheduler's start, wait for
--   its done, latch y0..y3 (Q.16 fixed-point) and pulse this module's own
--   valid_out, then move to window i+1. NOT overlapped with phase 1 or
--   with itself.
--
-- ---- Output ordering ------------------------------------------------------
-- WINDOW-MAJOR, KERNEL-MINOR: each valid_out pulse carries all 4 kernel
-- outputs for ONE window, in the same row-major window order
-- window3x3_stream already produces -- matching
-- resource_shared_first_layer_conv_bn_relu_plan.md Section 11 and the
-- Python golden generator's emission order.
--
-- ---- Limitation: buffered, not truly streaming end-to-end -----------------
-- Same as stream_conv3x3_3chan_8out_time_mux.vhd: phase 2 is far slower
-- per window than phase 1, so all windows must be captured before
-- processing begins. For a 5x5 image this buffering is trivial (9 windows
-- x 27 INT8 values = 243 bytes of registers). Not a real streaming buffer
-- for arbitrarily large images.
--
-- ---- Square-image assumption -----------------------------------------------
-- NUM_WINDOWS is computed from the IMG_WIDTH generic as (IMG_WIDTH-2)**2,
-- matching window3x3_stream.vhd's own valid-region rule and every existing
-- testbench in this directory. IMG_WIDTH must be >= 3.
--
-- ---- Weight/constant source / scope ---------------------------------------
-- Kernels 0-3's folded INT8 weights and Q.16 SCALE_FX/BIAS_FX constants
-- come from first_layer_4out_bn_relu_resource_shared_pkg.vhd via the
-- unmodified conv3x3_3chan_4out_bn_relu_time_mux and conv3x3_dot_time_mux
-- submodules. Only 4 of the 32 first-layer output channels are covered.
-- This is still NOT full U-Net FPGA inference, NOT a board demo, and NOT
-- a measured speedup claim.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity stream_conv3x3_3chan_4out_bn_relu_time_mux is
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
        y0 : out signed(47 downto 0);       -- Q.16 fixed-point, ReLU applied
        y1 : out signed(47 downto 0);
        y2 : out signed(47 downto 0);
        y3 : out signed(47 downto 0);

        -- One-cycle pulse after the LAST window's outputs are valid.
        all_done : out std_logic
    );
end entity stream_conv3x3_3chan_4out_bn_relu_time_mux;

architecture rtl of stream_conv3x3_3chan_4out_bn_relu_time_mux is

    constant NUM_WINDOWS : positive := (IMG_WIDTH - 2) * (IMG_WIDTH - 2);

    -- ----------------------------------------------------------------
    -- Window generators -- ONE PER INPUT CHANNEL, unmodified reuse of
    -- window3x3_stream.vhd.
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
    -- channel per slot.
    -- ----------------------------------------------------------------
    type win9_t     is array (0 to 8) of signed(7 downto 0);
    type win_buf_t  is array (0 to NUM_WINDOWS - 1) of win9_t;

    signal buf_c0, buf_c1, buf_c2 : win_buf_t;

    signal wr_ptr : integer range 0 to NUM_WINDOWS := 0;

    -- ----------------------------------------------------------------
    -- Phase 2 processing: one shared conv3x3_3chan_4out_bn_relu_time_mux
    -- instance, fed sequentially from the capture buffer.
    -- ----------------------------------------------------------------
    signal proc_idx : integer range 0 to NUM_WINDOWS - 1 := 0;

    signal sched_start : std_logic := '0';
    signal sched_busy   : std_logic;
    signal sched_done   : std_logic;
    signal sched_y0, sched_y1, sched_y2, sched_y3 : signed(47 downto 0);

    signal sched_c0_p0, sched_c0_p1, sched_c0_p2, sched_c0_p3, sched_c0_p4,
           sched_c0_p5, sched_c0_p6, sched_c0_p7, sched_c0_p8 : signed(7 downto 0);
    signal sched_c1_p0, sched_c1_p1, sched_c1_p2, sched_c1_p3, sched_c1_p4,
           sched_c1_p5, sched_c1_p6, sched_c1_p7, sched_c1_p8 : signed(7 downto 0);
    signal sched_c2_p0, sched_c2_p1, sched_c2_p2, sched_c2_p3, sched_c2_p4,
           sched_c2_p5, sched_c2_p6, sched_c2_p7, sched_c2_p8 : signed(7 downto 0);

    type y_arr_t is array (0 to 3) of signed(47 downto 0);
    signal y_out_regs : y_arr_t := (others => (others => '0'));

    signal valid_out_r : std_logic := '0';
    signal all_done_r  : std_logic := '0';

    type state_t is (S_CAPTURE, S_RUN, S_FINISHED);
    signal state : state_t := S_CAPTURE;

begin

    -- ================================================================
    -- PHASE 1: window generators (one per input channel), unmodified
    -- reuse of window3x3_stream.vhd.
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
    -- PHASE 1: capture each valid window into the buffer.
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
    -- (proc_idx) to the shared scheduler's pixel ports.
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
    -- The ONE shared resource-shared one-window scheduler, reused
    -- sequentially across all NUM_WINDOWS buffered windows.
    -- ================================================================
    sched_inst : entity work.conv3x3_3chan_4out_bn_relu_time_mux
        port map (
            clk   => clk,
            rst   => rst,
            start => sched_start,
            c0_p0 => sched_c0_p0, c0_p1 => sched_c0_p1, c0_p2 => sched_c0_p2,
            c0_p3 => sched_c0_p3, c0_p4 => sched_c0_p4, c0_p5 => sched_c0_p5,
            c0_p6 => sched_c0_p6, c0_p7 => sched_c0_p7, c0_p8 => sched_c0_p8,
            c1_p0 => sched_c1_p0, c1_p1 => sched_c1_p1, c1_p2 => sched_c1_p2,
            c1_p3 => sched_c1_p3, c1_p4 => sched_c1_p4, c1_p5 => sched_c1_p5,
            c1_p6 => sched_c1_p6, c1_p7 => sched_c1_p7, c1_p8 => sched_c1_p8,
            c2_p0 => sched_c2_p0, c2_p1 => sched_c2_p1, c2_p2 => sched_c2_p2,
            c2_p3 => sched_c2_p3, c2_p4 => sched_c2_p4, c2_p5 => sched_c2_p5,
            c2_p6 => sched_c2_p6, c2_p7 => sched_c2_p7, c2_p8 => sched_c2_p8,
            busy  => sched_busy,
            done  => sched_done,
            y0 => sched_y0, y1 => sched_y1, y2 => sched_y2, y3 => sched_y3
        );

    -- ================================================================
    -- PHASE 2 FSM: wait for capture to complete (wr_ptr = NUM_WINDOWS),
    -- then sequentially process each buffered window one at a time.
    -- ================================================================
    process_fsm : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state       <= S_CAPTURE;
                proc_idx    <= 0;
                sched_start <= '0';
                valid_out_r <= '0';
                all_done_r  <= '0';
                y_out_regs  <= (others => (others => '0'));

            else
                valid_out_r <= '0';
                all_done_r  <= '0';

                case state is
                    when S_CAPTURE =>
                        sched_start <= '0';
                        if wr_ptr = NUM_WINDOWS then
                            proc_idx    <= 0;
                            sched_start <= '1';
                            state       <= S_RUN;
                        end if;

                    when S_RUN =>
                        if sched_start = '1' then
                            sched_start <= '0';
                        end if;

                        if sched_done = '1' then
                            y_out_regs(0) <= sched_y0; y_out_regs(1) <= sched_y1;
                            y_out_regs(2) <= sched_y2; y_out_regs(3) <= sched_y3;
                            valid_out_r   <= '1';

                            if proc_idx = NUM_WINDOWS - 1 then
                                all_done_r <= '1';
                                state      <= S_FINISHED;
                            else
                                proc_idx    <= proc_idx + 1;
                                sched_start <= '1';
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

end architecture rtl;
