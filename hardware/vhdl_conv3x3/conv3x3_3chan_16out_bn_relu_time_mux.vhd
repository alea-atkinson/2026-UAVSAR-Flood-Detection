-- conv3x3_3chan_16out_bn_relu_time_mux.vhd
-- ONE LANE of the 2-lane resource-shared 32-output folded Conv-BN-ReLU
-- design: a single time-multiplexed dot-product engine and a single
-- shared Q.16 fixed-point BN+ReLU unit, scoped to a 16-kernel SUBSET of
-- the complete 32-kernel first Conv2d layer, selected at elaboration
-- time via the KERNEL_OFFSET generic (0 for kernels 0-15, 16 for kernels
-- 16-31).
--
-- This is architecturally IDENTICAL to
-- conv3x3_3chan_32out_bn_relu_time_mux.vhd (preserved unmodified) --
-- same FSM (S_IDLE/S_RUN/S_BN1/S_BN2), same two-stage Q.16 BN+ReLU
-- sequence, same single conv3x3_dot_time_mux instance reused across all
-- (kernel, channel) steps -- just with a LOCAL kernel count of 16
-- instead of 32, and a KERNEL_OFFSET added when indexing into the
-- EXISTING first_layer_32out_bn_relu_resource_shared_pkg's weight/
-- SCALE_FX/BIAS_FX LUTs (NOT a new package -- see the design memo,
-- multi_lane_resource_shared_conv_bn_relu_design_memo.md Section 4/7,
-- for why reusing the existing 32-kernel package's arrays, sliced per
-- lane by KERNEL_OFFSET, keeps the existing Python-generated golden
-- vectors valid unchanged for a 2-lane design built from two instances
-- of this module).
--
-- Two instances of this module (KERNEL_OFFSET => 0 and KERNEL_OFFSET =>
-- 16) form the two lanes of
-- stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd, run IN PARALLEL
-- on the SAME captured window, each producing its own disjoint 16-kernel
-- slice of that window's 32 outputs.
--
-- ---- Scheduling order (LOCAL to this lane) -------------------------------
-- local_step = local_kernel_idx * 3 + channel_idx, for local_step = 0..47
-- (16 local kernels x 3 channels). Global kernel index used for LUT
-- lookups is KERNEL_OFFSET + local_kernel_idx.
--
-- ---- Interface ------------------------------------------------------------
-- Start/done handshake, one shot per patch (identical contract to
-- conv3x3_3chan_32out_bn_relu_time_mux.vhd):
--   - While idle, asserting `start` for one cycle begins scheduling all 48
--     dot operations (plus 16 BN+ReLU applications) against the currently
--     -presented 3x3x3 patch. The patch must remain stable for the entire run.
--   - `done` pulses for exactly one cycle when y0..y15 (this lane's 16
--     kernels) are all valid.
--   - `start` is ignored while busy.
--
-- ---- Scope ------------------------------------------------------------------
-- This module computes ONE 3-channel 3x3 spatial window's 16 FOLDED
-- Conv-BN-ReLU output values for ITS assigned kernel subset only -- not
-- the complete first Conv2d layer by itself (that requires both lane
-- instances). It does NOT slide a window across an image, does NOT
-- stream pixels, does NOT implement padding, and has NOT been run on
-- real FPGA hardware. It is a resource-sharing PROTOTYPE, not a claim of
-- measured speedup, and not a claim that the full U-Net or downstream
-- layers are implemented.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_32out_bn_relu_resource_shared_pkg.all;

entity conv3x3_3chan_16out_bn_relu_time_mux is
    generic (
        KERNEL_OFFSET : natural := 0    -- 0 for kernels 0-15, 16 for kernels 16-31
    );
    port (
        clk   : in  std_logic;
        rst   : in  std_logic;             -- synchronous reset, active high
        start : in  std_logic;             -- pulse (while idle) to begin scheduling

        -- One 3-channel 3x3 patch (row-major: p0 p1 p2 / p3 p4 p5 / p6 p7 p8),
        -- held stable for the duration of one run.
        c0_p0 : in signed(7 downto 0);  c0_p1 : in signed(7 downto 0);  c0_p2 : in signed(7 downto 0);
        c0_p3 : in signed(7 downto 0);  c0_p4 : in signed(7 downto 0);  c0_p5 : in signed(7 downto 0);
        c0_p6 : in signed(7 downto 0);  c0_p7 : in signed(7 downto 0);  c0_p8 : in signed(7 downto 0);

        c1_p0 : in signed(7 downto 0);  c1_p1 : in signed(7 downto 0);  c1_p2 : in signed(7 downto 0);
        c1_p3 : in signed(7 downto 0);  c1_p4 : in signed(7 downto 0);  c1_p5 : in signed(7 downto 0);
        c1_p6 : in signed(7 downto 0);  c1_p7 : in signed(7 downto 0);  c1_p8 : in signed(7 downto 0);

        c2_p0 : in signed(7 downto 0);  c2_p1 : in signed(7 downto 0);  c2_p2 : in signed(7 downto 0);
        c2_p3 : in signed(7 downto 0);  c2_p4 : in signed(7 downto 0);  c2_p5 : in signed(7 downto 0);
        c2_p6 : in signed(7 downto 0);  c2_p7 : in signed(7 downto 0);  c2_p8 : in signed(7 downto 0);

        busy : out std_logic;              -- '1' while scheduling is in progress
        done : out std_logic;              -- one-cycle pulse when this lane's y0..y15 are all valid
        y0  : out signed(47 downto 0);
        y1  : out signed(47 downto 0);
        y2  : out signed(47 downto 0);
        y3  : out signed(47 downto 0);
        y4  : out signed(47 downto 0);
        y5  : out signed(47 downto 0);
        y6  : out signed(47 downto 0);
        y7  : out signed(47 downto 0);
        y8  : out signed(47 downto 0);
        y9  : out signed(47 downto 0);
        y10 : out signed(47 downto 0);
        y11 : out signed(47 downto 0);
        y12 : out signed(47 downto 0);
        y13 : out signed(47 downto 0);
        y14 : out signed(47 downto 0);
        y15 : out signed(47 downto 0)
    );
end entity conv3x3_3chan_16out_bn_relu_time_mux;

architecture rtl of conv3x3_3chan_16out_bn_relu_time_mux is

    constant LOCAL_NUM_KERNELS : integer := 16;

    -- ----------------------------------------------------------------
    -- Convert the package's plain-integer kernel constants to
    -- signed(7 downto 0) arrays, once, at elaboration time (same helper
    -- pattern as conv3x3_3chan_32out_bn_relu_time_mux.vhd).
    -- ----------------------------------------------------------------
    type signed8_kernel_t is array (0 to 8) of signed(7 downto 0);

    function to_signed_kernel(k : int8_kernel_t) return signed8_kernel_t is
        variable r : signed8_kernel_t;
    begin
        for i in 0 to 8 loop
            r(i) := to_signed(k(i), 8);
        end loop;
        return r;
    end function to_signed_kernel;

    -- ----------------------------------------------------------------
    -- Per-channel pixel windows, packed into arrays for the mux below.
    -- ----------------------------------------------------------------
    signal ch0_arr, ch1_arr, ch2_arr : signed8_kernel_t;

    -- ----------------------------------------------------------------
    -- Scheduler FSM (LOCAL kernel range only -- 0 .. LOCAL_NUM_KERNELS-1;
    -- global package-LUT index is KERNEL_OFFSET + local_kernel_idx).
    -- ----------------------------------------------------------------
    type state_t is (S_IDLE, S_RUN, S_BN1, S_BN2);
    signal state : state_t := S_IDLE;

    signal step        : integer range 0 to LOCAL_NUM_KERNELS * 3 - 1 := 0;
    signal channel_idx  : integer range 0 to 2 := 0;   -- step mod 3
    signal kernel_idx   : integer range 0 to LOCAL_NUM_KERNELS - 1 := 0;   -- step / 3 (local)

    signal partial_sum : signed(31 downto 0) := (others => '0');

    -- ---- Resource-shared Q.16 fixed-point BN+ReLU unit (ONE instance,
    -- reused once per local kernel, two registered stages -- same
    -- structure verified in the 32-kernel single-lane design) ----
    signal raw_sum_r     : signed(31 downto 0) := (others => '0');
    signal bn_kernel_idx : integer range 0 to LOCAL_NUM_KERNELS - 1 := 0;  -- latched local kernel_idx for BN stages
    signal product_fx_r  : signed(47 downto 0) := (others => '0');
    signal biased_fx_comb : signed(47 downto 0);

    type y_arr_t is array (0 to LOCAL_NUM_KERNELS - 1) of signed(47 downto 0);
    signal y_regs : y_arr_t := (others => (others => '0'));

    signal done_r : std_logic := '0';

    -- ----------------------------------------------------------------
    -- Single shared conv3x3_dot_time_mux instance and its interconnect.
    -- ----------------------------------------------------------------
    signal dot_start : std_logic := '0';
    signal dot_busy  : std_logic;
    signal dot_done  : std_logic;
    signal dot_y     : signed(31 downto 0);

    signal dot_p_arr : signed8_kernel_t;
    signal dot_w_arr : signed8_kernel_t;

    signal dot_p0, dot_p1, dot_p2, dot_p3, dot_p4, dot_p5, dot_p6, dot_p7, dot_p8 : signed(7 downto 0);
    signal dot_w0, dot_w1, dot_w2, dot_w3, dot_w4, dot_w5, dot_w6, dot_w7, dot_w8 : signed(7 downto 0);

    constant ZERO32 : signed(31 downto 0) := (others => '0');

begin

    -- ================================================================
    -- Pack input ports into arrays and derive step -> (kernel, channel).
    -- ================================================================
    ch0_arr(0) <= c0_p0; ch0_arr(1) <= c0_p1; ch0_arr(2) <= c0_p2;
    ch0_arr(3) <= c0_p3; ch0_arr(4) <= c0_p4; ch0_arr(5) <= c0_p5;
    ch0_arr(6) <= c0_p6; ch0_arr(7) <= c0_p7; ch0_arr(8) <= c0_p8;

    ch1_arr(0) <= c1_p0; ch1_arr(1) <= c1_p1; ch1_arr(2) <= c1_p2;
    ch1_arr(3) <= c1_p3; ch1_arr(4) <= c1_p4; ch1_arr(5) <= c1_p5;
    ch1_arr(6) <= c1_p6; ch1_arr(7) <= c1_p7; ch1_arr(8) <= c1_p8;

    ch2_arr(0) <= c2_p0; ch2_arr(1) <= c2_p1; ch2_arr(2) <= c2_p2;
    ch2_arr(3) <= c2_p3; ch2_arr(4) <= c2_p4; ch2_arr(5) <= c2_p5;
    ch2_arr(6) <= c2_p6; ch2_arr(7) <= c2_p7; ch2_arr(8) <= c2_p8;

    channel_idx <= step mod 3;
    kernel_idx  <= step / 3;

    -- ================================================================
    -- Combinational muxes: select this step's pixel channel and folded
    -- weight set (from the EXISTING package's per-channel weight LUTs,
    -- indexed by KERNEL_OFFSET + local kernel_idx) for the shared
    -- conv3x3_dot_time_mux instance.
    -- ================================================================
    dot_p_arr <= ch0_arr when channel_idx = 0 else
                 ch1_arr when channel_idx = 1 else
                 ch2_arr;

    dot_p0 <= dot_p_arr(0); dot_p1 <= dot_p_arr(1); dot_p2 <= dot_p_arr(2);
    dot_p3 <= dot_p_arr(3); dot_p4 <= dot_p_arr(4); dot_p5 <= dot_p_arr(5);
    dot_p6 <= dot_p_arr(6); dot_p7 <= dot_p_arr(7); dot_p8 <= dot_p_arr(8);

    dot_w_arr <= to_signed_kernel(CH0_WEIGHT_LUT(KERNEL_OFFSET + kernel_idx)) when channel_idx = 0 else
                 to_signed_kernel(CH1_WEIGHT_LUT(KERNEL_OFFSET + kernel_idx)) when channel_idx = 1 else
                 to_signed_kernel(CH2_WEIGHT_LUT(KERNEL_OFFSET + kernel_idx));

    dot_w0 <= dot_w_arr(0); dot_w1 <= dot_w_arr(1); dot_w2 <= dot_w_arr(2);
    dot_w3 <= dot_w_arr(3); dot_w4 <= dot_w_arr(4); dot_w5 <= dot_w_arr(5);
    dot_w6 <= dot_w_arr(6); dot_w7 <= dot_w_arr(7); dot_w8 <= dot_w_arr(8);

    -- ================================================================
    -- The ONE reusable time-multiplexed dot-product lane, shared across
    -- all 48 (local kernel, channel) operations assigned to this lane.
    -- ================================================================
    dot_inst : entity work.conv3x3_dot_time_mux
        port map (
            clk   => clk,
            rst   => rst,
            start => dot_start,
            p0 => dot_p0, p1 => dot_p1, p2 => dot_p2,
            p3 => dot_p3, p4 => dot_p4, p5 => dot_p5,
            p6 => dot_p6, p7 => dot_p7, p8 => dot_p8,
            w0 => dot_w0, w1 => dot_w1, w2 => dot_w2,
            w3 => dot_w3, w4 => dot_w4, w5 => dot_w5,
            w6 => dot_w6, w7 => dot_w7, w8 => dot_w8,
            bias  => ZERO32,   -- per-channel bias always 0; folded kernel bias added in BN+ReLU stage
            busy  => dot_busy,
            done  => dot_done,
            y     => dot_y
        );

    -- ================================================================
    -- Resource-shared BN+ReLU Stage 2 combinational expression (registered
    -- into y_regs in the FSM below), using the EXISTING package's
    -- BIAS_FX_LUT indexed by KERNEL_OFFSET + local bn_kernel_idx.
    -- ================================================================
    biased_fx_comb <= product_fx_r + to_signed(BIAS_FX_LUT(KERNEL_OFFSET + bn_kernel_idx), 48);

    -- ================================================================
    -- Scheduler FSM: dispatches this lane's 48 dot operations (S_RUN),
    -- then for each completed local kernel (channel_idx = 2), runs the
    -- shared 2-stage BN+ReLU sequence (S_BN1: register product_fx; S_BN2:
    -- compute biased_fx, apply ReLU, register y_regs(kernel_idx)) before
    -- either dispatching the next local kernel's first dot op or
    -- asserting done.
    -- ================================================================
    main : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state         <= S_IDLE;
                step          <= 0;
                partial_sum   <= (others => '0');
                raw_sum_r     <= (others => '0');
                bn_kernel_idx <= 0;
                product_fx_r  <= (others => '0');
                y_regs        <= (others => (others => '0'));
                done_r        <= '0';
                dot_start     <= '0';

            else
                done_r <= '0';

                case state is
                    when S_IDLE =>
                        dot_start <= '0';
                        if start = '1' then
                            step        <= 0;
                            partial_sum <= (others => '0');
                            dot_start   <= '1';
                            state       <= S_RUN;
                        end if;

                    when S_RUN =>
                        -- De-assert the one-cycle start pulse issued last edge.
                        if dot_start = '1' then
                            dot_start <= '0';
                        end if;

                        if dot_done = '1' then
                            if channel_idx = 2 then
                                -- Third channel of this local kernel just
                                -- completed: latch the exact raw INT32 sum
                                -- and enter the shared BN+ReLU sequence.
                                raw_sum_r     <= partial_sum + resize(dot_y, 32);
                                bn_kernel_idx <= kernel_idx;
                                partial_sum   <= (others => '0');
                                state         <= S_BN1;
                            else
                                partial_sum <= partial_sum + resize(dot_y, 32);
                                step        <= step + 1;
                                dot_start   <= '1';
                            end if;
                        end if;

                    when S_BN1 =>
                        -- Stage 1: register product_fx = raw_sum_r * SCALE_FX(global kernel)
                        product_fx_r <= resize(raw_sum_r * to_signed(SCALE_FX_LUT(KERNEL_OFFSET + bn_kernel_idx), 18), 48);
                        state        <= S_BN2;

                    when S_BN2 =>
                        -- Stage 2: biased_fx = product_fx_r + BIAS_FX(global kernel); ReLU;
                        -- register final y_regs(local kernel), then decide next step.
                        if biased_fx_comb > 0 then
                            y_regs(bn_kernel_idx) <= biased_fx_comb;
                        else
                            y_regs(bn_kernel_idx) <= (others => '0');
                        end if;

                        if bn_kernel_idx = LOCAL_NUM_KERNELS - 1 then
                            done_r <= '1';
                            state  <= S_IDLE;
                        else
                            step      <= step + 1;
                            dot_start <= '1';
                            state     <= S_RUN;
                        end if;
                end case;
            end if;
        end if;
    end process main;

    busy <= '0' when state = S_IDLE else '1';
    done <= done_r;

    y0  <= y_regs(0);
    y1  <= y_regs(1);
    y2  <= y_regs(2);
    y3  <= y_regs(3);
    y4  <= y_regs(4);
    y5  <= y_regs(5);
    y6  <= y_regs(6);
    y7  <= y_regs(7);
    y8  <= y_regs(8);
    y9  <= y_regs(9);
    y10 <= y_regs(10);
    y11 <= y_regs(11);
    y12 <= y_regs(12);
    y13 <= y_regs(13);
    y14 <= y_regs(14);
    y15 <= y_regs(15);

end architecture rtl;
