-- conv3x3_dot_time_mux.vhd
-- Time-multiplexed 3x3 signed INT8 dot-product with a SINGLE reusable MAC lane.
--
-- This is NOT a full U-Net or Conv2d implementation, and it is NOT a
-- streaming or multi-kernel accelerator. It is a first PROTOTYPE showing
-- that one 3x3 dot product (9 multiply-accumulates) can be computed with
-- ONE multiplier reused over 9 cycles, instead of 9 parallel multipliers
-- as in conv3x3_dot.vhd / conv3x3_dot_pipelined.vhd / conv3x3_dot_pipelined_dsp.vhd.
--
-- Motivation: the fully parallel 8-output prototype (24 dot-product units,
-- 216 multiplies total) already exceeds the xc7a35t part's 90 DSP48E1
-- slices when synthesized DSP-aware, and consumes 45% of LUTs when
-- synthesized LUT-only. Time-multiplexing is the next architecture idea:
-- reuse a small number of MAC resources across multiple cycles (and,
-- eventually, across multiple kernels/channels) rather than instantiating
-- one dedicated multiplier per weight tap per kernel. This module is the
-- smallest possible step in that direction: ONE dot product, ONE MAC lane,
-- 9 cycles of reuse. It does NOT yet implement scheduling across multiple
-- kernels, channels, or spatial positions -- that would be a follow-on
-- "time-multiplexed 8-output cell", not built here.
--
-- ---- Interface -----------------------------------------------------------
-- Start/done handshake:
--   - While idle, asserting `start` for one cycle latches all 9 pixels (p0..p8),
--     all 9 weights (w0..w8), and the bias, then begins computing.
--   - `start` is ignored while busy (i.e. between an accepted start and the
--     following done pulse) -- the module completes its current dot product
--     before it can accept a new one.
--   - One multiply-accumulate is performed per clock cycle: cycle 1 processes
--     tap 0 (p0*w0), cycle 2 processes tap 1, ..., cycle 9 processes tap 8.
--   - `done` pulses for exactly one cycle when `y` becomes valid.
--
-- ---- Latency ---------------------------------------------------------------
-- 10 clock cycles from the edge that samples start='1' to the edge that
-- asserts done='1' (and updates y):
--   Cycle 0 (load)      : latch p0..p8, w0..w8, bias; initialize accumulator
--   Cycles 1-8          : process taps 0..7, one multiply-accumulate per cycle
--   Cycle 9             : process tap 8 (final), register y, assert done
-- (Cycle numbers above are relative to the load edge = cycle 0; "10 clock
-- cycles" counts the load edge itself plus the 9 tap-processing edges.)
-- This is verified empirically by the accompanying testbench, which counts
-- clocks between the start pulse and the done pulse.
--
-- ---- Resource intent -----------------------------------------------------
-- Exactly ONE 8x8-bit signed multiplier is instantiated (the `p_reg(tap) *
-- w_reg(tap)` expression below), reused for all 9 taps. This contrasts with
-- conv3x3_dot.vhd / conv3x3_dot_pipelined.vhd, which each instantiate 9
-- parallel multipliers to compute the same dot product in 1 (combinational)
-- or 3 (pipelined) cycles. The tradeoff here is explicit: far fewer
-- multiplier resources, at the cost of much higher latency and only
-- one dot product "in flight" at a time (no pipelining/throughput).
--
-- ---- Scope ----------------------------------------------------------------
-- This is a synthesis-and-simulation PROTOTYPE of resource sharing via time
-- multiplexing for a single dot product. It is NOT a complete
-- time-multiplexed convolution engine, does NOT implement a scheduler across
-- kernels/channels/output positions, is NOT board-tested, and makes NO
-- throughput or latency speedup claim relative to the parallel designs.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity conv3x3_dot_time_mux is
    port (
        clk   : in  std_logic;
        rst   : in  std_logic;             -- synchronous reset, active high
        start : in  std_logic;             -- pulse (while idle) to latch inputs and begin

        -- 3x3 pixel window (row-major: p0 p1 p2 / p3 p4 p5 / p6 p7 p8)
        p0 : in signed(7 downto 0);
        p1 : in signed(7 downto 0);
        p2 : in signed(7 downto 0);
        p3 : in signed(7 downto 0);
        p4 : in signed(7 downto 0);
        p5 : in signed(7 downto 0);
        p6 : in signed(7 downto 0);
        p7 : in signed(7 downto 0);
        p8 : in signed(7 downto 0);

        -- 3x3 weight kernel (row-major, same spatial ordering as pixels)
        w0 : in signed(7 downto 0);
        w1 : in signed(7 downto 0);
        w2 : in signed(7 downto 0);
        w3 : in signed(7 downto 0);
        w4 : in signed(7 downto 0);
        w5 : in signed(7 downto 0);
        w6 : in signed(7 downto 0);
        w7 : in signed(7 downto 0);
        w8 : in signed(7 downto 0);

        -- INT32 bias, latched at start along with the pixels/weights
        bias : in  signed(31 downto 0);

        busy : out std_logic;              -- '1' while a dot product is in progress
        done : out std_logic;              -- one-cycle pulse when y is valid
        y    : out signed(31 downto 0)
    );
end entity conv3x3_dot_time_mux;

architecture rtl of conv3x3_dot_time_mux is

    type pixel_arr_t  is array (0 to 8) of signed(7 downto 0);
    type weight_arr_t is array (0 to 8) of signed(7 downto 0);

    type state_t is (S_IDLE, S_COMPUTE);
    signal state : state_t := S_IDLE;

    signal p_reg : pixel_arr_t  := (others => (others => '0'));
    signal w_reg : weight_arr_t := (others => (others => '0'));

    signal tap : integer range 0 to 8 := 0;

    signal acc_r  : signed(31 downto 0) := (others => '0');
    signal y_r    : signed(31 downto 0) := (others => '0');
    signal done_r : std_logic           := '0';

    -- Single reusable MAC product: one 8x8 signed multiply, time-shared
    -- across all 9 taps via p_reg(tap) / w_reg(tap) indexing.
    signal product : signed(15 downto 0);

begin

    product <= p_reg(tap) * w_reg(tap);

    main : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state  <= S_IDLE;
                tap    <= 0;
                acc_r  <= (others => '0');
                y_r    <= (others => '0');
                done_r <= '0';
                p_reg  <= (others => (others => '0'));
                w_reg  <= (others => (others => '0'));

            else
                done_r <= '0';  -- default: done is a one-cycle pulse

                case state is
                    when S_IDLE =>
                        if start = '1' then
                            p_reg(0) <= p0; p_reg(1) <= p1; p_reg(2) <= p2;
                            p_reg(3) <= p3; p_reg(4) <= p4; p_reg(5) <= p5;
                            p_reg(6) <= p6; p_reg(7) <= p7; p_reg(8) <= p8;
                            w_reg(0) <= w0; w_reg(1) <= w1; w_reg(2) <= w2;
                            w_reg(3) <= w3; w_reg(4) <= w4; w_reg(5) <= w5;
                            w_reg(6) <= w6; w_reg(7) <= w7; w_reg(8) <= w8;
                            acc_r <= bias;
                            tap   <= 0;
                            state <= S_COMPUTE;
                        end if;

                    when S_COMPUTE =>
                        -- Accumulate this cycle's tap using the single shared
                        -- multiplier (product <= p_reg(tap) * w_reg(tap)).
                        if tap = 8 then
                            acc_r  <= acc_r + resize(product, 32);
                            y_r    <= acc_r + resize(product, 32);
                            done_r <= '1';
                            state  <= S_IDLE;
                        else
                            acc_r <= acc_r + resize(product, 32);
                            tap   <= tap + 1;
                        end if;
                end case;
            end if;
        end if;
    end process main;

    busy <= '0' when state = S_IDLE else '1';
    done <= done_r;
    y    <= y_r;

end architecture rtl;
