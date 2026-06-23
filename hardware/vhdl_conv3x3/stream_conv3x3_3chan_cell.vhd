-- stream_conv3x3_3chan_cell.vhd
-- Three-input-channel streaming 3×3 convolution cell.
--
-- This is NOT a full U-Net or full Conv2d layer.
-- It computes ONE output channel of ONE Conv2d(3, N, 3) layer:
--
--   pixel_c0 → window_gen_c0 → dot_prod_c0 (bias=0) → y_c0  ─┐
--   pixel_c1 → window_gen_c1 → dot_prod_c1 (bias=0) → y_c1  ─┼→ [bias + y_c0 + y_c1 + y_c2] → y
--   pixel_c2 → window_gen_c2 → dot_prod_c2 (bias=0) → y_c2  ─┘
--
-- The UAVSAR U-Net first layer is Conv2d(3, base_channels, 3, padding=1),
-- where base_channels is a model hyperparameter (e.g. 32 in the tuned model).
-- That layer needs base_channels instances of this cell (one per output channel),
-- each with its own 3×27=81 weight values and one bias.  This prototype implements
-- one of those base_channels output channels.  Padding, multi-output-channel accumulation,
-- activation, and BatchNorm are not included.
--
-- ---- Architecture -------------------------------------------------------
-- Three parallel pipeline chains, all sharing valid_in and clk/rst:
--
--   window3x3_stream ×3    sliding 3×3 window buffers (3 rows × IMG_WIDTH)
--   conv3x3_dot_pipelined ×3   3-stage dot product with zero bias
--   final_sum (1 register)     y_r <= bias + y_c0 + y_c1 + y_c2
--
-- Because all three channels share the same valid_in, their window generators
-- are always in lock-step and their dot-product valid_out signals are identical.
-- The cell uses channel 0's dot-product valid to gate the final summation stage.
--
-- ---- Latency ------------------------------------------------------------
-- 4 clock cycles from triggering pixel to visible output.
--
--   Clock N:   All three window generators process pixel N and schedule
--              win_valid_{c0,c1,c2}='1'.
--   Clock N+1: All three dot-product stage 1 read win_valid='1' (1-cycle
--              interface lag from VHDL delta-cycle ordering).
--   Clock N+2: Dot-product stage 2 (partial sums per channel).
--   Clock N+3: Dot-product stage 3 outputs y_c0, y_c1, y_c2.
--   Clock N+4: Final summation register: y_r <= bias + y_c0 + y_c1 + y_c2.
--              valid_out='1', y valid.
--
-- Throughput: 1 output per clock once primed (within a valid-window row).
--
-- ---- Arithmetic width ---------------------------------------------------
-- INT8 × INT8 → INT32 per dot product.  Three INT32 values are summed with
-- a 32-bit bias; the result fits in INT32 for all realistic U-Net layer
-- weights and activations.  For layers with very large C_in, intermediate
-- widening via resize() would be needed.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity stream_conv3x3_3chan_cell is
    generic (
        IMG_WIDTH : positive := 5
    );
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;        -- synchronous, active high
        valid_in : in  std_logic;

        -- Three input-channel pixel streams (row-major, one triplet per clock)
        pixel_c0 : in signed(7 downto 0);
        pixel_c1 : in signed(7 downto 0);
        pixel_c2 : in signed(7 downto 0);

        -- Channel 0 kernel weights (row-major: c0_w0=top-left, c0_w8=bottom-right)
        c0_w0 : in signed(7 downto 0);
        c0_w1 : in signed(7 downto 0);
        c0_w2 : in signed(7 downto 0);
        c0_w3 : in signed(7 downto 0);
        c0_w4 : in signed(7 downto 0);
        c0_w5 : in signed(7 downto 0);
        c0_w6 : in signed(7 downto 0);
        c0_w7 : in signed(7 downto 0);
        c0_w8 : in signed(7 downto 0);

        -- Channel 1 kernel weights
        c1_w0 : in signed(7 downto 0);
        c1_w1 : in signed(7 downto 0);
        c1_w2 : in signed(7 downto 0);
        c1_w3 : in signed(7 downto 0);
        c1_w4 : in signed(7 downto 0);
        c1_w5 : in signed(7 downto 0);
        c1_w6 : in signed(7 downto 0);
        c1_w7 : in signed(7 downto 0);
        c1_w8 : in signed(7 downto 0);

        -- Channel 2 kernel weights
        c2_w0 : in signed(7 downto 0);
        c2_w1 : in signed(7 downto 0);
        c2_w2 : in signed(7 downto 0);
        c2_w3 : in signed(7 downto 0);
        c2_w4 : in signed(7 downto 0);
        c2_w5 : in signed(7 downto 0);
        c2_w6 : in signed(7 downto 0);
        c2_w7 : in signed(7 downto 0);
        c2_w8 : in signed(7 downto 0);

        -- Output channel bias (added once in the final summation stage)
        bias : in signed(31 downto 0);

        -- Output: one INT32 result per valid window triplet, 4-cycle latency
        valid_out : out std_logic;
        y         : out signed(31 downto 0)
    );
end entity stream_conv3x3_3chan_cell;

architecture rtl of stream_conv3x3_3chan_cell is

    -- ----------------------------------------------------------------
    -- Zero bias constant driven into each per-channel dot product.
    -- The shared bias is added only once in the final summation stage.
    -- ----------------------------------------------------------------
    constant ZERO32 : signed(31 downto 0) := (others => '0');

    -- ----------------------------------------------------------------
    -- Internal handshakes: window-generator → dot-product
    -- All three channels share the same valid_in so their win_valid
    -- signals are always identical in value and timing.
    -- ----------------------------------------------------------------
    signal win_valid_c0, win_valid_c1, win_valid_c2 : std_logic;

    -- ----------------------------------------------------------------
    -- 3×3 pixel windows (9 signals per channel)
    -- ----------------------------------------------------------------
    signal c0_p0, c0_p1, c0_p2, c0_p3, c0_p4, c0_p5,
           c0_p6, c0_p7, c0_p8 : signed(7 downto 0);

    signal c1_p0, c1_p1, c1_p2, c1_p3, c1_p4, c1_p5,
           c1_p6, c1_p7, c1_p8 : signed(7 downto 0);

    signal c2_p0, c2_p1, c2_p2, c2_p3, c2_p4, c2_p5,
           c2_p6, c2_p7, c2_p8 : signed(7 downto 0);

    -- ----------------------------------------------------------------
    -- Per-channel dot-product outputs (zero bias baked in)
    -- ----------------------------------------------------------------
    signal y_c0, y_c1, y_c2 : signed(31 downto 0);
    signal valid_dot         : std_logic;   -- channel 0 gate; all three are identical

    -- ----------------------------------------------------------------
    -- Final registered summation stage (adds shared bias once)
    -- ----------------------------------------------------------------
    signal y_r     : signed(31 downto 0) := (others => '0');
    signal valid_r : std_logic            := '0';

begin

    -- ================================================================
    -- Window generators (one per input channel)
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
    -- Per-channel pipelined dot products (zero bias each)
    -- ================================================================
    dot_c0 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => c0_w0,  w1 => c0_w1,  w2 => c0_w2,
            w3 => c0_w3,  w4 => c0_w4,  w5 => c0_w5,
            w6 => c0_w6,  w7 => c0_w7,  w8 => c0_w8,
            bias      => ZERO32,
            valid_out => valid_dot,     -- drives the final summation gate
            y         => y_c0
        );

    dot_c1 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => c1_w0,  w1 => c1_w1,  w2 => c1_w2,
            w3 => c1_w3,  w4 => c1_w4,  w5 => c1_w5,
            w6 => c1_w6,  w7 => c1_w7,  w8 => c1_w8,
            bias      => ZERO32,
            valid_out => open,          -- identical to valid_dot; not needed separately
            y         => y_c1
        );

    dot_c2 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => c2_w0,  w1 => c2_w1,  w2 => c2_w2,
            w3 => c2_w3,  w4 => c2_w4,  w5 => c2_w5,
            w6 => c2_w6,  w7 => c2_w7,  w8 => c2_w8,
            bias      => ZERO32,
            valid_out => open,          -- identical to valid_dot
            y         => y_c2
        );

    -- ================================================================
    -- Final registered summation stage
    --
    -- Adds the three per-channel dot-product results and the shared
    -- output-channel bias in one registered step.  This extra stage:
    --   (a) avoids a long combinational adder chain on the critical path,
    --   (b) is the reason the total cell latency is 4 cycles (not 3).
    --
    -- Arithmetic: bias + y_c0 + y_c1 + y_c2 all in signed(31 downto 0).
    -- For typical U-Net INT8 weights and 256×256 tiles the values are
    -- well within INT32 range; widening is not required here.
    -- ================================================================
    final_sum : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                y_r     <= (others => '0');
                valid_r <= '0';
            else
                y_r     <= bias + y_c0 + y_c1 + y_c2;
                valid_r <= valid_dot;
            end if;
        end if;
    end process final_sum;

    valid_out <= valid_r;
    y         <= y_r;

end architecture rtl;
