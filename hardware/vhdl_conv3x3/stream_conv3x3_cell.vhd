-- stream_conv3x3_cell.vhd
-- Top-level streaming 3×3 convolution cell.
--
-- This is NOT a full U-Net or full Conv2d layer.
-- It is a first integrated mini convolution accelerator prototype that connects
-- the sliding-window generator and the pipelined dot-product block into a
-- single streaming pipeline:
--
--   pixel_in (row-major stream)
--       ↓
--   window3x3_stream          (line buffers, 3 rows × IMG_WIDTH pixels)
--       ↓  p0..p8 + win_valid
--   conv3x3_dot_pipelined     (3-stage registered dot product, INT8 × INT8 → INT32)
--       ↓  y + valid_out
--   convolution output stream
--
-- Latency (from triggering pixel to visible output):
--   3 clock cycles.
--   Breakdown:
--     Clock N:   window3x3_stream processes pixel N, schedules win_valid_out='1'.
--     Clock N+1: conv3x3_dot_pipelined stage 1 reads win_valid_out='1' (1-cycle
--                interface lag: both submodules are clocked; delta-cycle ordering
--                means the dot-product sees the window's registered output on the
--                FOLLOWING clock, not the same one).
--     Clock N+2: stage 2 partial sums.
--     Clock N+3: stage 3 final output; cell valid_out='1', y valid.
--
-- Throughput:
--   1 result per clock once the pipeline is full (within a row of valid windows).
--   Gaps occur between rows (columns 0–1 of each new row do not produce windows).
--
-- Limitations (still not a full Conv2d layer):
--   - Single input channel only.  A full layer sums over C_in channels.
--   - No padding: first valid output requires 2 full rows + 2 columns.
--   - Weights and bias are applied once per pixel position, not summed
--     over channels.  For C_in>1, instantiate C_in cells and accumulate.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity stream_conv3x3_cell is
    generic (
        IMG_WIDTH : positive := 5       -- pixels per row (propagated to window generator)
    );
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;       -- synchronous, active high; routed to both sub-blocks
        valid_in : in  std_logic;       -- '1' each clock a new pixel is presented
        pixel_in : in  signed(7 downto 0);

        -- 3×3 convolution kernel weights (row-major, fixed for this computation)
        w0 : in signed(7 downto 0);
        w1 : in signed(7 downto 0);
        w2 : in signed(7 downto 0);
        w3 : in signed(7 downto 0);
        w4 : in signed(7 downto 0);
        w5 : in signed(7 downto 0);
        w6 : in signed(7 downto 0);
        w7 : in signed(7 downto 0);
        w8 : in signed(7 downto 0);

        -- INT32 channel bias added to each dot product
        bias      : in  signed(31 downto 0);

        -- Output: one INT32 result per valid 3×3 window, with 3-cycle latency
        valid_out : out std_logic;
        y         : out signed(31 downto 0)
    );
end entity stream_conv3x3_cell;

architecture rtl of stream_conv3x3_cell is

    -- ----------------------------------------------------------------
    -- Internal handshake between window generator and dot-product block
    -- ----------------------------------------------------------------
    signal win_valid : std_logic;

    -- 3×3 pixel window from the sliding-window generator
    signal s_p0, s_p1, s_p2,
           s_p3, s_p4, s_p5,
           s_p6, s_p7, s_p8 : signed(7 downto 0);

begin

    -- ================================================================
    -- Stage A: sliding 3×3 window generator
    -- Accepts a row-major pixel stream; produces overlapping 3×3 windows.
    -- win_valid goes '1' once enough context (2 full rows + 2 columns) exists.
    -- ================================================================
    win_gen : entity work.window3x3_stream
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk       => clk,
            rst       => rst,
            valid_in  => valid_in,
            pixel_in  => pixel_in,
            valid_out => win_valid,
            p0 => s_p0,  p1 => s_p1,  p2 => s_p2,
            p3 => s_p3,  p4 => s_p4,  p5 => s_p5,
            p6 => s_p6,  p7 => s_p7,  p8 => s_p8
        );

    -- ================================================================
    -- Stage B: 3-stage pipelined dot product
    -- Computes  y = bias + Σ(p_i * w_i)  over the 3×3 window.
    -- valid_in is driven by win_valid from stage A.
    -- Due to VHDL concurrent-process semantics, stage B reads the
    -- registered outputs of stage A on the following clock cycle;
    -- this 1-cycle interface lag is included in the 3-cycle total latency.
    -- ================================================================
    dot_prod : entity work.conv3x3_dot_pipelined
        port map (
            clk       => clk,
            rst       => rst,
            valid_in  => win_valid,
            p0 => s_p0,  p1 => s_p1,  p2 => s_p2,
            p3 => s_p3,  p4 => s_p4,  p5 => s_p5,
            p6 => s_p6,  p7 => s_p7,  p8 => s_p8,
            w0 => w0,    w1 => w1,    w2 => w2,
            w3 => w3,    w4 => w4,    w5 => w5,
            w6 => w6,    w7 => w7,    w8 => w8,
            bias      => bias,
            valid_out => valid_out,
            y         => y
        );

end architecture rtl;
