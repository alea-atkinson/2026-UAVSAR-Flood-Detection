-- window3x3_stream_3chan_flattened.vhd
--
-- Minimal 3-CHANNEL STREAMING FRONT END for the direct-parallel folded
-- 32-output arithmetic core (first_layer_32out_folded_arithmetic_core.vhd).
--
-- ---- Why this exists -------------------------------------------------------
-- window3x3_stream.vhd (existing, UNMODIFIED) turns ONE row-major pixel
-- stream into a sliding 3x3 window (9 pixels), single channel only. The
-- direct-parallel arithmetic core's benchmark and tile-throughput reports
-- both assumed 3x3x3 windows were ALREADY EXTRACTED -- this module is the
-- missing piece that answers "what would actually stream pixels in and
-- hand the arithmetic core one flattened 27-value window per cycle?"
--
-- This is NOT a new sliding-window algorithm. It is exactly the SAME
-- pattern already used elsewhere in this repo (e.g.
-- stream_conv3x3_3chan_cell.vhd, conv3x3_3chan_32out_bn_relu_time_mux.vhd):
-- THREE existing, UNMODIFIED window3x3_stream instances, one per input
-- channel, all sharing the same valid_in/clk/rst so their outputs are
-- always in lock-step. This wrapper only adds a name mapping so its 27
-- output ports (c0_p0..c2_p8) match EXACTLY the port names
-- first_layer_32out_folded_arithmetic_core.vhd already expects -- no
-- packing into a single bus is needed in VHDL; 27 individually-named
-- signed(7 downto 0) signals ARE the "flattened 27-value window."
--
-- ---- Latency / throughput (from window3x3_stream's own documented
-- behavior, unchanged here) --------------------------------------------
-- Each of the 3 window3x3_stream instances outputs a registered window
-- ONE cycle after its 9th (bottom-right) pixel is presented at pixel_in.
-- All three channels share valid_in, so their win_valid outputs are
-- always identical in value and timing -- this wrapper's valid_out is
-- simply channel 0's win_valid (channels 1/2 are guaranteed identical).
-- Fill: the first valid window appears once 2 full rows plus 3 pixels of
-- the 3rd row have been streamed (2*IMG_WIDTH + 3 pixels), exactly
-- matching window3x3_stream's own documented/tested behavior
-- (tb_window3x3_stream.vhd: 5x5 image, first window triggered by pixel
-- 13 = 2*5+3). Throughput: 1 window/cycle once primed.
--
-- ---- Scope ------------------------------------------------------------------
-- First layer only, 3 input channels, 3x3 VALID (no padding) convolution
-- windows. No full U-Net, no downstream layers, no board implementation.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity window3x3_stream_3chan_flattened is
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

        -- One-cycle pulse per completed 3x3x3 window, in window order
        -- (row-major). Port names match
        -- first_layer_32out_folded_arithmetic_core.vhd's input ports exactly.
        valid_out : out std_logic;
        c0_p0 : out signed(7 downto 0);  c0_p1 : out signed(7 downto 0);  c0_p2 : out signed(7 downto 0);
        c0_p3 : out signed(7 downto 0);  c0_p4 : out signed(7 downto 0);  c0_p5 : out signed(7 downto 0);
        c0_p6 : out signed(7 downto 0);  c0_p7 : out signed(7 downto 0);  c0_p8 : out signed(7 downto 0);

        c1_p0 : out signed(7 downto 0);  c1_p1 : out signed(7 downto 0);  c1_p2 : out signed(7 downto 0);
        c1_p3 : out signed(7 downto 0);  c1_p4 : out signed(7 downto 0);  c1_p5 : out signed(7 downto 0);
        c1_p6 : out signed(7 downto 0);  c1_p7 : out signed(7 downto 0);  c1_p8 : out signed(7 downto 0);

        c2_p0 : out signed(7 downto 0);  c2_p1 : out signed(7 downto 0);  c2_p2 : out signed(7 downto 0);
        c2_p3 : out signed(7 downto 0);  c2_p4 : out signed(7 downto 0);  c2_p5 : out signed(7 downto 0);
        c2_p6 : out signed(7 downto 0);  c2_p7 : out signed(7 downto 0);  c2_p8 : out signed(7 downto 0)
    );
end entity window3x3_stream_3chan_flattened;

architecture rtl of window3x3_stream_3chan_flattened is

    signal win_valid_c0, win_valid_c1, win_valid_c2 : std_logic;

begin

    -- ================================================================
    -- Three EXISTING, UNMODIFIED window3x3_stream instances, one per
    -- input channel, sharing valid_in/clk/rst (identical timing).
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

    -- All three channels share valid_in, so their win_valid signals are
    -- always identical in value and timing -- channel 0's is used as the
    -- single shared valid_out (same pattern as stream_conv3x3_3chan_cell.vhd).
    valid_out <= win_valid_c0;

end architecture rtl;
