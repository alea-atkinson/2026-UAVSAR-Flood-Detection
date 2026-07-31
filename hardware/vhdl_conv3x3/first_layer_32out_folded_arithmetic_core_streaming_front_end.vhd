-- first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd
--
-- Tiny INTEGRATED SKELETON: chains the NEW 3-channel streaming front end
-- (window3x3_stream_3chan_flattened.vhd) directly into the EXISTING,
-- UNMODIFIED direct-parallel folded 32-output arithmetic core
-- (first_layer_32out_folded_arithmetic_core.vhd), so the whole pipeline
-- can be driven by a real ROW-MAJOR PIXEL STREAM (one pixel triplet per
-- clock) instead of pre-extracted 27-value windows.
--
-- ---- Why this exists -------------------------------------------------------
-- Every prior benchmark/estimate for the direct-parallel arithmetic core
-- (the GPU-vs-FPGA benchmark, the tile-throughput comparison) assumed
-- 3x3x3 windows were ALREADY EXTRACTED, with an explicit "no line-buffer
-- overhead" claim boundary. This skeleton is the concrete feasibility
-- check for that assumption: does a real streaming front end actually
-- exist, does it correctly hand off to the arithmetic core's port
-- interface, and what latency does it add?
--
-- ---- What is reused, unmodified ---------------------------------------
--   window3x3_stream.vhd                          (existing, unmodified)
--   first_layer_32out_folded_arithmetic_core.vhd   (existing, unmodified)
--   first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd (existing,
--       unmodified -- the arithmetic core's own weight/scale/bias source)
--
-- ---- What is NEW ------------------------------------------------------------
--   window3x3_stream_3chan_flattened.vhd (new front-end wrapper, itself
--       built entirely from 3 unmodified window3x3_stream instances)
--   This top-level chaining entity (pure port-map wiring, no new
--       arithmetic logic of its own).
--
-- ---- Scope ------------------------------------------------------------------
-- First layer only. 3 input channels. 3x3 VALID (no padding) convolution
-- windows. NOT a full U-Net, NOT a full 256x256 board implementation,
-- NOT a board demo. This is a feasibility skeleton confirming that a
-- streaming front end CAN correctly feed this specific arithmetic core --
-- it does NOT claim this combined design has been synthesized (see the
-- accompanying report for what has and has not been confirmed).

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity first_layer_32out_folded_arithmetic_core_streaming_front_end is
    generic (
        IMG_WIDTH : positive := 6       -- square image; must be >= 3
    );
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;        -- synchronous, active high
        valid_in : in  std_logic;        -- one new pixel triplet presented this cycle

        -- Three input-channel pixel streams (row-major, one triplet per clock)
        pixel_c0 : in signed(7 downto 0);
        pixel_c1 : in signed(7 downto 0);
        pixel_c2 : in signed(7 downto 0);

        -- Output: all 32 Q.20 fixed-point folded Conv-BN-ReLU results,
        -- ReLU already applied, one pulse per valid 3x3x3 window.
        valid_out : out std_logic;
        y0  : out signed(47 downto 0);  y1  : out signed(47 downto 0);
        y2  : out signed(47 downto 0);  y3  : out signed(47 downto 0);
        y4  : out signed(47 downto 0);  y5  : out signed(47 downto 0);
        y6  : out signed(47 downto 0);  y7  : out signed(47 downto 0);
        y8  : out signed(47 downto 0);  y9  : out signed(47 downto 0);
        y10 : out signed(47 downto 0);  y11 : out signed(47 downto 0);
        y12 : out signed(47 downto 0);  y13 : out signed(47 downto 0);
        y14 : out signed(47 downto 0);  y15 : out signed(47 downto 0);
        y16 : out signed(47 downto 0);  y17 : out signed(47 downto 0);
        y18 : out signed(47 downto 0);  y19 : out signed(47 downto 0);
        y20 : out signed(47 downto 0);  y21 : out signed(47 downto 0);
        y22 : out signed(47 downto 0);  y23 : out signed(47 downto 0);
        y24 : out signed(47 downto 0);  y25 : out signed(47 downto 0);
        y26 : out signed(47 downto 0);  y27 : out signed(47 downto 0);
        y28 : out signed(47 downto 0);  y29 : out signed(47 downto 0);
        y30 : out signed(47 downto 0);  y31 : out signed(47 downto 0)
    );
end entity first_layer_32out_folded_arithmetic_core_streaming_front_end;

architecture rtl of first_layer_32out_folded_arithmetic_core_streaming_front_end is

    signal fe_valid_out : std_logic;
    signal fe_c0_p0, fe_c0_p1, fe_c0_p2, fe_c0_p3, fe_c0_p4, fe_c0_p5, fe_c0_p6, fe_c0_p7, fe_c0_p8 : signed(7 downto 0);
    signal fe_c1_p0, fe_c1_p1, fe_c1_p2, fe_c1_p3, fe_c1_p4, fe_c1_p5, fe_c1_p6, fe_c1_p7, fe_c1_p8 : signed(7 downto 0);
    signal fe_c2_p0, fe_c2_p1, fe_c2_p2, fe_c2_p3, fe_c2_p4, fe_c2_p5, fe_c2_p6, fe_c2_p7, fe_c2_p8 : signed(7 downto 0);

begin

    -- ================================================================
    -- Front end: 3-channel streaming window generator (new wrapper,
    -- built from 3 existing, unmodified window3x3_stream instances).
    -- ================================================================
    front_end : entity work.window3x3_stream_3chan_flattened
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk => clk, rst => rst, valid_in => valid_in,
            pixel_c0 => pixel_c0, pixel_c1 => pixel_c1, pixel_c2 => pixel_c2,
            valid_out => fe_valid_out,
            c0_p0 => fe_c0_p0, c0_p1 => fe_c0_p1, c0_p2 => fe_c0_p2,
            c0_p3 => fe_c0_p3, c0_p4 => fe_c0_p4, c0_p5 => fe_c0_p5,
            c0_p6 => fe_c0_p6, c0_p7 => fe_c0_p7, c0_p8 => fe_c0_p8,
            c1_p0 => fe_c1_p0, c1_p1 => fe_c1_p1, c1_p2 => fe_c1_p2,
            c1_p3 => fe_c1_p3, c1_p4 => fe_c1_p4, c1_p5 => fe_c1_p5,
            c1_p6 => fe_c1_p6, c1_p7 => fe_c1_p7, c1_p8 => fe_c1_p8,
            c2_p0 => fe_c2_p0, c2_p1 => fe_c2_p1, c2_p2 => fe_c2_p2,
            c2_p3 => fe_c2_p3, c2_p4 => fe_c2_p4, c2_p5 => fe_c2_p5,
            c2_p6 => fe_c2_p6, c2_p7 => fe_c2_p7, c2_p8 => fe_c2_p8
        );

    -- ================================================================
    -- Core: the EXISTING, UNMODIFIED direct-parallel folded 32-output
    -- arithmetic core, fed directly from the front end's window outputs.
    -- ================================================================
    core : entity work.first_layer_32out_folded_arithmetic_core
        port map (
            clk => clk, rst => rst, valid_in => fe_valid_out,
            c0_p0 => fe_c0_p0, c0_p1 => fe_c0_p1, c0_p2 => fe_c0_p2,
            c0_p3 => fe_c0_p3, c0_p4 => fe_c0_p4, c0_p5 => fe_c0_p5,
            c0_p6 => fe_c0_p6, c0_p7 => fe_c0_p7, c0_p8 => fe_c0_p8,
            c1_p0 => fe_c1_p0, c1_p1 => fe_c1_p1, c1_p2 => fe_c1_p2,
            c1_p3 => fe_c1_p3, c1_p4 => fe_c1_p4, c1_p5 => fe_c1_p5,
            c1_p6 => fe_c1_p6, c1_p7 => fe_c1_p7, c1_p8 => fe_c1_p8,
            c2_p0 => fe_c2_p0, c2_p1 => fe_c2_p1, c2_p2 => fe_c2_p2,
            c2_p3 => fe_c2_p3, c2_p4 => fe_c2_p4, c2_p5 => fe_c2_p5,
            c2_p6 => fe_c2_p6, c2_p7 => fe_c2_p7, c2_p8 => fe_c2_p8,
            valid_out => valid_out,
            y0 => y0, y1 => y1, y2 => y2, y3 => y3, y4 => y4, y5 => y5, y6 => y6, y7 => y7,
            y8 => y8, y9 => y9, y10 => y10, y11 => y11, y12 => y12, y13 => y13, y14 => y14, y15 => y15,
            y16 => y16, y17 => y17, y18 => y18, y19 => y19, y20 => y20, y21 => y21, y22 => y22, y23 => y23,
            y24 => y24, y25 => y25, y26 => y26, y27 => y27, y28 => y28, y29 => y29, y30 => y30, y31 => y31
        );

end architecture rtl;
