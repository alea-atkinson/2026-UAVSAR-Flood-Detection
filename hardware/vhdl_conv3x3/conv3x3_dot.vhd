-- conv3x3_dot.vhd
-- Purely combinational 3x3 dot-product block.
--
-- Computes:  y = bias + sum(p_i * w_i)  for i = 0..8
--
-- This is the core operation of one output channel, one spatial position,
-- in a single-input-channel 3x3 Conv2d layer — the dominant operation in the
-- U-Net DoubleConv blocks responsible for ~12 GMACs per 256x256 tile.
--
-- This is NOT a full Conv2d implementation. A real Conv2d layer:
--   - sums over all input channels (not just 1)
--   - slides this window across the full spatial map (H x W positions)
--   - uses a line buffer to stream pixel windows without re-reading from DRAM
--   - chains DoubleConv pairs for each encoder/decoder stage
--
-- Bit widths:
--   p_i, w_i : signed(7 downto 0)   -- INT8 pixels and weights
--   product   : signed(15 downto 0)  -- 8*8 = 16-bit intermediate  (auto in numeric_std)
--   bias      : signed(31 downto 0)  -- INT32 bias matches INT8 GEMM convention
--   y         : signed(31 downto 0)  -- INT32 output (before quantization / activation)
--
-- Maximum intermediate value: 9 * (127*127) = 145,161 < 2^17 < 2^31.
-- Signed 32-bit accumulator is safe against overflow for any INT8 input/weight
-- combination across the 9 kernel positions.
--
-- Latch avoidance: only concurrent signal assignments are used; no process
-- with incomplete sensitivity lists. Synthesis tools should infer pure LUT logic.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity conv3x3_dot is
    port (
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
        -- 3x3 weight kernel (row-major, same ordering as pixels)
        w0 : in signed(7 downto 0);
        w1 : in signed(7 downto 0);
        w2 : in signed(7 downto 0);
        w3 : in signed(7 downto 0);
        w4 : in signed(7 downto 0);
        w5 : in signed(7 downto 0);
        w6 : in signed(7 downto 0);
        w7 : in signed(7 downto 0);
        w8 : in signed(7 downto 0);
        -- INT32 bias
        bias : in  signed(31 downto 0);
        -- INT32 dot-product result
        y    : out signed(31 downto 0)
    );
end entity conv3x3_dot;

architecture rtl of conv3x3_dot is
    -- 16-bit intermediate products: signed(7..0) * signed(7..0) = signed(15..0)
    -- IEEE numeric_std "*" on SIGNED returns length L'LENGTH + R'LENGTH = 16 bits.
    signal pr0 : signed(15 downto 0);
    signal pr1 : signed(15 downto 0);
    signal pr2 : signed(15 downto 0);
    signal pr3 : signed(15 downto 0);
    signal pr4 : signed(15 downto 0);
    signal pr5 : signed(15 downto 0);
    signal pr6 : signed(15 downto 0);
    signal pr7 : signed(15 downto 0);
    signal pr8 : signed(15 downto 0);
begin

    -- Multipliers (combinational)
    pr0 <= p0 * w0;
    pr1 <= p1 * w1;
    pr2 <= p2 * w2;
    pr3 <= p3 * w3;
    pr4 <= p4 * w4;
    pr5 <= p5 * w5;
    pr6 <= p6 * w6;
    pr7 <= p7 * w7;
    pr8 <= p8 * w8;

    -- Adder tree: resize each 16-bit product to 32 bits (sign-extend), then sum.
    -- All arithmetic is signed; no unsigned casting needed.
    y <= bias
       + resize(pr0, 32)
       + resize(pr1, 32)
       + resize(pr2, 32)
       + resize(pr3, 32)
       + resize(pr4, 32)
       + resize(pr5, 32)
       + resize(pr6, 32)
       + resize(pr7, 32)
       + resize(pr8, 32);

end architecture rtl;
