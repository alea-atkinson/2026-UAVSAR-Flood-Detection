-- stream_conv3x3_3chan_8out_cell.vhd
-- Three-input-channel streaming 3x3 convolution cell with EIGHT parallel
-- output channels (kernels 0-7 of the first U-Net Conv2d layer).
--
-- This is NOT a full U-Net or full Conv2d layer.
-- It computes EIGHT output channels of ONE Conv2d(3, N, 3) layer, sharing
-- one 3x3 sliding window per input channel across all eight kernels -- the
-- same window-reuse structure as stream_conv3x3_3chan_4out_cell.vhd, scaled
-- from 4 to 8 output kernels.
--
-- The UAVSAR U-Net first layer is Conv2d(3, base_channels, 3, padding=1),
-- where base_channels=32 in the tuned model. That layer needs base_channels
-- output-channel datapaths; this prototype implements EIGHT of them (kernels
-- 0-7) sharing the input window logic. Padding, the remaining 24 output
-- channels, activation, and BatchNorm are not included.
--
-- ---- Architecture (window reuse) -----------------------------------------
-- Three window3x3_stream instances (one PER INPUT CHANNEL, not per kernel):
-- all eight output kernels read the SAME c0_p0..c0_p8 / c1_p* / c2_p* window
-- signals, just with different weight constants. Four independent
-- single-output cells would instantiate 8x3=24 window generators; here only
-- 3 are instantiated and reused by all 8 kernels.
--
--   window3x3_stream ×3          sliding 3x3 window buffers, shared by all kernels
--   conv3x3_dot_pipelined ×24    8 kernels x 3 input channels, zero bias each
--   final_sum ×8 (1 register each)   y{k}_r <= bias_k + yk_c0 + yk_c1 + yk_c2
--
-- Because all input channels (and therefore all kernels) share the same
-- valid_in, every window generator is in lock-step and every dot-product's
-- valid_out is identical. The cell uses kernel 0 / channel 0's dot-product
-- valid to gate all eight final summation stages.
--
-- ---- Latency ---------------------------------------------------------------
-- 4 clock cycles from triggering pixel to visible output -- IDENTICAL to
-- stream_conv3x3_3chan_cell.vhd and stream_conv3x3_3chan_4out_cell.vhd,
-- because each kernel's datapath (window gen -> conv3x3_dot_pipelined ->
-- final registered sum) is structurally the same chain, just replicated 8x
-- with shared window generators feeding all eight.
--
--   Clock N:   All three window generators process pixel N.
--   Clock N+1: All twenty-four dot-product stage-1 registers latch win_valid='1'.
--   Clock N+2: Dot-product stage 2 (partial sums per channel per kernel).
--   Clock N+3: Dot-product stage 3 outputs y{k}_c0, y{k}_c1, y{k}_c2 for k=0..7.
--   Clock N+4: Final summation registers: y{k}_r <= bias_k + sum(y{k}_c*).
--              valid_out='1'; y0..y7 all valid simultaneously.
--
-- Throughput: 1 set of 8 outputs per clock once primed (within a valid-window row).
--
-- ---- Weight source -----------------------------------------------------
-- Kernels 0-7 (INT8 weights, symmetric per-tensor quantization) are taken
-- directly from hardware/vhdl_conv3x3/first_layer_kernels0_to7_pkg.vhd,
-- generated from enc1.block.0.weight of the trained Alea-tuned checkpoint.
-- Bias is 0 for all eight kernels (Conv2d has no direct bias; BatchNorm is
-- NOT folded in here).
--
-- ---- Arithmetic width ---------------------------------------------------
-- INT8 x INT8 -> INT32 per dot product, identical to the 1-output and
-- 4-output cells. Twenty-four DSP-mappable multiply groups (8 kernels x 3
-- channels x 3x3=9 taps each = 216 multiplies total) run in parallel.
--
-- ---- Scope ----------------------------------------------------------------
-- This is an 8-output first-layer convolution HARDWARE PROTOTYPE, not full
-- U-Net FPGA inference, not board-tested, and does not include BatchNorm
-- folding or measured speedup figures. It demonstrates that eight real
-- trained-model kernels can be evaluated in parallel from one shared input
-- window, extending the same structure used by the 4-output prototype.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_layer_kernels0_to7_pkg.all;

entity stream_conv3x3_3chan_8out_cell is
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

        -- Output: eight INT32 results (one per kernel) per valid window
        -- triplet, 4-cycle latency -- identical timing to
        -- stream_conv3x3_3chan_cell.vhd's and
        -- stream_conv3x3_3chan_4out_cell.vhd's valid_out/y.
        valid_out : out std_logic;
        y0 : out signed(31 downto 0);   -- kernel 0 output
        y1 : out signed(31 downto 0);   -- kernel 1 output
        y2 : out signed(31 downto 0);   -- kernel 2 output
        y3 : out signed(31 downto 0);   -- kernel 3 output
        y4 : out signed(31 downto 0);   -- kernel 4 output
        y5 : out signed(31 downto 0);   -- kernel 5 output
        y6 : out signed(31 downto 0);   -- kernel 6 output
        y7 : out signed(31 downto 0)    -- kernel 7 output
    );
end entity stream_conv3x3_3chan_8out_cell;

architecture rtl of stream_conv3x3_3chan_8out_cell is

    -- ----------------------------------------------------------------
    -- Zero bias constant driven into each per-channel dot product.
    -- Each kernel's shared bias is added only once in its final
    -- summation stage.
    -- ----------------------------------------------------------------
    constant ZERO32 : signed(31 downto 0) := (others => '0');

    -- ----------------------------------------------------------------
    -- Convert the package's plain-integer kernel constants to
    -- signed(7 downto 0) arrays, once, at elaboration time.
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

    constant K0_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL0_CH0_W);
    constant K0_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL0_CH1_W);
    constant K0_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL0_CH2_W);
    constant K0_BIAS32 : signed(31 downto 0) := to_signed(KERNEL0_BIAS, 32);

    constant K1_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL1_CH0_W);
    constant K1_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL1_CH1_W);
    constant K1_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL1_CH2_W);
    constant K1_BIAS32 : signed(31 downto 0) := to_signed(KERNEL1_BIAS, 32);

    constant K2_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL2_CH0_W);
    constant K2_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL2_CH1_W);
    constant K2_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL2_CH2_W);
    constant K2_BIAS32 : signed(31 downto 0) := to_signed(KERNEL2_BIAS, 32);

    constant K3_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL3_CH0_W);
    constant K3_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL3_CH1_W);
    constant K3_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL3_CH2_W);
    constant K3_BIAS32 : signed(31 downto 0) := to_signed(KERNEL3_BIAS, 32);

    constant K4_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL4_CH0_W);
    constant K4_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL4_CH1_W);
    constant K4_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL4_CH2_W);
    constant K4_BIAS32 : signed(31 downto 0) := to_signed(KERNEL4_BIAS, 32);

    constant K5_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL5_CH0_W);
    constant K5_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL5_CH1_W);
    constant K5_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL5_CH2_W);
    constant K5_BIAS32 : signed(31 downto 0) := to_signed(KERNEL5_BIAS, 32);

    constant K6_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL6_CH0_W);
    constant K6_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL6_CH1_W);
    constant K6_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL6_CH2_W);
    constant K6_BIAS32 : signed(31 downto 0) := to_signed(KERNEL6_BIAS, 32);

    constant K7_CH0 : signed8_kernel_t := to_signed_kernel(KERNEL7_CH0_W);
    constant K7_CH1 : signed8_kernel_t := to_signed_kernel(KERNEL7_CH1_W);
    constant K7_CH2 : signed8_kernel_t := to_signed_kernel(KERNEL7_CH2_W);
    constant K7_BIAS32 : signed(31 downto 0) := to_signed(KERNEL7_BIAS, 32);

    -- ----------------------------------------------------------------
    -- Internal handshakes: window-generator -> dot-product.
    -- All three channels share the same valid_in so their win_valid
    -- signals are always identical in value and timing.
    -- ----------------------------------------------------------------
    signal win_valid_c0, win_valid_c1, win_valid_c2 : std_logic;

    -- ----------------------------------------------------------------
    -- 3x3 pixel windows (9 signals per input channel), shared by all
    -- eight output kernels.
    -- ----------------------------------------------------------------
    signal c0_p0, c0_p1, c0_p2, c0_p3, c0_p4, c0_p5,
           c0_p6, c0_p7, c0_p8 : signed(7 downto 0);

    signal c1_p0, c1_p1, c1_p2, c1_p3, c1_p4, c1_p5,
           c1_p6, c1_p7, c1_p8 : signed(7 downto 0);

    signal c2_p0, c2_p1, c2_p2, c2_p3, c2_p4, c2_p5,
           c2_p6, c2_p7, c2_p8 : signed(7 downto 0);

    -- ----------------------------------------------------------------
    -- Per-kernel, per-channel dot-product outputs (zero bias baked in)
    -- ----------------------------------------------------------------
    signal y0_c0, y0_c1, y0_c2 : signed(31 downto 0);
    signal y1_c0, y1_c1, y1_c2 : signed(31 downto 0);
    signal y2_c0, y2_c1, y2_c2 : signed(31 downto 0);
    signal y3_c0, y3_c1, y3_c2 : signed(31 downto 0);
    signal y4_c0, y4_c1, y4_c2 : signed(31 downto 0);
    signal y5_c0, y5_c1, y5_c2 : signed(31 downto 0);
    signal y6_c0, y6_c1, y6_c2 : signed(31 downto 0);
    signal y7_c0, y7_c1, y7_c2 : signed(31 downto 0);

    -- Single shared valid gate: kernel 0 / channel 0's dot valid_out.
    -- All twenty-four dot products share identical valid_in timing, so
    -- their valid_out signals are identical; only one needs to be observed.
    signal valid_dot : std_logic;

    -- ----------------------------------------------------------------
    -- Final registered summation stage, one per kernel (adds that
    -- kernel's bias once).
    -- ----------------------------------------------------------------
    signal y0_r, y1_r, y2_r, y3_r, y4_r, y5_r, y6_r, y7_r : signed(31 downto 0) := (others => '0');
    signal valid_r                : std_logic            := '0';

begin

    -- ================================================================
    -- Window generators -- ONE PER INPUT CHANNEL, shared by all 8 kernels.
    -- (Not one per kernel: that would require 8x3=24 window generators.)
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
    -- Kernel 0 -- per-channel pipelined dot products (zero bias each)
    -- ================================================================
    dot_k0_c0 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K0_CH0(0),  w1 => K0_CH0(1),  w2 => K0_CH0(2),
            w3 => K0_CH0(3),  w4 => K0_CH0(4),  w5 => K0_CH0(5),
            w6 => K0_CH0(6),  w7 => K0_CH0(7),  w8 => K0_CH0(8),
            bias      => ZERO32,
            valid_out => valid_dot,     -- drives all eight final summation gates
            y         => y0_c0
        );

    dot_k0_c1 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K0_CH1(0),  w1 => K0_CH1(1),  w2 => K0_CH1(2),
            w3 => K0_CH1(3),  w4 => K0_CH1(4),  w5 => K0_CH1(5),
            w6 => K0_CH1(6),  w7 => K0_CH1(7),  w8 => K0_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y0_c1
        );

    dot_k0_c2 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K0_CH2(0),  w1 => K0_CH2(1),  w2 => K0_CH2(2),
            w3 => K0_CH2(3),  w4 => K0_CH2(4),  w5 => K0_CH2(5),
            w6 => K0_CH2(6),  w7 => K0_CH2(7),  w8 => K0_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y0_c2
        );

    -- ================================================================
    -- Kernel 1 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as kernel 0 above.
    -- ================================================================
    dot_k1_c0 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K1_CH0(0),  w1 => K1_CH0(1),  w2 => K1_CH0(2),
            w3 => K1_CH0(3),  w4 => K1_CH0(4),  w5 => K1_CH0(5),
            w6 => K1_CH0(6),  w7 => K1_CH0(7),  w8 => K1_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y1_c0
        );

    dot_k1_c1 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K1_CH1(0),  w1 => K1_CH1(1),  w2 => K1_CH1(2),
            w3 => K1_CH1(3),  w4 => K1_CH1(4),  w5 => K1_CH1(5),
            w6 => K1_CH1(6),  w7 => K1_CH1(7),  w8 => K1_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y1_c1
        );

    dot_k1_c2 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K1_CH2(0),  w1 => K1_CH2(1),  w2 => K1_CH2(2),
            w3 => K1_CH2(3),  w4 => K1_CH2(4),  w5 => K1_CH2(5),
            w6 => K1_CH2(6),  w7 => K1_CH2(7),  w8 => K1_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y1_c2
        );

    -- ================================================================
    -- Kernel 2 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as kernel 0 above.
    -- ================================================================
    dot_k2_c0 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K2_CH0(0),  w1 => K2_CH0(1),  w2 => K2_CH0(2),
            w3 => K2_CH0(3),  w4 => K2_CH0(4),  w5 => K2_CH0(5),
            w6 => K2_CH0(6),  w7 => K2_CH0(7),  w8 => K2_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y2_c0
        );

    dot_k2_c1 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K2_CH1(0),  w1 => K2_CH1(1),  w2 => K2_CH1(2),
            w3 => K2_CH1(3),  w4 => K2_CH1(4),  w5 => K2_CH1(5),
            w6 => K2_CH1(6),  w7 => K2_CH1(7),  w8 => K2_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y2_c1
        );

    dot_k2_c2 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K2_CH2(0),  w1 => K2_CH2(1),  w2 => K2_CH2(2),
            w3 => K2_CH2(3),  w4 => K2_CH2(4),  w5 => K2_CH2(5),
            w6 => K2_CH2(6),  w7 => K2_CH2(7),  w8 => K2_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y2_c2
        );

    -- ================================================================
    -- Kernel 3 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as kernel 0 above.
    -- ================================================================
    dot_k3_c0 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K3_CH0(0),  w1 => K3_CH0(1),  w2 => K3_CH0(2),
            w3 => K3_CH0(3),  w4 => K3_CH0(4),  w5 => K3_CH0(5),
            w6 => K3_CH0(6),  w7 => K3_CH0(7),  w8 => K3_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y3_c0
        );

    dot_k3_c1 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K3_CH1(0),  w1 => K3_CH1(1),  w2 => K3_CH1(2),
            w3 => K3_CH1(3),  w4 => K3_CH1(4),  w5 => K3_CH1(5),
            w6 => K3_CH1(6),  w7 => K3_CH1(7),  w8 => K3_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y3_c1
        );

    dot_k3_c2 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K3_CH2(0),  w1 => K3_CH2(1),  w2 => K3_CH2(2),
            w3 => K3_CH2(3),  w4 => K3_CH2(4),  w5 => K3_CH2(5),
            w6 => K3_CH2(6),  w7 => K3_CH2(7),  w8 => K3_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y3_c2
        );

    -- ================================================================
    -- Kernel 4 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as kernel 0 above.
    -- ================================================================
    dot_k4_c0 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K4_CH0(0),  w1 => K4_CH0(1),  w2 => K4_CH0(2),
            w3 => K4_CH0(3),  w4 => K4_CH0(4),  w5 => K4_CH0(5),
            w6 => K4_CH0(6),  w7 => K4_CH0(7),  w8 => K4_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y4_c0
        );

    dot_k4_c1 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K4_CH1(0),  w1 => K4_CH1(1),  w2 => K4_CH1(2),
            w3 => K4_CH1(3),  w4 => K4_CH1(4),  w5 => K4_CH1(5),
            w6 => K4_CH1(6),  w7 => K4_CH1(7),  w8 => K4_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y4_c1
        );

    dot_k4_c2 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K4_CH2(0),  w1 => K4_CH2(1),  w2 => K4_CH2(2),
            w3 => K4_CH2(3),  w4 => K4_CH2(4),  w5 => K4_CH2(5),
            w6 => K4_CH2(6),  w7 => K4_CH2(7),  w8 => K4_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y4_c2
        );

    -- ================================================================
    -- Kernel 5 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as kernel 0 above.
    -- ================================================================
    dot_k5_c0 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K5_CH0(0),  w1 => K5_CH0(1),  w2 => K5_CH0(2),
            w3 => K5_CH0(3),  w4 => K5_CH0(4),  w5 => K5_CH0(5),
            w6 => K5_CH0(6),  w7 => K5_CH0(7),  w8 => K5_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y5_c0
        );

    dot_k5_c1 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K5_CH1(0),  w1 => K5_CH1(1),  w2 => K5_CH1(2),
            w3 => K5_CH1(3),  w4 => K5_CH1(4),  w5 => K5_CH1(5),
            w6 => K5_CH1(6),  w7 => K5_CH1(7),  w8 => K5_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y5_c1
        );

    dot_k5_c2 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K5_CH2(0),  w1 => K5_CH2(1),  w2 => K5_CH2(2),
            w3 => K5_CH2(3),  w4 => K5_CH2(4),  w5 => K5_CH2(5),
            w6 => K5_CH2(6),  w7 => K5_CH2(7),  w8 => K5_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y5_c2
        );

    -- ================================================================
    -- Kernel 6 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as kernel 0 above.
    -- ================================================================
    dot_k6_c0 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K6_CH0(0),  w1 => K6_CH0(1),  w2 => K6_CH0(2),
            w3 => K6_CH0(3),  w4 => K6_CH0(4),  w5 => K6_CH0(5),
            w6 => K6_CH0(6),  w7 => K6_CH0(7),  w8 => K6_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y6_c0
        );

    dot_k6_c1 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K6_CH1(0),  w1 => K6_CH1(1),  w2 => K6_CH1(2),
            w3 => K6_CH1(3),  w4 => K6_CH1(4),  w5 => K6_CH1(5),
            w6 => K6_CH1(6),  w7 => K6_CH1(7),  w8 => K6_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y6_c1
        );

    dot_k6_c2 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K6_CH2(0),  w1 => K6_CH2(1),  w2 => K6_CH2(2),
            w3 => K6_CH2(3),  w4 => K6_CH2(4),  w5 => K6_CH2(5),
            w6 => K6_CH2(6),  w7 => K6_CH2(7),  w8 => K6_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y6_c2
        );

    -- ================================================================
    -- Kernel 7 -- per-channel pipelined dot products (zero bias each)
    -- Reads the SAME c0_p*/c1_p*/c2_p* windows as kernel 0 above.
    -- ================================================================
    dot_k7_c0 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c0,
            p0 => c0_p0,  p1 => c0_p1,  p2 => c0_p2,
            p3 => c0_p3,  p4 => c0_p4,  p5 => c0_p5,
            p6 => c0_p6,  p7 => c0_p7,  p8 => c0_p8,
            w0 => K7_CH0(0),  w1 => K7_CH0(1),  w2 => K7_CH0(2),
            w3 => K7_CH0(3),  w4 => K7_CH0(4),  w5 => K7_CH0(5),
            w6 => K7_CH0(6),  w7 => K7_CH0(7),  w8 => K7_CH0(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y7_c0
        );

    dot_k7_c1 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c1,
            p0 => c1_p0,  p1 => c1_p1,  p2 => c1_p2,
            p3 => c1_p3,  p4 => c1_p4,  p5 => c1_p5,
            p6 => c1_p6,  p7 => c1_p7,  p8 => c1_p8,
            w0 => K7_CH1(0),  w1 => K7_CH1(1),  w2 => K7_CH1(2),
            w3 => K7_CH1(3),  w4 => K7_CH1(4),  w5 => K7_CH1(5),
            w6 => K7_CH1(6),  w7 => K7_CH1(7),  w8 => K7_CH1(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y7_c1
        );

    dot_k7_c2 : entity work.conv3x3_dot_pipelined
        port map (
            clk => clk,  rst => rst,  valid_in => win_valid_c2,
            p0 => c2_p0,  p1 => c2_p1,  p2 => c2_p2,
            p3 => c2_p3,  p4 => c2_p4,  p5 => c2_p5,
            p6 => c2_p6,  p7 => c2_p7,  p8 => c2_p8,
            w0 => K7_CH2(0),  w1 => K7_CH2(1),  w2 => K7_CH2(2),
            w3 => K7_CH2(3),  w4 => K7_CH2(4),  w5 => K7_CH2(5),
            w6 => K7_CH2(6),  w7 => K7_CH2(7),  w8 => K7_CH2(8),
            bias      => ZERO32,
            valid_out => open,
            y         => y7_c2
        );

    -- ================================================================
    -- Final registered summation stage, one per kernel.
    --
    -- Adds each kernel's three per-channel dot-product results and its
    -- own bias in one registered step, gated by the shared valid_dot.
    -- Same structure as stream_conv3x3_3chan_cell.vhd's single final_sum
    -- and stream_conv3x3_3chan_4out_cell.vhd's four final_sum registers,
    -- replicated 8x -- this is why the cell's overall latency (4 cycles)
    -- is unchanged from the 1-output and 4-output cells.
    -- ================================================================
    final_sum : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                y0_r    <= (others => '0');
                y1_r    <= (others => '0');
                y2_r    <= (others => '0');
                y3_r    <= (others => '0');
                y4_r    <= (others => '0');
                y5_r    <= (others => '0');
                y6_r    <= (others => '0');
                y7_r    <= (others => '0');
                valid_r <= '0';
            else
                y0_r    <= K0_BIAS32 + y0_c0 + y0_c1 + y0_c2;
                y1_r    <= K1_BIAS32 + y1_c0 + y1_c1 + y1_c2;
                y2_r    <= K2_BIAS32 + y2_c0 + y2_c1 + y2_c2;
                y3_r    <= K3_BIAS32 + y3_c0 + y3_c1 + y3_c2;
                y4_r    <= K4_BIAS32 + y4_c0 + y4_c1 + y4_c2;
                y5_r    <= K5_BIAS32 + y5_c0 + y5_c1 + y5_c2;
                y6_r    <= K6_BIAS32 + y6_c0 + y6_c1 + y6_c2;
                y7_r    <= K7_BIAS32 + y7_c0 + y7_c1 + y7_c2;
                valid_r <= valid_dot;
            end if;
        end if;
    end process final_sum;

    valid_out <= valid_r;
    y0 <= y0_r;
    y1 <= y1_r;
    y2 <= y2_r;
    y3 <= y3_r;
    y4 <= y4_r;
    y5 <= y5_r;
    y6 <= y6_r;
    y7 <= y7_r;

end architecture rtl;
