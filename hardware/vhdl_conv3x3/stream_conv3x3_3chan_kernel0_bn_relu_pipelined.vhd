-- stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd
-- Pipelined variant of stream_conv3x3_3chan_kernel0_bn_relu.vhd.
--
-- This is NOT a new convolution architecture and NOT a new numerical
-- scheme. It reuses the SAME existing, unmodified stream_conv3x3_3chan_cell
-- sub-component, the SAME first_conv_bn_relu_kernel0_pkg Q.16 fixed-point
-- constants (SCALE_FX, BIAS_FX), and produces IDENTICAL final output
-- values to the unpipelined design -- only the number of pipeline
-- registers between the raw convolution and the final output changes.
--
-- Motivation: hardware/vhdl_conv3x3/first_conv_bn_relu_kernel0_vhdl_summary.md
-- reported that the unpipelined design's single combinational
-- multiply -> add -> ReLU-compare stage did not meet the 100 MHz timing
-- target on Artix-7 200T (WNS = -0.609 ns, 42 failing endpoints), even
-- though the raw convolution alone comfortably did (+4.456 ns). This
-- module tests whether splitting that one combinational stage into two
-- registered stages (multiply in one cycle, bias-add + ReLU in the next)
-- recovers positive WNS.
--
-- ---- Architecture -----------------------------------------------------
--   stream_conv3x3_3chan_cell (existing, unmodified, kernel-0 folded
--     INT8 weights, bias=0)          -> raw_y (INT32), 4-cycle latency
--   Stage 1 (registered): product_fx_r  <= raw_y * SCALE_FX   (Q.16, exact)
--   Stage 2 (registered): biased_fx     = product_fx_r + BIAS_FX (Q.16)
--                          y_bn_relu_fx_r <= biased_fx if biased_fx > 0
--                                            else 0                 (ReLU)
--
-- ---- Fixed-point format -------------------------------------------------
-- Q.16 (signed, 16 fractional bits): real_value ~= fixed_value / 2^16.
-- SCALE_FX and BIAS_FX are the SAME compile-time constants from
-- first_conv_bn_relu_kernel0_pkg used by the unpipelined design -- no
-- change to the Python golden arithmetic. Because raw_y has 0 fractional
-- bits and SCALE_FX has 16, product_fx_r is automatically Q.16 with no
-- shift; BIAS_FX is already Q.16, so the Stage-2 add needs no shift either.
--
-- ---- Latency ------------------------------------------------------------
-- 6 clock cycles total: 4 cycles from stream_conv3x3_3chan_cell (unchanged)
-- + 2 cycles for the now-split multiply / bias-add+ReLU stages -- ONE
-- MORE cycle than the unpipelined design's 5-cycle latency, as expected
-- from adding one extra pipeline register.
--
-- ---- Scope ----------------------------------------------------------------
-- Kernel 0 ONLY (not all 32 output channels). Canonical 5x5x3 toy input,
-- valid (no-padding) 3x3 output region only -- same as every other VHDL
-- prototype in this repo. No pooling, no downstream layers, no full model
-- pipeline. Not board-tested, not a measured-speedup or measured-power
-- claim.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.first_conv_bn_relu_kernel0_pkg.all;

entity stream_conv3x3_3chan_kernel0_bn_relu_pipelined is
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

        -- Output: one Q.16 fixed-point result per valid window triplet,
        -- ReLU already applied, 6-cycle latency (one more than the
        -- unpipelined design).
        valid_out    : out std_logic;
        y_bn_relu_fx : out signed(47 downto 0)
    );
end entity stream_conv3x3_3chan_kernel0_bn_relu_pipelined;

architecture rtl of stream_conv3x3_3chan_kernel0_bn_relu_pipelined is

    constant ZERO32 : signed(31 downto 0) := (others => '0');

    -- Fixed-point constants from the package, widened to fixed VHDL widths
    -- -- IDENTICAL to the unpipelined design.
    constant SCALE_FX_SIGNED : signed(17 downto 0) := to_signed(SCALE_FX, 18);
    constant BIAS_FX_SIGNED  : signed(47 downto 0) := to_signed(BIAS_FX, 48);

    -- Raw INT32 convolution output from the reused, unmodified 1-output cell
    signal raw_y     : signed(31 downto 0);
    signal raw_valid : std_logic;

    -- ---- Stage 1: registered multiply ----
    signal product_fx_r : signed(47 downto 0) := (others => '0');
    signal valid_s1_r    : std_logic           := '0';

    -- ---- Stage 2: registered bias-add + ReLU ----
    signal biased_fx_comb : signed(47 downto 0);   -- combinational, within stage 2 only
    signal y_bn_relu_fx_r : signed(47 downto 0) := (others => '0');
    signal valid_s2_r      : std_logic           := '0';

begin

    -- ================================================================
    -- Raw convolution: EXISTING, UNMODIFIED stream_conv3x3_3chan_cell,
    -- fed kernel 0's FOLDED INT8 weights with bias=0 -- IDENTICAL to the
    -- unpipelined design's instantiation.
    -- ================================================================
    raw_conv : entity work.stream_conv3x3_3chan_cell
        generic map (IMG_WIDTH => IMG_WIDTH)
        port map (
            clk => clk,  rst => rst,  valid_in => valid_in,
            pixel_c0 => pixel_c0,  pixel_c1 => pixel_c1,  pixel_c2 => pixel_c2,
            c0_w0 => to_signed(K0_FOLDED_CH0_W(0), 8),
            c0_w1 => to_signed(K0_FOLDED_CH0_W(1), 8),
            c0_w2 => to_signed(K0_FOLDED_CH0_W(2), 8),
            c0_w3 => to_signed(K0_FOLDED_CH0_W(3), 8),
            c0_w4 => to_signed(K0_FOLDED_CH0_W(4), 8),
            c0_w5 => to_signed(K0_FOLDED_CH0_W(5), 8),
            c0_w6 => to_signed(K0_FOLDED_CH0_W(6), 8),
            c0_w7 => to_signed(K0_FOLDED_CH0_W(7), 8),
            c0_w8 => to_signed(K0_FOLDED_CH0_W(8), 8),
            c1_w0 => to_signed(K0_FOLDED_CH1_W(0), 8),
            c1_w1 => to_signed(K0_FOLDED_CH1_W(1), 8),
            c1_w2 => to_signed(K0_FOLDED_CH1_W(2), 8),
            c1_w3 => to_signed(K0_FOLDED_CH1_W(3), 8),
            c1_w4 => to_signed(K0_FOLDED_CH1_W(4), 8),
            c1_w5 => to_signed(K0_FOLDED_CH1_W(5), 8),
            c1_w6 => to_signed(K0_FOLDED_CH1_W(6), 8),
            c1_w7 => to_signed(K0_FOLDED_CH1_W(7), 8),
            c1_w8 => to_signed(K0_FOLDED_CH1_W(8), 8),
            c2_w0 => to_signed(K0_FOLDED_CH2_W(0), 8),
            c2_w1 => to_signed(K0_FOLDED_CH2_W(1), 8),
            c2_w2 => to_signed(K0_FOLDED_CH2_W(2), 8),
            c2_w3 => to_signed(K0_FOLDED_CH2_W(3), 8),
            c2_w4 => to_signed(K0_FOLDED_CH2_W(4), 8),
            c2_w5 => to_signed(K0_FOLDED_CH2_W(5), 8),
            c2_w6 => to_signed(K0_FOLDED_CH2_W(6), 8),
            c2_w7 => to_signed(K0_FOLDED_CH2_W(7), 8),
            c2_w8 => to_signed(K0_FOLDED_CH2_W(8), 8),
            bias      => ZERO32,
            valid_out => raw_valid,
            y         => raw_y
        );

    -- ================================================================
    -- Stage 1 (registered): product_fx = raw_y * SCALE_FX
    -- raw_y is Q.0 (exact integer); SCALE_FX is Q.16 -- product is
    -- automatically Q.16, no shift needed.
    -- ================================================================
    stage1_multiply : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                product_fx_r <= (others => '0');
                valid_s1_r    <= '0';
            else
                product_fx_r <= resize(raw_y * SCALE_FX_SIGNED, 48);
                valid_s1_r    <= raw_valid;
            end if;
        end if;
    end process stage1_multiply;

    -- ================================================================
    -- Stage 2 (registered): biased_fx = product_fx_r + BIAS_FX (Q.16,
    -- same format as product_fx_r, no shift), then ReLU clamp.
    -- ================================================================
    biased_fx_comb <= product_fx_r + BIAS_FX_SIGNED;

    stage2_bias_relu : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                y_bn_relu_fx_r <= (others => '0');
                valid_s2_r      <= '0';
            else
                if biased_fx_comb > 0 then
                    y_bn_relu_fx_r <= biased_fx_comb;
                else
                    y_bn_relu_fx_r <= (others => '0');
                end if;
                valid_s2_r <= valid_s1_r;
            end if;
        end if;
    end process stage2_bias_relu;

    valid_out    <= valid_s2_r;
    y_bn_relu_fx <= y_bn_relu_fx_r;

end architecture rtl;
