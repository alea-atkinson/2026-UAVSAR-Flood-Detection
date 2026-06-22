-- window3x3_stream.vhd
-- Sliding 3×3 window generator for streaming image convolution.
--
-- This is NOT a full Conv2d layer or full U-Net block.
-- It turns a row-major pixel stream into overlapping 3×3 pixel windows,
-- providing the pixel inputs that conv3x3_dot.vhd or conv3x3_dot_pipelined.vhd
-- expects.  Chain them together for a full streaming convolution cell.
--
-- How it works:
--   Three internal line buffers (shift-register rows) store the most recent
--   3 rows of pixels.  A round-robin write pointer cycles among them.
--   Once at least 2 complete rows have been stored AND at least 3 columns
--   have arrived in the current row, valid_out is asserted and p0..p8 hold
--   the 3×3 window centred on the current position.
--
--   Pixel layout (row-major, top-left = p0):
--       p0 p1 p2
--       p3 p4 p5
--       p6 p7 p8
--
--   Throughput : 1 window per clock once primed (back-pressure not supported).
--   Latency    : outputs appear on the same clock as the bottom-right pixel (p8).
--
-- Valid-output region (no padding):
--   Requires col_cnt >= 2 AND at least 2 complete rows already received.
--   For a 5×5 image with valid convolution, this produces 3×3 = 9 windows.
--
-- Implementation note:
--   Position (wptr, col_cnt) is the pixel being written THIS clock cycle.
--   All other positions in line_buf were written in earlier cycles and are
--   stable.  p8 is therefore taken from pixel_in directly rather than from
--   line_buf, avoiding a read-before-write hazard.
--
-- Limitations of this educational prototype:
--   - No padding (zero, reflect, etc.).
--   - IMG_WIDTH must be known at elaboration time.
--   - Assumes valid_in stays high for a complete row; mid-row pausing is not
--     supported and would corrupt the column counter.
--   - Not timed or synthesised.  For large IMG_WIDTH, line buffers should
--     be implemented with BRAM instead of flip-flops.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity window3x3_stream is
    generic (
        IMG_WIDTH : positive := 5       -- number of pixels per row
    );
    port (
        clk       : in  std_logic;
        rst       : in  std_logic;      -- synchronous, active high
        valid_in  : in  std_logic;      -- '1' when pixel_in holds a new pixel
        pixel_in  : in  signed(7 downto 0);

        valid_out : out std_logic;      -- '1' when p0..p8 hold a valid window
        -- 3×3 window, row-major (top-left = p0, bottom-right = p8)
        p0 : out signed(7 downto 0);
        p1 : out signed(7 downto 0);
        p2 : out signed(7 downto 0);
        p3 : out signed(7 downto 0);
        p4 : out signed(7 downto 0);
        p5 : out signed(7 downto 0);
        p6 : out signed(7 downto 0);
        p7 : out signed(7 downto 0);
        p8 : out signed(7 downto 0)
    );
end entity window3x3_stream;

architecture rtl of window3x3_stream is

    -- ----------------------------------------------------------------
    -- Line buffer: 3 rows × IMG_WIDTH columns of signed-8 pixels.
    -- Line buffers rotate: wptr is the row currently being written.
    -- ----------------------------------------------------------------
    type row_t     is array(0 to IMG_WIDTH - 1) of signed(7 downto 0);
    type line_buf_t is array(0 to 2) of row_t;

    signal line_buf : line_buf_t := (others => (others => (others => '0')));

    -- ----------------------------------------------------------------
    -- Datapath state
    -- ----------------------------------------------------------------
    signal col_cnt   : integer range 0 to IMG_WIDTH - 1 := 0;
    -- Count of fully received rows (saturates at 3; only need >=2)
    signal rows_done : integer range 0 to 3             := 0;
    -- Round-robin write pointer: which of the 3 line buffers is newest
    signal wptr      : integer range 0 to 2             := 0;

    -- ----------------------------------------------------------------
    -- Registered outputs (all updated on rising_edge when valid_in='1')
    -- ----------------------------------------------------------------
    signal valid_out_r                                         : std_logic := '0';
    signal p0r, p1r, p2r, p3r, p4r, p5r, p6r, p7r, p8r : signed(7 downto 0) :=
                                                              (others => '0');

begin

    -- ================================================================
    -- Main clocked process
    -- ================================================================
    main : process(clk)
        variable c       : integer range 0 to IMG_WIDTH - 1;
        variable oldest  : integer range 0 to 2;   -- oldest complete row
        variable mid_row : integer range 0 to 2;   -- middle row
    begin
        if rising_edge(clk) then

            if rst = '1' then
                col_cnt     <= 0;
                rows_done   <= 0;
                wptr        <= 0;
                valid_out_r <= '0';
                p0r <= (others => '0');  p1r <= (others => '0');
                p2r <= (others => '0');  p3r <= (others => '0');
                p4r <= (others => '0');  p5r <= (others => '0');
                p6r <= (others => '0');  p7r <= (others => '0');
                p8r <= (others => '0');
                for i in 0 to 2 loop
                    for j in 0 to IMG_WIDTH - 1 loop
                        line_buf(i)(j) <= (others => '0');
                    end loop;
                end loop;

            elsif valid_in = '1' then
                c := col_cnt;

                -- --------------------------------------------------------
                -- Write incoming pixel into the current line buffer row.
                -- This assignment takes effect after the process suspends;
                -- we must not read line_buf(wptr)(c) in the same delta.
                -- --------------------------------------------------------
                line_buf(wptr)(c) <= pixel_in;

                -- --------------------------------------------------------
                -- Round-robin pointer arithmetic.
                --   wptr  = newest row (being written right now)
                --   mid   = previous row
                --   oldest = row before that
                -- Using (wptr + k) mod 3 with k=2,1 avoids subtraction.
                -- --------------------------------------------------------
                oldest  := (wptr + 1) mod 3;
                mid_row := (wptr + 2) mod 3;

                -- --------------------------------------------------------
                -- Output a 3×3 window when enough context exists.
                -- All of oldest(0..c) and mid_row(0..c) were written in
                -- earlier clocks (safe to read).
                -- line_buf(wptr)(c-2) and line_buf(wptr)(c-1) were also
                -- written in earlier clocks.
                -- line_buf(wptr)(c) is being written NOW — use pixel_in.
                -- --------------------------------------------------------
                if rows_done >= 2 and c >= 2 then
                    valid_out_r <= '1';
                    -- Top row (oldest complete row)
                    p0r <= line_buf(oldest)(c - 2);
                    p1r <= line_buf(oldest)(c - 1);
                    p2r <= line_buf(oldest)(c);
                    -- Middle row
                    p3r <= line_buf(mid_row)(c - 2);
                    p4r <= line_buf(mid_row)(c - 1);
                    p5r <= line_buf(mid_row)(c);
                    -- Bottom row (newest; positions c-2 and c-1 already written)
                    p6r <= line_buf(wptr)(c - 2);
                    p7r <= line_buf(wptr)(c - 1);
                    p8r <= pixel_in;   -- bottom-right: use input directly
                else
                    valid_out_r <= '0';
                end if;

                -- --------------------------------------------------------
                -- Advance column; on row-end, rotate buffer and track rows.
                -- --------------------------------------------------------
                if c = IMG_WIDTH - 1 then
                    col_cnt <= 0;
                    wptr    <= (wptr + 1) mod 3;
                    if rows_done < 3 then
                        rows_done <= rows_done + 1;
                    end if;
                else
                    col_cnt <= c + 1;
                end if;

            else
                -- No valid pixel this cycle; hold output low
                valid_out_r <= '0';
            end if;

        end if;
    end process main;

    -- ================================================================
    -- Drive output ports from registered signals
    -- ================================================================
    valid_out <= valid_out_r;
    p0 <= p0r;  p1 <= p1r;  p2 <= p2r;
    p3 <= p3r;  p4 <= p4r;  p5 <= p5r;
    p6 <= p6r;  p7 <= p7r;  p8 <= p8r;

end architecture rtl;
