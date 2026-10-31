`timescale 1ns/1ps
// Streaming KxK window generator (line buffers + shift window).
// One pixel is pushed per 'en' pulse (raster order, row width W).
// After the pulse, 'win' holds the KxK neighbourhood whose bottom-right
// element is the pixel just pushed; element (r,c) is at bits [(r*K+c)*DW +: DW]
// (r=0 top/oldest row, c=0 left/oldest column).  'valid' = window fully inside image.
// Requirement: consecutive 'en' pulses are >= 3 clocks apart (line-buffer read latency).
module window_gen #(
    parameter K  = 5,
    parameter DW = 4,
    parameter W  = 512,
    parameter AW = 9
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                clr,      // start of frame
    input  wire                en,
    input  wire [DW-1:0]       din,
    output wire [K*K*DW-1:0]   win,
    output reg                 valid
);
    reg [AW-1:0]       col;
    reg [3:0]          row;              // saturates at K-1
    reg [K*K*DW-1:0]   win_r;
    assign win = win_r;

    wire [(K-1)*DW-1:0] rq;              // line buffer j (j rows above) read data
    wire [(K-1)*DW-1:0] wd;              // line buffer j write data

    genvar j;
    generate
        for (j = 0; j < K-1; j = j + 1) begin : lb
            if (j == 0) begin : first
                assign wd[0 +: DW] = din;
            end else begin : rest
                assign wd[j*DW +: DW] = rq[(j-1)*DW +: DW];
            end
            linebuf #(.DW(DW), .DEPTH(W), .AW(AW)) u (
                .clk(clk), .addr(col), .we(en), .din(wd[j*DW +: DW]), .dout(rq[j*DW +: DW]));
        end
    endgenerate

    integer r, c;
    initial win_r = {(K*K*DW){1'b0}};
    always @(posedge clk) begin
        if (!rst_n || clr) begin
            col   <= {AW{1'b0}};
            row   <= 4'd0;
            valid <= 1'b0;
        end else if (en) begin
            for (r = 0; r < K; r = r + 1) begin
                for (c = 0; c < K-1; c = c + 1)
                    win_r[(r*K+c)*DW +: DW] <= win_r[(r*K+c+1)*DW +: DW];
                if (r == K-1) win_r[(r*K+K-1)*DW +: DW] <= din;
                else          win_r[(r*K+K-1)*DW +: DW] <= rq[(K-2-r)*DW +: DW];
            end
            valid <= (row == K-1) && (col >= K-1);
            if (col == W-1) begin
                col <= {AW{1'b0}};
                if (row != K-1) row <= row + 4'd1;
            end else begin
                col <= col + 1'b1;
            end
        end
    end
endmodule
