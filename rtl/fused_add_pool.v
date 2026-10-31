`timescale 1ns/1ps
// Fused Add-Pool unit.
// Input pairs (a = Branch-A conv2 pixel, b = Branch-B conv5 pixel) arrive in raster order
// of the WIN x WIN map, one pair per 'en' pulse.  The saturating add (UINT4, clamp 15) and
// the 2x2/stride-2 max-pool are performed in ONE streaming unit: the add map is never stored,
// only one half-row of column-pair maxima (WIN/2 words) is kept.
// A pooled pixel is emitted (ovalid pulse, oaddr = raster index) on every odd row / odd col.
module fused_add_pool #(
    parameter NF  = 8,
    parameter WIN = 508,          // must be even
    parameter OAW = 16
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 clr,
    input  wire                 en,
    input  wire [NF*4-1:0]      a,
    input  wire [NF*4-1:0]      b,
    output reg                  ovalid,
    output reg  [OAW-1:0]       oaddr,
    output reg  [NF*4-1:0]      odata
);
    function integer clog2; input integer v; integer i; begin
        clog2 = 1; for (i = 1; (1 << i) < v; i = i + 1) clog2 = i + 1; end
    endfunction
    localparam HW  = WIN/2;
    localparam HAW = clog2(HW);

    function [NF*4-1:0] satadd; input [NF*4-1:0] x; input [NF*4-1:0] y;
        integer i; reg [4:0] t; begin
            for (i = 0; i < NF; i = i + 1) begin
                t = {1'b0, x[i*4 +: 4]} + {1'b0, y[i*4 +: 4]};
                satadd[i*4 +: 4] = (t > 5'd15) ? 4'd15 : t[3:0];
            end
        end
    endfunction
    function [NF*4-1:0] maxn; input [NF*4-1:0] x; input [NF*4-1:0] y;
        integer i; begin
            for (i = 0; i < NF; i = i + 1)
                maxn[i*4 +: 4] = (x[i*4 +: 4] > y[i*4 +: 4]) ? x[i*4 +: 4] : y[i*4 +: 4];
        end
    endfunction

    reg              xodd, yodd;
    reg [HAW-1:0]    hx;
    reg [NF*4-1:0]   prev;
    reg [OAW-1:0]    ocnt;

    wire [NF*4-1:0]  s  = satadd(a, b);
    wire [NF*4-1:0]  m  = maxn(prev, s);
    wire [NF*4-1:0]  rb_q;
    wire             rb_we = en & xodd & ~yodd;

    linebuf #(.DW(NF*4), .DEPTH(HW), .AW(HAW)) rowbuf (
        .clk(clk), .addr(hx), .we(rb_we), .din(m), .dout(rb_q));

    always @(posedge clk) begin
        if (!rst_n || clr) begin
            xodd <= 1'b0; yodd <= 1'b0; hx <= 0; ocnt <= 0; ovalid <= 1'b0;
        end else begin
            ovalid <= 1'b0;
            if (en) begin
                if (!xodd) begin
                    prev <= s;
                    xodd <= 1'b1;
                end else begin
                    xodd <= 1'b0;
                    if (yodd) begin
                        ovalid <= 1'b1;
                        odata  <= maxn(rb_q, m);
                        oaddr  <= ocnt;
                        ocnt   <= ocnt + 1'b1;
                    end
                    if (hx == HW-1) begin hx <= 0; yodd <= ~yodd; end
                    else hx <= hx + 1'b1;
                end
            end
        end
    end
endmodule
