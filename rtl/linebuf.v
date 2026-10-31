`timescale 1ns/1ps
// Synchronous-read RAM used for line buffers (infers BRAM / distributed RAM).
// Read-first: dout shows the OLD contents when we=1 at the same address.
module linebuf #(
    parameter DW    = 4,
    parameter DEPTH = 512,
    parameter AW    = 9
)(
    input  wire          clk,
    input  wire [AW-1:0] addr,
    input  wire          we,
    input  wire [DW-1:0] din,
    output reg  [DW-1:0] dout
);
    reg [DW-1:0] mem [0:DEPTH-1];
    integer i;
    initial for (i = 0; i < DEPTH; i = i + 1) mem[i] = {DW{1'b0}};
    always @(posedge clk) begin
        if (we) mem[addr] <= din;
        dout <= mem[addr];
    end
endmodule
