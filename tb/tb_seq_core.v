`timescale 1ns/1ps
// Self-checking testbench for the BASELINE dual_branch_seq_core (same files / golden as tb_core).
// Run from the directory holding input_image.mem + golden_pool_words.mem.
module tb_seq_core;
    parameter IMG = 512;
    parameter Q1 = 0, Q2 = 0, Q5 = 0;

    function integer clog2; input integer v; integer i; begin
        clog2 = 1; for (i = 1; (1 << i) < v; i = i + 1) clog2 = i + 1; end
    endfunction
    localparam NPIX = IMG*IMG;
    localparam NW   = (NPIX+7)/8;
    localparam HP   = (IMG-4)/2;
    localparam IAW  = clog2(NW);
    localparam OAW  = clog2(HP*HP);

    reg clk = 0, rst_n = 0, start = 0;
    always #5 clk = ~clk;

    wire busy, done, ready;
    wire [23:0] nsteps;
    wire [IAW-1:0] img_addr;
    reg  [31:0] img_q;
    wire out_we;  wire [OAW-1:0] out_addr;  wire [31:0] out_data;

    dual_branch_seq_core #(.IMG(IMG), .IAW(IAW), .OAW(OAW), .Q1(Q1), .Q2(Q2), .Q5(Q5)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done), .ready(ready),
        .nsteps_info(nsteps), .img_addr(img_addr), .img_q(img_q),
        .out_we(out_we), .out_addr(out_addr), .out_data(out_data));

    reg [3:0]  nib  [0:NPIX-1];
    reg [31:0] imem [0:NW-1];
    reg [31:0] omem [0:HP*HP-1];
    reg [31:0] gold [0:HP*HP-1];
    integer i, errs, cycles, writes;

    always @(posedge clk) img_q <= imem[img_addr];
    always @(posedge clk) if (out_we) begin omem[out_addr] <= out_data; writes <= writes + 1; end

    initial begin
        $readmemh("input_image.mem", nib);
        $readmemh("golden_pool_words.mem", gold);
        for (i = 0; i < NW; i = i + 1) imem[i] = 32'd0;
        for (i = 0; i < NPIX; i = i + 1) imem[i>>3][(i&7)*4 +: 4] = nib[i];
        for (i = 0; i < HP*HP; i = i + 1) omem[i] = 32'hDEADBEEF;
        writes = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        wait (ready);
        @(posedge clk); start = 1; @(posedge clk); start = 0;
        cycles = 0;
        while (!done) begin @(posedge clk); cycles = cycles + 1; if (cycles > 2000000000) begin $display("TIMEOUT"); $finish; end end
        repeat (3) @(posedge clk);
        errs = 0;
        for (i = 0; i < HP*HP; i = i + 1) if (omem[i] !== gold[i]) begin
            errs = errs + 1;
            if (errs <= 5) $display("MISMATCH word %0d: rtl=%h gold=%h", i, omem[i], gold[i]);
        end
        $writememh("rtl_pool_words.mem", omem);
        $display("SEQ-BASELINE IMG=%0d cycles=%0d (%0.3f per input pixel) output writes=%0d (expected %0d)",
                 IMG, cycles, cycles*1.0/NPIX, writes, HP*HP);
        if (errs == 0 && writes == HP*HP) $display("RESULT: PASS  (all %0d pooled words bit-exact)", HP*HP);
        else $display("RESULT: FAIL  mismatches=%0d", errs);
        $finish;
    end
endmodule
