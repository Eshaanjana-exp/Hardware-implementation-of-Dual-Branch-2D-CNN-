`timescale 1ns/1ps
// End-to-end testbench of seq_axi_top through its AXI4-Lite port (what the PYNQ PS does):
// load image words, wait ready, start, poll STATUS, read all output words, compare with golden.
module tb_seq_axi;
    parameter IMG = 512;
    function integer clog2; input integer v; integer i; begin
        clog2 = 1; for (i = 1; (1 << i) < v; i = i + 1) clog2 = i + 1; end
    endfunction
    localparam NPIX = IMG*IMG;
    localparam NW   = (NPIX+7)/8;
    localparam HP   = (IMG-4)/2;

    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;

    reg  [18:0] awaddr = 0; reg awvalid = 0; wire awready;
    reg  [31:0] wdata = 0;  reg wvalid = 0;  wire wready;
    wire [1:0]  bresp; wire bvalid; reg bready = 0;
    reg  [18:0] araddr = 0; reg arvalid = 0; wire arready;
    wire [31:0] rdata; wire [1:0] rresp; wire rvalid; reg rready = 0;

    seq_axi_top #(.IMG(IMG)) dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rstn),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'b0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arprot(3'b0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready));

    task axi_write; input [18:0] a; input [31:0] d; begin
        @(posedge clk); awaddr <= a; awvalid <= 1; wdata <= d; wvalid <= 1; bready <= 1;
        fork
            begin : aw wait (awready && awvalid); @(posedge clk); awvalid <= 0; end
            begin : w  wait (wready && wvalid);   @(posedge clk); wvalid  <= 0; end
        join
        wait (bvalid); @(posedge clk); bready <= 0;
    end endtask

    task axi_read; input [18:0] a; output [31:0] d; begin
        @(posedge clk); araddr <= a; arvalid <= 1; rready <= 1;
        wait (arready && arvalid); @(posedge clk); arvalid <= 0;
        wait (rvalid); d = rdata; @(posedge clk); rready <= 0;
    end endtask

    reg [3:0]  nib  [0:NPIX-1];
    reg [31:0] gold [0:HP*HP-1];
    reg [31:0] w, st, rd;
    integer i, errs;

    initial begin
        $readmemh("input_image.mem", nib);
        $readmemh("golden_pool_words.mem", gold);
        repeat (5) @(posedge clk); rstn = 1; repeat (2) @(posedge clk);

        st = 0;
        while (!st[2]) axi_read(19'h00004, st);                  // wait ready
        axi_read(19'h00010, rd); $display("INFO reg (IMG)  = %0d", rd);
        axi_read(19'h0000C, rd); $display("NSTEPS reg      = n1=%0d n2=%0d n5=%0d", rd[7:0], rd[15:8], rd[23:16]);

        for (i = 0; i < NW; i = i + 1) begin                      // load image
            w = 0;
            for (errs = 0; errs < 8; errs = errs + 1)
                if (i*8+errs < NPIX) w[errs*4 +: 4] = nib[i*8+errs];
            axi_write(19'h20000 + i*4, w);
        end
        axi_write(19'h00000, 32'd1);                              // start
        axi_read(19'h00004, st);
        $display("STATUS right after start = %b (busy=%b)", st[2:0], st[0]);
        while (!st[1]) axi_read(19'h00004, st);                   // poll done
        axi_read(19'h00008, rd); $display("CYCLES reg = %0d  (%0.1f ms @100MHz)", rd, rd/100000.0);

        errs = 0;
        for (i = 0; i < HP*HP; i = i + 1) begin
            axi_read(19'h40000 + i*4, rd);
            if (rd !== gold[i]) begin
                errs = errs + 1;
                if (errs <= 5) $display("MISMATCH word %0d: rtl=%h gold=%h", i, rd, gold[i]);
            end
        end
        axi_read(19'h20000, rd); if (rd !== 0) $display("note: image region read = %h", rd);
        if (errs == 0) $display("RESULT: PASS  AXI end-to-end, %0d output words bit-exact", HP*HP);
        else           $display("RESULT: FAIL  mismatches=%0d", errs);
        $finish;
    end
    initial begin #4000000000; $display("TIMEOUT"); $finish; end
endmodule
