`timescale 1ns/1ps
// Self-checking testbench for dual_branch_core.  Run from the directory that holds the
// .mem files + golden_pool_words.mem (see python/golden_model.py and python/run_tests.sh).
module tb_core;
    parameter IMG = 512;
    parameter Q1 = 0, Q2 = 0, Q5 = 0;
    parameter WRITE_BMP = 1;      // 1 = write all layer images as .bmp from this testbench (pure Verilog)
    parameter SCALE     = 1;      // integer enlargement of every BMP (nearest neighbour)

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

    dual_branch_core #(.IMG(IMG), .IAW(IAW), .OAW(OAW), .Q1(Q1), .Q2(Q2), .Q5(Q5)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done), .ready(ready),
        .nsteps_info(nsteps), .img_addr(img_addr), .img_q(img_q),
        .out_we(out_we), .out_addr(out_addr), .out_data(out_data));

    reg [3:0]  nib  [0:NPIX-1];
    reg [31:0] imem [0:NW-1];
    reg [31:0] omem [0:HP*HP-1];
    reg [31:0] gold [0:HP*HP-1];
    integer i, k, errs, cycles, writes;

    // ---- layer taps (simulation only): capture every engine result with its pixel position ----
    localparam N1 = IMG-2, N2 = IMG-4;
    reg [31:0] a1m [0:N1*N1-1];
    reg [31:0] a2m [0:N2*N2-1];
    reg [31:0] bm  [0:N2*N2-1];
    always @(posedge clk) begin
        if (dut.done1) a1m[(dut.py-2)*N1 + (dut.px-2)] <= dut.y1;
        if (dut.done2) a2m[(dut.py-4)*N2 + (dut.px-4)] <= dut.y2;
        if (dut.done5) bm [(dut.py-4)*N2 + (dut.px-4)] <= dut.y5;
    end


    // ======================= BMP writer (24-bit, grey = value*17) =======================
    // layer: 0 input | 1 A1 | 2 A2 | 3 B | 4 ADD | 5 POOL ;  f: filter 0..7, or 8 = all 8 tiled 4x2
    integer bfd;
    function integer lw; input integer layer; begin
        case (layer) 0: lw = IMG; 1: lw = N1; 5: lw = HP; default: lw = N2; endcase
    end endfunction
    function [3:0] px4; input integer layer; input integer f; input integer x; input integer y;
        reg [4:0] t; reg [31:0] wa, wb; begin
        case (layer)
            0: px4 = nib[y*IMG + x];
            1: px4 = a1m[y*N1 + x] >> (4*f);
            2: px4 = a2m[y*N2 + x] >> (4*f);
            3: px4 = bm [y*N2 + x] >> (4*f);
            4: begin wa = a2m[y*N2 + x] >> (4*f); wb = bm[y*N2 + x] >> (4*f);
                     t = wa[3:0] + wb[3:0]; px4 = (t > 15) ? 4'd15 : t[3:0]; end
            default: px4 = omem[y*HP + x] >> (4*f);
        endcase
    end endfunction
    task wr32; input integer v; begin
        $fwrite(bfd, "%c%c%c%c", v[7:0], v[15:8], v[23:16], v[31:24]);
    end endtask
    task write_bmp; input integer layer; input integer f;
        integer ww, cw, ch, ow, oh, rowb, xx, yy, sx, sy, tc, tr, ix, iy, k;
        reg [7:0] g; reg [3:0] v; reg [8*48-1:0] fname; begin
        ww = lw(layer);
        if (f == 8) begin cw = 4*ww + 6; ch = 2*ww + 2; end else begin cw = ww; ch = ww; end
        ow = cw*SCALE; oh = ch*SCALE; rowb = ((ow*3 + 3)/4)*4;
        case (layer)
            0: $sformat(fname, "bmp_00_input.bmp");
            1: if (f == 8) $sformat(fname, "bmp_A1_all.bmp");   else $sformat(fname, "bmp_A1_f%0d.bmp", f);
            2: if (f == 8) $sformat(fname, "bmp_A2_all.bmp");   else $sformat(fname, "bmp_A2_f%0d.bmp", f);
            3: if (f == 8) $sformat(fname, "bmp_B_all.bmp");    else $sformat(fname, "bmp_B_f%0d.bmp", f);
            4: if (f == 8) $sformat(fname, "bmp_ADD_all.bmp");  else $sformat(fname, "bmp_ADD_f%0d.bmp", f);
            default: if (f == 8) $sformat(fname, "bmp_POOL_all.bmp"); else $sformat(fname, "bmp_POOL_f%0d.bmp", f);
        endcase
        bfd = $fopen(fname, "wb");
        $fwrite(bfd, "BM"); wr32(54 + rowb*oh); wr32(0); wr32(54);
        wr32(40); wr32(ow); wr32(oh); $fwrite(bfd, "%c%c%c%c", 8'd1, 8'd0, 8'd24, 8'd0);
        wr32(0); wr32(rowb*oh); wr32(2835); wr32(2835); wr32(0); wr32(0);
        for (yy = oh-1; yy >= 0; yy = yy - 1) begin
            sy = yy / SCALE;
            for (xx = 0; xx < ow; xx = xx + 1) begin
                sx = xx / SCALE;
                if (f < 8) v = px4(layer, f, sx, sy);
                else begin
                    tc = sx / (ww+2); tr = sy / (ww+2); ix = sx % (ww+2); iy = sy % (ww+2);
                    if (tc < 4 && tr < 2 && ix < ww && iy < ww) v = px4(layer, tr*4 + tc, ix, iy);
                    else v = 4'd8;
                end
                g = v * 17;
                $fwrite(bfd, "%c%c%c", g, g, g);
            end
            for (k = 0; k < rowb - ow*3; k = k + 1) $fwrite(bfd, "%c", 8'h00);
        end
        $fclose(bfd);
    end endtask
    integer bl, bf;

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
        $display("SZW steps kept: conv1=%0d/9  conv2=%0d/72  conv5=%0d/25",
                 nsteps[7:0], nsteps[15:8], nsteps[23:16]);
        @(posedge clk); start = 1; @(posedge clk); start = 0;
        cycles = 0;
        while (!done) begin @(posedge clk); cycles = cycles + 1; if (cycles > 400000000) begin $display("TIMEOUT"); $finish; end end
        repeat (3) @(posedge clk);
        errs = 0;
        for (i = 0; i < HP*HP; i = i + 1) if (omem[i] !== gold[i]) begin
            errs = errs + 1;
            if (errs <= 5) $display("MISMATCH word %0d: rtl=%h gold=%h", i, omem[i], gold[i]);
        end
        $writememh("rtl_a1.mem", a1m);  $writememh("rtl_a2.mem", a2m);
        $writememh("rtl_b.mem", bm);    $writememh("rtl_pool_words.mem", omem);
        if (WRITE_BMP) begin
            write_bmp(0, 0);
            for (bl = 1; bl <= 5; bl = bl + 1) for (bf = 0; bf <= 8; bf = bf + 1) write_bmp(bl, bf);
            $display("BMP files written to the simulation working directory (bmp_*.bmp)");
        end
        $display("IMG=%0d cycles=%0d (%0d per input pixel) output writes=%0d (expected %0d)",
                 IMG, cycles, cycles/NPIX, writes, HP*HP);
        if (errs == 0 && writes == HP*HP) $display("RESULT: PASS  (all %0d pooled words bit-exact)", HP*HP);
        else $display("RESULT: FAIL  mismatches=%0d", errs);
        $finish;
    end
endmodule
