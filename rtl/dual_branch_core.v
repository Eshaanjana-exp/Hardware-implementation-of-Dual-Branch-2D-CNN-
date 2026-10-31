`timescale 1ns/1ps
`include "weights_data.vh"   // generated from mem/*.mem by python/gen_weights_vh.py
// Dual-Branch PQP/SZW CNN core (Verilog-2001, synthesizable).
//
//   IMGxIMG UINT4  --+--> window5 (shared line buffers) --> 3x3 sub-window --> PQP conv1 (1->8, 3x3)
//                    |                                                         |
//                    |                                  window3 (8ch line buffers) --> PQP conv2 (8->8, 3x3)
//                    +--> PQP conv5 (1->8, 5x5) ---------------------------------+
//                                                                                 v
//                                              Fused Add-Pool (sat-add + 2x2 max, stride 2)
//                                                                                 v
//                                                      8 x (IMG-4)/2 x (IMG-4)/2 UINT4
//
// Pixels are processed one per "tick" (self-timed FSM, no fixed period).  Both branches
// complete for the same tick (output pixel (y,x) <- input pixel (y+4,x+4)), so no alignment
// FIFO is needed.  Memories: image  = 8 pixels per 32-bit word, nibble i&7 of word i>>3;
//                            output = one 32-bit word per pooled pixel (nibble f = filter f),
//                                     word address = y*((IMG-4)/2) + x.
module dual_branch_core #(
    parameter IMG     = 512,
    parameter IAW     = 15,            // = ceil(log2(ceil(IMG*IMG/8)))   (image word address bits)
    parameter OAW     = 16,            // = ceil(log2(((IMG-4)/2)^2))      (output word address bits)
    parameter [2303:0] W2_DATA = `PQP_W2,
    parameter [287:0] W1_DATA = `PQP_W1,
    parameter [799:0] W5_DATA = `PQP_W5,
    parameter [63:0]  B1_DATA = `PQP_B1,
    parameter [63:0]  B2_DATA = `PQP_B2,
    parameter [63:0]  B5_DATA = `PQP_B5,
    parameter Q1      = 0,
    parameter Q2      = 0,
    parameter Q5      = 0
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,          // pulse
    output wire        busy,
    output reg         done,           // level, set at end of run, cleared by next start
    output wire        ready,          // SZW lists built
    output wire [23:0] nsteps_info,    // {n5[7:0], n2[7:0], n1[7:0]}
    // image memory (synchronous read, 1 clk)
    output wire [IAW-1:0] img_addr,
    input  wire [31:0]    img_q,
    // output memory write port
    output wire           out_we,
    output wire [OAW-1:0] out_addr,
    output wire [31:0]    out_data
);
    function integer clog2; input integer v; integer i; begin
        clog2 = 1; for (i = 1; (1 << i) < v; i = i + 1) clog2 = i + 1; end
    endfunction

    localparam NPIX = IMG*IMG;
    localparam W1   = IMG-2;
    localparam W2   = IMG-4;
    localparam HP   = W2/2;
    localparam PXW  = clog2(IMG);
    localparam W1AW = clog2(W1);

    // ---------------------------------------------------------------- SZW lists
    wire [3:0]  l1_n;  wire [6:0] l2_n;  wire [4:0] l5_n;
    wire [3:0]  l1_ra; wire [6:0] l2_ra; wire [4:0] l5_ra;
    wire [3+32:0] l1_q; wire [6+32:0] l2_q; wire [4+32:0] l5_q;
    wire r1, r2, r5;

    szw_list #(.NF(8), .NCH(1), .KK(9),  .IDXW(4), .WDATA(W1_DATA)) u_l1 (
        .clk(clk), .rst_n(rst_n), .raddr(l1_ra), .rq(l1_q), .nsteps(l1_n), .ready(r1));
    szw_list #(.NF(8), .NCH(8), .KK(9),  .IDXW(7), .WDATA(W2_DATA)) u_l2 (
        .clk(clk), .rst_n(rst_n), .raddr(l2_ra), .rq(l2_q), .nsteps(l2_n), .ready(r2));
    szw_list #(.NF(8), .NCH(1), .KK(25), .IDXW(5), .WDATA(W5_DATA)) u_l5 (
        .clk(clk), .rst_n(rst_n), .raddr(l5_ra), .rq(l5_q), .nsteps(l5_n), .ready(r5));

    assign ready       = r1 & r2 & r5;
    assign nsteps_info = {3'b000, l5_n, 1'b0, l2_n, 4'b0000, l1_n};

    // ---------------------------------------------------------------- control FSM
    localparam S_IDLE = 4'd0, S_F0 = 4'd1, S_SHIFT = 4'd2, S_W5 = 4'd3, S_C1 = 4'd4,
               S_SHIFT2 = 4'd5, S_W2 = 4'd6, S_C2 = 4'd7, S_FUSE = 4'd8, S_NEXT = 4'd9;

    reg [3:0]      state;
    reg [IAW+2:0]  pidx;
    reg [PXW-1:0]  px, py;
    reg            d2f, d5f;

    assign busy = (state != S_IDLE);
    wire go = start & ready & ~busy;

    wire v3 = (py >= 2) && (px >= 2);
    wire v5 = (py >= 4) && (px >= 4);

    wire done1, done2, done5, wg2_valid;
    wire en5      = (state == S_SHIFT);
    wire en2      = (state == S_SHIFT2);
    wire fuse_en  = (state == S_FUSE);
    wire start1   = (state == S_W5) & v3;
    wire start5   = (state == S_W5) & v5;
    wire start2   = (state == S_W2) & wg2_valid;
    wire c2_ok    = (d2f | done2) & (d5f | done5);

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; pidx <= 0; px <= 0; py <= 0; done <= 1'b0; d2f <= 1'b0; d5f <= 1'b0;
        end else begin
            if (done2) d2f <= 1'b1;
            if (done5) d5f <= 1'b1;
            case (state)
                S_IDLE:   if (go) begin
                              state <= S_F0; pidx <= 0; px <= 0; py <= 0;
                              done <= 1'b0; d2f <= 1'b0; d5f <= 1'b0;
                          end
                S_F0:     state <= S_SHIFT;                    // image RAM address settles
                S_SHIFT:  state <= S_W5;                       // window5 captures pixel
                S_W5:     begin d2f <= 1'b0; d5f <= 1'b0; state <= v3 ? S_C1 : S_NEXT; end
                S_C1:     if (done1) state <= S_SHIFT2;
                S_SHIFT2: state <= S_W2;
                S_W2:     state <= wg2_valid ? S_C2 : S_NEXT;
                S_C2:     if (c2_ok) state <= S_FUSE;
                S_FUSE:   state <= S_NEXT;
                S_NEXT:   begin
                              if (pidx == NPIX-1) begin
                                  state <= S_IDLE; done <= 1'b1;
                              end else begin
                                  state <= S_F0; pidx <= pidx + 1'b1;
                                  if (px == IMG-1) begin px <= 0; py <= py + 1'b1; end
                                  else px <= px + 1'b1;
                              end
                          end
                default:  state <= S_IDLE;
            endcase
        end
    end

    // ---------------------------------------------------------------- input stage
    assign img_addr = pidx[IAW+2:3];
    wire [3:0] pix_in = img_q[pidx[2:0]*4 +: 4];

    wire [5*5*4-1:0] win5;
    wire [3*3*4-1:0] win3;
    window_gen #(.K(5), .DW(4), .W(IMG), .AW(PXW)) u_wg5 (
        .clk(clk), .rst_n(rst_n), .clr(go), .en(en5), .din(pix_in), .win(win5), .valid());

    genvar gi, gj;
    generate
        for (gi = 0; gi < 3; gi = gi + 1) begin : g3r
            for (gj = 0; gj < 3; gj = gj + 1) begin : g3c
                assign win3[(gi*3+gj)*4 +: 4] = win5[((gi+2)*5+(gj+2))*4 +: 4];
            end
        end
    endgenerate

    // ---------------------------------------------------------------- conv1 (Branch A)
    wire [3:0]  p1_idx;  wire [31:0] y1;
    pqp_engine #(.NF(8), .IDXW(4), .QSHIFT(Q1), .BDATA(B1_DATA)) u_e1 (
        .clk(clk), .rst_n(rst_n), .start(start1), .nsteps(l1_n),
        .list_raddr(l1_ra), .list_q(l1_q), .pix_idx(p1_idx),
        .pix_val(win3[p1_idx*4 +: 4]), .done(done1), .y(y1));

    // ---------------------------------------------------------------- conv2 (Branch A)
    wire [3*3*32-1:0] win2;
    window_gen #(.K(3), .DW(32), .W(W1), .AW(W1AW)) u_wg2 (
        .clk(clk), .rst_n(rst_n), .clr(go), .en(en2), .din(y1), .win(win2), .valid(wg2_valid));

    wire [6:0]  p2_idx;  wire [31:0] y2;
    pqp_engine #(.NF(8), .IDXW(7), .QSHIFT(Q2), .BDATA(B2_DATA)) u_e2 (
        .clk(clk), .rst_n(rst_n), .start(start2), .nsteps(l2_n),
        .list_raddr(l2_ra), .list_q(l2_q), .pix_idx(p2_idx),
        .pix_val(win2[p2_idx*4 +: 4]), .done(done2), .y(y2));

    // ---------------------------------------------------------------- conv5 (Branch B)
    wire [4:0]  p5_idx;  wire [31:0] y5;
    pqp_engine #(.NF(8), .IDXW(5), .QSHIFT(Q5), .BDATA(B5_DATA)) u_e5 (
        .clk(clk), .rst_n(rst_n), .start(start5), .nsteps(l5_n),
        .list_raddr(l5_ra), .list_q(l5_q), .pix_idx(p5_idx),
        .pix_val(win5[p5_idx*4 +: 4]), .done(done5), .y(y5));

    // ---------------------------------------------------------------- fused add + pool
    fused_add_pool #(.NF(8), .WIN(W2), .OAW(OAW)) u_fap (
        .clk(clk), .rst_n(rst_n), .clr(go), .en(fuse_en), .a(y2), .b(y5),
        .ovalid(out_we), .oaddr(out_addr), .odata(out_data));
endmodule
