`timescale 1ns/1ps
// AXI4-Lite wrapper around dual_branch_core for PYNQ-Z2 (PS M_AXI_GP0 -> this slave).
// Contains the image RAM and output RAM (inferred BRAM), so no AXI-BRAM-controller / DMA needed.
//
// Address map (byte offsets, 512 KiB window):
//   0x00000 CTRL    W : bit0 = start (self-clearing pulse)
//   0x00004 STATUS  R : bit0 busy, bit1 done (sticky until next start), bit2 ready (SZW lists built)
//   0x00008 CYCLES  R : clock cycles of the last / running frame
//   0x0000C NSTEPS  R : {8'b0, n5[7:0], n2[7:0], n1[7:0]}  (steps kept by SZW)
//   0x00010 INFO    R : IMG
//   0x20000-0x3FFFF   IMAGE  (write-only; 32768 words; pixel i = nibble (i&7) of word i>>3; reads return 0)
//   0x40000-0x7FFFF   OUTPUT (read-only; word y*((IMG-4)/2)+x ; nibble f = filter f)
module pqp_axi_top #(
    parameter IMG     = 512,
    parameter Q1      = 0,
    parameter Q2      = 0,
    parameter Q5      = 0
)(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET s_axi_aresetn" *)
    input  wire        s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        s_axi_aresetn,
    input  wire [18:0] s_axi_awaddr,
    input  wire [2:0]  s_axi_awprot,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [18:0] s_axi_araddr,
    input  wire [2:0]  s_axi_arprot,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready
);
    function integer clog2; input integer v; integer i; begin
        clog2 = 1; for (i = 1; (1 << i) < v; i = i + 1) clog2 = i + 1; end
    endfunction
    localparam NPIX  = IMG*IMG;
    localparam NW    = (NPIX+7)/8;
    localparam HP    = (IMG-4)/2;
    localparam NO    = HP*HP;
    localparam IAW   = clog2(NW);
    localparam OAW   = clog2(NO);

    wire clk   = s_axi_aclk;
    wire rst_n = s_axi_aresetn;

    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    // ------------------------------------------------------------------ write channel
    reg         aw_got, w_got;
    reg [18:0]  waddr_r;
    reg [31:0]  wdata_r;
    assign s_axi_awready = ~aw_got & ~s_axi_bvalid;
    assign s_axi_wready  = ~w_got  & ~s_axi_bvalid;
    wire wr_fire = aw_got & w_got & ~s_axi_bvalid;

    wire wr_is_reg = (waddr_r[18:17] == 2'b00);
    wire wr_is_img = (waddr_r[18:17] == 2'b01);
    wire core_start = wr_fire & wr_is_reg & (waddr_r[5:2] == 4'd0) & wdata_r[0];

    always @(posedge clk) begin
        if (!rst_n) begin
            aw_got <= 1'b0; w_got <= 1'b0; s_axi_bvalid <= 1'b0;
        end else begin
            if (s_axi_awvalid & s_axi_awready) begin aw_got <= 1'b1; waddr_r <= s_axi_awaddr; end
            if (s_axi_wvalid  & s_axi_wready)  begin w_got  <= 1'b1; wdata_r <= s_axi_wdata;  end
            if (wr_fire) begin aw_got <= 1'b0; w_got <= 1'b0; s_axi_bvalid <= 1'b1; end
            if (s_axi_bvalid & s_axi_bready) s_axi_bvalid <= 1'b0;
        end
    end

    // ------------------------------------------------------------------ image RAM (A: AXI write, B: core read)
    reg [31:0] img_mem [0:NW-1];
    wire          img_we    = wr_fire & wr_is_img;
    wire [IAW-1:0] img_waddr = waddr_r[IAW+1:2];
    wire [IAW-1:0] core_img_addr;
    reg  [31:0]    core_img_q;
    always @(posedge clk) if (img_we) img_mem[img_waddr] <= wdata_r;
    always @(posedge clk) core_img_q <= img_mem[core_img_addr];

    // ------------------------------------------------------------------ core
    wire        busy, done, ready;
    wire [23:0] nsteps;
    wire          out_we;
    wire [OAW-1:0] out_addr;
    wire [31:0]    out_data;

    dual_branch_core #(.IMG(IMG), .IAW(IAW), .OAW(OAW),
        .Q1(Q1), .Q2(Q2), .Q5(Q5)) u_core (
        .clk(clk), .rst_n(rst_n), .start(core_start), .busy(busy), .done(done), .ready(ready),
        .nsteps_info(nsteps), .img_addr(core_img_addr), .img_q(core_img_q),
        .out_we(out_we), .out_addr(out_addr), .out_data(out_data));

    reg [31:0] cycles;
    always @(posedge clk) begin
        if (!rst_n) cycles <= 32'd0;
        else if (core_start) cycles <= 32'd0;
        else if (busy) cycles <= cycles + 1'b1;
    end

    // ------------------------------------------------------------------ output RAM (A: core write, B: AXI read)
    reg [31:0] out_mem [0:NO-1];
    reg [31:0] out_q;
    reg [OAW-1:0] out_raddr;
    always @(posedge clk) if (out_we) out_mem[out_addr] <= out_data;
    always @(posedge clk) out_q <= out_mem[out_raddr];

    // ------------------------------------------------------------------ read channel
    reg [1:0]  rst_st;                  // 0 idle, 1 address settle, 2 data valid
    reg [18:0] raddr_r;
    assign s_axi_arready = (rst_st == 2'd0) & ~s_axi_rvalid;

    reg [31:0] reg_rd;
    always @* begin
        case (raddr_r[5:2])
            4'd1:    reg_rd = {29'd0, ready, done, busy};
            4'd2:    reg_rd = cycles;
            4'd3:    reg_rd = {8'd0, nsteps};
            4'd4:    reg_rd = IMG;
            default: reg_rd = 32'd0;
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            rst_st <= 2'd0; s_axi_rvalid <= 1'b0; s_axi_rdata <= 32'd0;
        end else begin
            case (rst_st)
                2'd0: if (s_axi_arvalid & s_axi_arready) begin
                          raddr_r <= s_axi_araddr; rst_st <= 2'd1;
                          out_raddr <= s_axi_araddr[OAW+1:2];
                      end
                2'd1: rst_st <= 2'd2;                       // out_q captures out_mem[out_raddr]
                2'd2: begin
                          s_axi_rvalid <= 1'b1;
                          s_axi_rdata  <= raddr_r[18] ? out_q :
                                          (raddr_r[18:17] == 2'b00) ? reg_rd : 32'd0;
                          rst_st <= 2'd3;
                      end
                default: if (s_axi_rvalid & s_axi_rready) begin
                              s_axi_rvalid <= 1'b0; rst_st <= 2'd0;
                         end
            endcase
        end
    end
endmodule
