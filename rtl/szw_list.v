`timescale 1ns/1ps
// SZW (Skip-Zero-Weight) step-list builder.
// The dense weights (parameter WDATA, same flat order as the .mem file: w[f*S + c*KK + k], 4-bit two's complement) is scanned ONCE
// after reset by a small FSM.  Every "step" s=(c,k) is one input-pixel position that is
// broadcast to all NF filters.  If all NF weights of a step are zero the step is dropped;
// otherwise {pixel_nibble_address, NF weights} is appended to a compact list RAM.
// The PQP engine then iterates ONLY over that list -> zero-weight steps cost no cycles
// and no multiplier activity.  nsteps = number of kept steps (>=1).
module szw_list #(
    parameter NF   = 8,
    parameter NCH  = 1,          // input channels
    parameter KK   = 9,          // taps per channel (K*K)
    parameter IDXW = 4,          // bits for step counter / pixel nibble address (S < 2**IDXW)
    parameter [NF*NCH*KK*4-1:0] WDATA = 0   // element i (= f*S + c*KK + k) at bits [4*i +: 4]
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [IDXW-1:0]         raddr,
    output reg  [IDXW+NF*4-1:0]    rq,        // {pix_nibble_addr, weights}, 1-clk read latency
    output wire [IDXW-1:0]         nsteps,
    output reg                     ready
);
    localparam S = NCH*KK;

    reg [3:0] wflat [0:NF*S-1];
    integer wi;
    initial for (wi = 0; wi < NF*S; wi = wi + 1) wflat[wi] = WDATA[wi*4 +: 4];

    reg [IDXW+NF*4-1:0] mem [0:S-1];

    reg [1:0]       st;
    reg [IDXW-1:0]  s_lin;
    reg [IDXW-1:0]  c_i;
    reg [IDXW-1:0]  k_i;
    reg [IDXW-1:0]  cnt;
    assign nsteps = cnt;

    wire [NF*4-1:0] dword;
    genvar f;
    generate
        for (f = 0; f < NF; f = f + 1) begin : rd
            assign dword[f*4 +: 4] = wflat[f*S + s_lin];
        end
    endgenerate

    // nibble index inside the flat window bus: element k (x NCH channels) + channel c
    wire [IDXW-1:0] pa = k_i*NCH + c_i;

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= 2'd0; s_lin <= 0; c_i <= 0; k_i <= 0; cnt <= 0; ready <= 1'b0;
        end else begin
            case (st)
                2'd0: begin                                   // SCAN
                    if (dword != {NF*4{1'b0}}) begin
                        mem[cnt] <= {pa, dword};
                        cnt      <= cnt + 1'b1;
                    end
                    if (s_lin == S-1) st <= 2'd1;
                    else begin
                        s_lin <= s_lin + 1'b1;
                        if (k_i == KK-1) begin k_i <= 0; c_i <= c_i + 1'b1; end
                        else k_i <= k_i + 1'b1;
                    end
                end
                2'd1: begin                                   // FIX: all-zero layer -> one dummy step
                    if (cnt == 0) begin
                        mem[0] <= {(IDXW+NF*4){1'b0}};
                        cnt    <= 1;
                    end
                    st <= 2'd2;
                end
                default: ready <= 1'b1;                       // DONE
            endcase
        end
    end

    always @(posedge clk) rq <= mem[raddr];
endmodule
