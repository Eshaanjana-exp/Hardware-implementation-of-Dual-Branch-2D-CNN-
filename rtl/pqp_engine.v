`timescale 1ns/1ps
// PQP-Conv engine: Parallel-Quantized-Pixel-wise convolution.
//  * ONE input pixel (UINT4) is fetched per cycle and BROADCAST to NF filter lanes,
//  * each lane multiplies by its own INT4 weight (LUT multiplier, no DSP),
//  * partial sums are accumulated PROGRESSIVELY over all steps (channels x taps),
//  * the step sequence comes from the SZW compact list (zero-weight steps skipped).
// Output per lane: clamp( (acc + bias) >>> QSHIFT , 0 , 15 )  (UINT4).
// With bias = 0 and QSHIFT = 0 this is exactly clamp(sum(p*w),0,15).
(* use_dsp48 = "no" *)
module pqp_engine #(
    parameter NF        = 8,
    parameter IDXW      = 7,
    parameter ACCW      = 20,
    parameter QSHIFT    = 0,
    parameter [NF*8-1:0] BDATA = 0      // signed 8-bit bias per lane, lane f at [8*f +: 8]
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire [IDXW-1:0]         nsteps,
    output wire [IDXW-1:0]         list_raddr,
    input  wire [IDXW+NF*4-1:0]    list_q,
    output wire [IDXW-1:0]         pix_idx,     // nibble address into the owner's window bus
    input  wire [3:0]              pix_val,     // window nibble selected by pix_idx (combinational)
    output reg                     done,        // 1-clk pulse, y valid (and held) afterwards
    output wire [NF*4-1:0]         y
);
    reg             run;
    reg [IDXW-1:0]  cnt;
    reg             v1, f1, l1;
    reg             v2, f2, l2;
    reg             v3, l3;
    reg [3:0]       pix_r;
    reg [NF*4-1:0]  w_r;

    assign list_raddr = cnt;
    assign pix_idx    = list_q[NF*4 +: IDXW];

    wire last_issue = (cnt == nsteps - 1'b1);

    always @(posedge clk) begin
        if (!rst_n) begin
            run <= 1'b0; cnt <= 0;
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; l3 <= 1'b0; done <= 1'b0;
            f1 <= 1'b0; l1 <= 1'b0; f2 <= 1'b0; l2 <= 1'b0;
        end else begin
            done <= v3 & l3;
            if (start) begin
                run <= 1'b1; cnt <= 0;
            end else if (run) begin
                cnt <= cnt + 1'b1;
                if (last_issue) run <= 1'b0;
            end
            // stage 0 -> 1 (list RAM read happens on this edge)
            v1 <= run; f1 <= (cnt == 0); l1 <= last_issue;
            // stage 1 -> 2 (window nibble + weights captured)
            pix_r <= pix_val;
            w_r   <= list_q[NF*4-1:0];
            v2 <= v1; f2 <= f1; l2 <= l1;
            // stage 2 -> 3 (accumulate)
            v3 <= v2; l3 <= l2;
        end
    end

    genvar f;
    generate
        for (f = 0; f < NF; f = f + 1) begin : lane
            (* use_dsp48 = "no" *) wire signed [8:0] prod =
                $signed({1'b0, pix_r}) * $signed(w_r[f*4 +: 4]);
            reg  signed [ACCW-1:0] acc;
            reg         [3:0]      yf;
            wire signed [ACCW-1:0] bias_ext = {{(ACCW-8){BDATA[f*8+7]}}, BDATA[f*8 +: 8]};
            wire signed [ACCW-1:0] base     = f2 ? bias_ext : acc;
            wire signed [ACCW-1:0] shifted  = acc >>> QSHIFT;
            always @(posedge clk) begin
                if (v2) acc <= base + prod;
                if (v3 & l3)
                    yf <= (shifted < 0) ? 4'd0 : (shifted > 15) ? 4'd15 : shifted[3:0];
            end
            assign y[f*4 +: 4] = yf;
        end
    endgenerate
endmodule
