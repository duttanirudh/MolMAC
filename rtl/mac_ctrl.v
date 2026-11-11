// Digital control for the 4-input hybrid MAC.
//  - 32-bit shift register holds four 8-bit weights (sent to the DACs)
//  - Four staggered 8-bit PWM generators carry the activations
// PERIOD is 255 (not 256) on purpose: duty = value/255, so 255 means always on.
// This matches the Python model, which quantizes with round(a*255)/255.
//
// Duty timing: the duty input is sampled at the start of each period and held in
// duty_l for the rest of it. On the first cycle of a period (cnt == 0) the compare
// uses the incoming duty directly, so the new value applies from cycle 0 and every
// period carries exactly `value` high cycles. The duty input must therefore be
// stable by the start of the cnt == 0 cycle (the cycle where period_start is high).

module shift_reg #(parameter W = 32) (
    input              clk,
    input              rst,
    input              shift_en,
    input              din,          // serial data, MSB first
    output reg [W-1:0] q
);
    always @(posedge clk) begin
        if (rst)           q <= {W{1'b0}};
        else if (shift_en) q <= {q[W-2:0], din};
    end
endmodule


module pwm4 #(parameter PERIOD = 255) (
    input         clk,
    input         rst,
    input  [31:0] duty,              // {d4, d3, d2, d1}, 8 bits each
    output reg [3:0] pwm,
    output        period_start,      // 1-cycle marker for ADC sync
    output reg [7:0] cnt
);
    reg [31:0] duty_l;               // duty held for cnt >= 1 (no mid-period glitches)

    assign period_start = (cnt == 8'd0);

    // Duty used for this cycle's compare: the incoming value on the first cycle of
    // the period, the latched value afterwards.
    wire [31:0] d = (cnt == 8'd0) ? duty : duty_l;

    // Phase-shifted counter value: (cnt + off) mod PERIOD
    function [7:0] ph(input [7:0] c, input [8:0] off);
        reg [8:0] s;
        begin
            s  = c + off;
            ph = (s >= PERIOD) ? (s - PERIOD) : s;
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            cnt    <= 8'd0;
            duty_l <= 32'd0;
            pwm    <= 4'b0000;
        end else begin
            cnt <= (cnt == PERIOD - 1) ? 8'd0 : cnt + 8'd1;
            if (cnt == 8'd0) duty_l <= duty;

            // Staggered by about a quarter period per channel
            pwm[0] <= ph(cnt,   9'd0)   < d[7:0];
            pwm[1] <= ph(cnt,   9'd64)  < d[15:8];
            pwm[2] <= ph(cnt,   9'd128) < d[23:16];
            pwm[3] <= ph(cnt,   9'd192) < d[31:24];
        end
    end
endmodule


module mac_ctrl (
    input         clk,
    input         rst,
    // weight loading (serial)
    input         shift_en,
    input         din,
    output [31:0] weights,           // {w4, w3, w2, w1} -> DAC inputs
    // activations (parallel)
    input  [31:0] duty,
    output [3:0]  pwm,               // -> analog switch controls
    output        period_start,
    output [7:0]  cnt
);
    shift_reg #(.W(32)) u_sr  (.clk(clk), .rst(rst), .shift_en(shift_en),
                               .din(din), .q(weights));
    pwm4      #(.PERIOD(255)) u_pwm (.clk(clk), .rst(rst), .duty(duty),
                               .pwm(pwm), .period_start(period_start), .cnt(cnt));
endmodule