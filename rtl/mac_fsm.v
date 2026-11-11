// Controller FSM for one 4-input MAC tile.
// Sequence: [load weights] -> pos group (K reads) -> neg group (K reads) -> done.
// Grouped read order (all +, then all -) cuts the settling-memory bias.
// Weights go through a shadow register (weights_dac), so the DAC inputs never
// move while the shift register is shifting.
// Sample strobe is placed at PWM counter value sample_cnt (not at period_start),
// which avoids the ripple bias at the period start.
//
// Clock: everything is cycle-counted. PWM period = 255 clk cycles.
// SETTLE_PERIODS = ceil(8*tau / T_pwm) = ceil(2016 us / T_pwm):
//   25 MHz clk (10.2 us)  -> 198
//   100 MHz clk (2.55 us) -> 791
module mac_fsm #(
    parameter SETTLE_PERIODS = 198,
    parameter ARM_CNT        = 253     // duty is written when cnt == ARM_CNT
)(
    input         clk,
    input         rst,
    input         start,               // 1-cycle pulse, ignored while busy
    input         do_load,             // 1: shift in weight_word first
    input  [31:0] weight_word,         // {w4,w3,w2,w1}
    input  [31:0] dp_word,             // duty codes for the + pass
    input  [31:0] dn_word,             // duty codes for the - pass
    input  [4:0]  reads,               // K >= 1 reads per group
    input  [7:0]  sample_cnt,          // 1..253; strobe appears at sample_cnt+1
    // ADC handshake
    output reg    adc_sample,          // 1-cycle strobe
    input         adc_done,            // 1-cycle pulse when adc_data is valid
    input  signed [9:0] adc_data,
    // status and results
    output reg    busy,
    output reg    done,                // 1-cycle pulse
    output reg signed [15:0] sum_pos,
    output reg signed [15:0] sum_neg,
    output signed [16:0]     diff,     // sum_pos - sum_neg
    // datapath
    output reg [31:0] weights_dac,
    output [3:0]  pwm,
    output        period_start,
    output [7:0]  cnt
);
    localparam S_IDLE=4'd0, S_LOAD=4'd1, S_FLUSH=4'd2, S_CAP=4'd3, S_ARM=4'd4,
               S_SETTLE=4'd5, S_SWAIT=4'd6, S_ADC=4'd7, S_NEXT=4'd8, S_DONE=4'd9;

    reg [3:0]  state;
    reg        shift_en_r, din_r;
    reg [31:0] duty_r;
    wire [31:0] weights_sr;

    mac_ctrl u_mac (.clk(clk), .rst(rst), .shift_en(shift_en_r), .din(din_r),
                    .weights(weights_sr), .duty(duty_r), .pwm(pwm),
                    .period_start(period_start), .cnt(cnt));

    // latched job parameters
    reg [31:0] w_l, dp_l, dn_l;
    reg [4:0]  k_l, rd;
    reg [7:0]  s_l;
    reg        load_l, grp;            // grp: 0 = + group, 1 = - group
    reg [4:0]  ld_cnt;
    reg [9:0]  pcount;                 // supports SETTLE_PERIODS up to 1023

    assign diff = sum_pos - sum_neg;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; shift_en_r <= 0; din_r <= 0; duty_r <= 0;
            adc_sample <= 0; busy <= 0; done <= 0;
            sum_pos <= 0; sum_neg <= 0; weights_dac <= 0;
            w_l <= 0; dp_l <= 0; dn_l <= 0; k_l <= 1; rd <= 0; s_l <= 8'd1;
            load_l <= 0; grp <= 0; ld_cnt <= 0; pcount <= 0;
        end else begin
            adc_sample <= 1'b0;
            done       <= 1'b0;
            case (state)
            S_IDLE: if (start) begin
                w_l <= weight_word; dp_l <= dp_word; dn_l <= dn_word;
                k_l <= (reads == 0) ? 5'd1 : reads;
                s_l <= (sample_cnt < 1) ? 8'd1 : sample_cnt;
                load_l <= do_load;
                sum_pos <= 0; sum_neg <= 0; grp <= 0; ld_cnt <= 0;
                busy <= 1;
                state <= do_load ? S_LOAD : S_ARM;
            end
            S_LOAD: begin                       // 32 shifts, MSB first
                shift_en_r <= 1'b1;
                din_r      <= w_l[31 - ld_cnt];
                ld_cnt     <= ld_cnt + 5'd1;
                if (ld_cnt == 5'd31) state <= S_FLUSH;
            end
            S_FLUSH: begin shift_en_r <= 1'b0; state <= S_CAP; end
            S_CAP:   begin weights_dac <= weights_sr; state <= S_ARM; end
            S_ARM: if (cnt == ARM_CNT) begin    // duty stable before the cnt==0 cycle
                duty_r <= grp ? dn_l : dp_l;
                pcount <= 0; rd <= 0;
                state  <= S_SETTLE;
            end
            S_SETTLE: if (period_start) begin   // start of period index pcount
                if (pcount == SETTLE_PERIODS) state <= S_SWAIT;
                else pcount <= pcount + 10'd1;
            end
            S_SWAIT: if (cnt == s_l) begin
                adc_sample <= 1'b1;
                state <= S_ADC;
            end
            S_ADC: if (adc_done) begin
                if (grp) sum_neg <= sum_neg + adc_data;
                else     sum_pos <= sum_pos + adc_data;
                rd <= rd + 5'd1;
                if (rd + 5'd1 == k_l) begin
                    if (grp) state <= S_DONE;
                    else begin grp <= 1'b1; state <= S_ARM; end
                end else state <= S_NEXT;
            end
            S_NEXT: if (period_start) state <= S_SWAIT;  // next read, next period
            S_DONE: begin done <= 1'b1; busy <= 1'b0; state <= S_IDLE; end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule