module tb_mac_ctrl;
    reg         clk = 0, rst = 1, shift_en = 0, din = 0;
    reg  [31:0] duty = 0;
    wire [31:0] weights;
    wire [3:0]  pwm;
    wire        period_start;
    integer     errors = 0;

    mac_ctrl dut (.clk(clk), .rst(rst), .shift_en(shift_en), .din(din),
                  .weights(weights), .duty(duty), .pwm(pwm),
                  .period_start(period_start));
    always #5 clk = ~clk;

    wire [7:0] cnt = dut.u_pwm.cnt;

    // ---- weight loading: MSB first, one bit per clock ----
    task load_word(input [31:0] w);
        integer i;
        begin
            for (i = 31; i >= 0; i = i - 1) begin
                @(negedge clk); shift_en = 1; din = w[i];
            end
            @(negedge clk); shift_en = 0;
            if (weights !== w) begin
                errors = errors + 1;
                $display("FAIL load: sent %h, got %h", w, weights);
            end
        end
    endtask

    // ---- apply a duty vector just before a period boundary, hold 3 periods ----
    reg [31:0] cur_vec = 0, next_vec = 0;
    task apply(input [31:0] v);
        begin
            while (cnt != 8'd254) @(negedge clk);
            duty = v; next_vec = v;
            repeat (3 * 255) @(negedge clk);
        end
    endtask

    // ---- count high cycles per counter period (output lags cnt by one cycle) ----
    reg        started = 0;
    integer    periods = 0, k;
    reg [8:0]  acc [0:3];
    always @(negedge clk) if (!rst) begin
        if (cnt == 8'd1) begin
            started = 1;
            for (k = 0; k < 4; k = k + 1) acc[k] = 0;
        end
        if (started) begin
            for (k = 0; k < 4; k = k + 1) acc[k] = acc[k] + pwm[k];
            if (cnt == 8'd0) begin
                periods = periods + 1;
                for (k = 0; k < 4; k = k + 1)
                    if (acc[k] !== cur_vec[8*k +: 8]) begin
                        errors = errors + 1;
                        $display("FAIL pwm: period %0d ch%0d expected %0d, got %0d",
                                 periods, k, cur_vec[8*k +: 8], acc[k]);
                    end
                cur_vec = next_vec;
            end
        end
    end

    initial begin
        repeat (3) @(negedge clk);
        rst = 0;
        @(negedge clk);

        load_word(32'h0000EB00);          // benzene row 0, tile 0: code 235 at element 1
        load_word(32'hEBEBEBEB);
        load_word(32'h80000001);
        load_word(32'h00000000);

        apply({8'd0,   8'd0,   8'd0,   8'd0});
        apply({8'd200, 8'd130, 8'd70,  8'd10});
        apply({8'd0,   8'd0,   8'd0,   8'd0});
        apply({8'd255, 8'd255, 8'd255, 8'd255});
        apply({8'd1,   8'd1,   8'd1,   8'd1});
        apply({8'd193, 8'd129, 8'd65,  8'd64});   // just above each offset
        apply({8'd192, 8'd128, 8'd64,  8'd0});    // exactly at each offset
        apply({8'd100, 8'd100, 8'd100, 8'd100});
        apply({8'd100, 8'd100, 8'd100, 8'd100});  // no change: steady state

        if (errors == 0) $display("PASS");
        else             $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule