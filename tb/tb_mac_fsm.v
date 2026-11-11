module tb_mac_fsm;
    localparam SETTLE = 3, K = 3, SCNT = 32, LAT = 20;

    reg clk = 0, rst = 1, start = 0, do_load = 1, adc_done = 0;
    reg [31:0] wword, dp, dn;
    reg [4:0]  reads;
    reg [7:0]  sample_cnt;
    reg signed [9:0] adc_data = 0;
    wire adc_sample, busy, done, period_start;
    wire signed [15:0] sum_pos, sum_neg;
    wire signed [16:0] diff;
    wire [31:0] weights_dac;
    wire [3:0]  pwm;
    wire [7:0]  cnt;
    integer errors = 0, nsamp = 0, lat = 0, ps_since = 0, j;
    reg [31:0] last_duty = 0;

    mac_fsm #(.SETTLE_PERIODS(SETTLE)) dut (
        .clk(clk), .rst(rst), .start(start), .do_load(do_load),
        .weight_word(wword), .dp_word(dp), .dn_word(dn), .reads(reads),
        .sample_cnt(sample_cnt), .adc_sample(adc_sample), .adc_done(adc_done),
        .adc_data(adc_data), .busy(busy), .done(done), .sum_pos(sum_pos),
        .sum_neg(sum_neg), .diff(diff), .weights_dac(weights_dac), .pwm(pwm),
        .period_start(period_start), .cnt(cnt));
    always #5 clk = ~clk;

    always @(negedge clk) if (!rst) begin
        // periods elapsed since the duty register last changed
        if (dut.duty_r !== last_duty) begin ps_since = 0; last_duty = dut.duty_r; end
        else if (period_start) ps_since = ps_since + 1;

        // ADC stub: fixed latency, returns the 1-based sample number
        adc_done = 0;
        if (lat > 0) begin
            lat = lat - 1;
            if (lat == 0) begin adc_done = 1; adc_data = nsamp; end
        end

        if (adc_sample) begin
            j = nsamp % K;                          // read index within its group
            if (cnt !== SCNT + 1) begin
                errors = errors + 1;
                $display("FAIL strobe at cnt=%0d, expected %0d", cnt, SCNT + 1);
            end
            if (ps_since !== SETTLE + 1 + j) begin
                errors = errors + 1;
                $display("FAIL sample %0d: periods since duty change %0d, expected %0d",
                         nsamp, ps_since, SETTLE + 1 + j);
            end
            if (dut.duty_r !== ((nsamp < K) ? dp : dn)) begin
                errors = errors + 1;
                $display("FAIL sample %0d: wrong duty word %h", nsamp, dut.duty_r);
            end
            nsamp = nsamp + 1; lat = LAT;
        end
    end

    initial begin
        #500000; $display("FAIL: timeout"); $finish;
    end

    initial begin
        wword = 32'h0000EB00;
        dp = {8'd200, 8'd130, 8'd70, 8'd10};
        dn = {8'd0,   8'd30,  8'd0,  8'd90};
        reads = K; sample_cnt = SCNT;
        repeat (3) @(negedge clk);
        rst = 0;
        repeat (2) @(negedge clk);
        start = 1; @(negedge clk); start = 0;

        @(posedge done);
        @(negedge clk);

        if (weights_dac !== wword) begin errors = errors + 1;
            $display("FAIL weights_dac %h, expected %h", weights_dac, wword); end
        if (nsamp !== 2 * K) begin errors = errors + 1;
            $display("FAIL sample count %0d, expected %0d", nsamp, 2 * K); end
        if (sum_pos !== 16'sd6)  begin errors = errors + 1;
            $display("FAIL sum_pos %0d, expected 6", sum_pos); end
        if (sum_neg !== 16'sd15) begin errors = errors + 1;
            $display("FAIL sum_neg %0d, expected 15", sum_neg); end
        if (diff !== -17'sd9)    begin errors = errors + 1;
            $display("FAIL diff %0d, expected -9", diff); end
        if (busy !== 1'b0)       begin errors = errors + 1;
            $display("FAIL busy still high"); end

        if (errors == 0) $display("PASS");
        else             $display("FAIL: %0d errors", errors);
        $finish;
    end
endmodule