module tb_cosim;
    // Match these to the export's "n=..., tiles/row=..., passes=..." line.
    // benzene: 6, 2, 24      naphthalene --iters 8: 10, 3, 16
    localparam ROWS = 6, TILES = 2, PASSES = 24;
    localparam SETTLE = 3, SCNT = 32, LAT = 20;
    localparam TOTAL_CHECKS = PASSES * ROWS * TILES * 4;

    reg clk = 0, rst = 1, start = 0, do_load = 1, adc_done = 0;
    reg [31:0] wword = 0, dp = 0, dn = 0;
    reg [4:0]  reads;
    reg [7:0]  sample_cnt;
    wire signed [9:0] adc_data = 10'sd0;
    wire adc_sample, busy, done, period_start;
    wire signed [15:0] sum_pos, sum_neg;
    wire signed [16:0] diff;
    wire [31:0] weights_dac;
    wire [3:0]  pwm;
    wire [7:0]  cnt;

    mac_fsm #(.SETTLE_PERIODS(SETTLE)) dut (
        .clk(clk), .rst(rst), .start(start), .do_load(do_load),
        .weight_word(wword), .dp_word(dp), .dn_word(dn), .reads(reads),
        .sample_cnt(sample_cnt), .adc_sample(adc_sample), .adc_done(adc_done),
        .adc_data(adc_data), .busy(busy), .done(done), .sum_pos(sum_pos),
        .sum_neg(sum_neg), .diff(diff), .weights_dac(weights_dac), .pwm(pwm),
        .period_start(period_start), .cnt(cnt));
    always #5 clk = ~clk;

    reg [31:0] w_mem [0:ROWS*TILES-1];
    reg [31:0] d_mem [0:PASSES*TILES-1];
    reg [63:0] e_mem [0:PASSES*ROWS*TILES-1];

    integer cur_m = 0, cur_r = 0, cur_t = 0;
    integer gidx = 0, pend_g = 0, lat = 0, k;
    integer checks = 0, errors = 0, exp_errs = 0;
    reg        pending = 0;
    reg [15:0] acc [0:3];
    reg [15:0] comp, hexf;
    reg [31:0] dsel;
    reg [63:0] ev;

    always @(negedge clk) if (!rst) begin
        // fixed-latency ADC stub (value is ignored in this test)
        adc_done = 0;
        if (lat > 0) begin
            lat = lat - 1;
            if (lat == 0) adc_done = 1;
        end
        if (adc_sample) begin
            pending = 1; pend_g = gidx; gidx = gidx + 1; lat = LAT;
        end

        // ideal-analog accounting: sum of weight x PWM-high over one period
        if (cnt == 8'd1) for (k = 0; k < 4; k = k + 1) acc[k] = 0;
        for (k = 0; k < 4; k = k + 1)
            acc[k] = acc[k] + (pwm[k] ? weights_dac[8*k +: 8] : 8'd0);

        // end of the period that contained the strobe: compare
        if (cnt == 8'd0 && pending) begin
            pending = 0;
            dsel = pend_g ? dn : dp;
            ev   = e_mem[((2*cur_m + pend_g) * ROWS + cur_r) * TILES + cur_t];
            for (k = 0; k < 4; k = k + 1) begin
                comp = wword[8*k +: 8] * dsel[8*k +: 8];
                hexf = ev[16*k +: 16];
                checks = checks + 1;
                if (acc[k] !== comp) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("FAIL pwm: m%0d r%0d t%0d %s ch%0d hw %0d, w*D %0d",
                                 cur_m, cur_r, cur_t, pend_g ? "neg" : "pos",
                                 k, acc[k], comp);
                end
                if (hexf !== comp) begin
                    exp_errs = exp_errs + 1;
                    if (exp_errs <= 10)
                        $display("FAIL export: m%0d r%0d t%0d %s ch%0d expect.hex %0d, w*D %0d",
                                 cur_m, cur_r, cur_t, pend_g ? "neg" : "pos",
                                 k, hexf, comp);
                end
            end
        end
    end

    initial begin
        #40000000; $display("FAIL: timeout"); $finish;
    end

    integer m, r, t;
    initial begin
        $readmemh("../cosim/weights.hex", w_mem);
        $readmemh("../cosim/duty.hex",    d_mem);
        $readmemh("../cosim/expect.hex",  e_mem);
        reads = 1; sample_cnt = SCNT;
        repeat (3) @(negedge clk);
        rst = 0;
        repeat (2) @(negedge clk);

        for (m = 0; m < PASSES/2; m = m + 1)
          for (r = 0; r < ROWS; r = r + 1)
            for (t = 0; t < TILES; t = t + 1) begin
                cur_m = m; cur_r = r; cur_t = t;
                wword = w_mem[r*TILES + t];
                dp    = d_mem[(2*m)*TILES + t];
                dn    = d_mem[(2*m + 1)*TILES + t];
                gidx  = 0;
                start = 1; @(negedge clk); start = 0;
                @(posedge done);
                @(negedge clk);
                while (pending) @(negedge clk);   // let the last period finish
            end

        $display("checked %0d channel sums (expected %0d)", checks, TOTAL_CHECKS);
        if (checks !== TOTAL_CHECKS)
            $display("FAIL: wrong number of checks");
        else if (errors == 0 && exp_errs == 0) $display("PASS");
        else $display("FAIL: %0d pwm errors, %0d export errors", errors, exp_errs);
        $finish;
    end
endmodule
