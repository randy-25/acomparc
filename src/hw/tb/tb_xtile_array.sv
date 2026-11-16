`timescale 1ns/1ps

module tb_xtile_array;
    localparam int SIZE = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear_acc = 1'b0;

    logic signed [8:0] a_left [0:SIZE-1];
    logic signed [8:0] b_top  [0:SIZE-1];
    logic a_left_valid [0:SIZE-1];
    logic b_top_valid  [0:SIZE-1];

    wire signed [31:0] c_out [0:SIZE-1][0:SIZE-1];

    localparam logic signed [31:0] EXPECTED [0:1][0:1] =
        '{'{32'sd19, 32'sd22}, '{32'sd43, 32'sd50}};

    xtile_array dut (.*);

    always #5 clk = ~clk;

    task automatic step(
        input logic signed [8:0] a0, a1, b0, b1,
        input logic [1:0] a_valid, b_valid
    );
        @(negedge clk);
        a_left[0] = a0;
        a_left[1] = a1;
        b_top[0] = b0;
        b_top[1] = b1;

        a_left_valid[0] = a_valid[0];
        a_left_valid[1] = a_valid[1];
        b_top_valid[0] = b_valid[0];
        b_top_valid[1] = b_valid[1];

        @(posedge clk);
        #1;
    endtask

    initial begin
        for (int lane = 0; lane < SIZE; lane++) begin
            a_left[lane] = '0;
            b_top[lane] = '0;
            a_left_valid[lane] = 1'b0;
            b_top_valid[lane] = 1'b0;
        end

        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        clear_acc = 1'b1;
        @(negedge clk);
        clear_acc = 1'b0;

        //       A row 0  A row 1  B col 0  B col 1  A mask  B mask
        step(     9'sd1,      '0,    9'sd5,      '0,  2'b01,  2'b01); // Edge 0
        step(     9'sd2,   9'sd3,    9'sd7,   9'sd6,  2'b11,  2'b11); // Edge 1
        step(        '0,   9'sd4,       '0,   9'sd8,  2'b10,  2'b10); // Edge 2
        repeat (3) step('0, '0, '0, '0, 2'b00, 2'b00);               // Edges 3–5

        for (int row = 0; row < SIZE; row++) begin
            for (int col = 0; col < SIZE; col++) begin
                logic signed [31:0] expected_c;

                expected_c = '0;
                if (row < 2 && col < 2)
                    expected_c = EXPECTED[row][col];

                if (c_out[row][col] !== expected_c)
                    $fatal(1, "C[%0d][%0d] = %0d, expected %0d",
                           row, col, c_out[row][col], expected_c);
            end
        end

        $display("PASS: xtile_array 2x2");
        $finish;
    end
endmodule