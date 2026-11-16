`timescale 1ns/1ps

module tb_single_pe;
    logic clk = 1'b1;
    logic rst_n = 1'b0;
    logic clear_acc = 1'b0;
    logic a_valid_in = 1'b0;
    logic b_valid_in = 1'b0;
    logic signed [8:0] a_in = '0;
    logic signed [8:0] b_in = '0;

    logic a_valid_out, b_valid_out;
    logic signed [8:0] a_out, b_out;
    logic signed [31:0] c_out;

    single_pe dut (.*);

    always #5 clk = ~clk;

    task automatic step(
        input logic signed [8:0] a,
        input logic signed [8:0] b,
        input logic a_valid,
        input logic b_valid,
        input logic clear,
        input logic signed [31:0] expected_c
    );
        @(posedge clk);
        a_in = a;
        b_in = b;
        a_valid_in = a_valid;
        b_valid_in = b_valid;
        clear_acc = clear;

        // @(posedge clk);
        #1; // Sample after the registers update.

        if (a_valid_out !== a_valid || (a_valid && a_out !== a))
            $fatal(1, "A forwarding failed at %0t", $time);

        if (b_valid_out !== b_valid || (b_valid && b_out !== b))
            $fatal(1, "B forwarding failed at %0t", $time);

        if (c_out !== expected_c)
            $fatal(1, "At %0t: c_out = %0d, expected %0d",
                   $time, c_out, expected_c);
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1'b1;

        //   A       B       A valid B valid Clear  Expected C
        step( 9'sd7, -9'sd3, 1'b1,   1'b1,   1'b0,  32'sd0);
        step(-9'sd5,  9'sd6, 1'b1,   1'b1,   1'b0,  32'sd0);
        step( 9'sd9,  9'sd4, 1'b1,   1'b0,   1'b0, -32'sd0);
        step( 9'sd8, -9'sd2, 1'b0,   1'b1,   1'b0, -32'sd21);
        step( 9'sd0,  9'sd0, 1'b0,   1'b0,   1'b0, -32'sd51);
        step( 9'sd0,  9'sd0, 1'b0,   1'b0,   1'b0, -32'sd51);
        step( 9'sd0,  9'sd0, 1'b0,   1'b0,   1'b0, -32'sd51);
        step( 9'sd1,  9'sd2, 1'b1,   1'b1,   1'b0, -32'sd51);
        step( 9'sd0,  9'sd0, 1'b0,   1'b0,   1'b0, -32'sd51);
        step( 9'sd0,  9'sd0, 1'b0,   1'b0,   1'b0, -32'sd51);
        step( 9'sd0,  9'sd0, 1'b0,   1'b0,   1'b0, -32'sd49);
        step( 9'sd0,  9'sd0, 1'b0,   1'b0,   1'b1,  32'sd0);

        $display("PASS: single_pe");
        $finish;
    end
endmodule