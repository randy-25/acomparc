module xtile_array #(
    parameter int ARRAY_ROWS    = 8,
    parameter int ARRAY_COLUMNS = 8,
    parameter int INPUT_WIDTH   = 9,
    parameter int ACC_WIDTH     = 32
)(
    input logic clk,
    input logic rst_n,
    input logic clear_acc,

    input logic signed [INPUT_WIDTH-1:0] a_left [0:ARRAY_ROWS-1],
    input logic a_left_valid [0:ARRAY_ROWS-1],
    input logic signed [INPUT_WIDTH-1:0] b_top [0:ARRAY_COLUMNS-1],
    input logic b_top_valid [0:ARRAY_COLUMNS-1],

    output wire signed [ACC_WIDTH-1:0] c_out [0:ARRAY_ROWS-1][0:ARRAY_COLUMNS-1]
);

wire signed [INPUT_WIDTH-1:0] a_link [0:ARRAY_ROWS-1][0:ARRAY_COLUMNS];
wire a_valid [0:ARRAY_ROWS-1][0:ARRAY_COLUMNS];

wire signed [INPUT_WIDTH-1:0] b_link [0:ARRAY_ROWS][0:ARRAY_COLUMNS-1];
wire b_valid [0:ARRAY_ROWS][0:ARRAY_COLUMNS-1];

for (genvar row = 0; row < ARRAY_ROWS; row++) begin : g_left
    assign a_link[row][0] = a_left[row];
    assign a_valid[row][0] = a_left_valid[row];
end

for (genvar col = 0; col < ARRAY_COLUMNS; col++) begin : g_top
    assign b_link[0][col] = b_top[col];
    assign b_valid[0][col] = b_top_valid[col];
end

for (genvar row = 0; row < ARRAY_ROWS; row++) begin : g_row
    for (genvar col = 0; col < ARRAY_COLUMNS; col++) begin : g_col
        single_pe #(
            .INPUT_WIDTH(INPUT_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) pe (
            .clk(clk),
            .rst_n(rst_n),
            .clear_acc(clear_acc),
            .a_in(a_link[row][col]),
            .a_valid_in(a_valid[row][col]),
            .a_out(a_link[row][col+1]),
            .a_valid_out(a_valid[row][col+1]),
            .b_in(b_link[row][col]),
            .b_valid_in(b_valid[row][col]),
            .b_out(b_link[row+1][col]),
            .b_valid_out(b_valid[row+1][col]),
            .c_out(c_out[row][col])
        );
    end
end
endmodule