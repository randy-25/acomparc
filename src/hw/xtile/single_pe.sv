module single_pe #(
    parameter int INPUT_WIDTH = 9,
    parameter int ACC_WIDTH = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic clear_acc,
    input  logic a_valid_in,
    input  logic b_valid_in,
    input  logic signed [INPUT_WIDTH-1:0] a_in,
    input  logic signed [INPUT_WIDTH-1:0] b_in,
    output logic signed [ACC_WIDTH-1:0] c_out,
    output logic signed [INPUT_WIDTH-1:0] a_out,
    output logic signed [INPUT_WIDTH-1:0] b_out,
    output logic a_valid_out,
    output logic b_valid_out
);

localparam int PRODUCT_WIDTH = 2*INPUT_WIDTH;
localparam int HALF_PRODUCT_INPUT = INPUT_WIDTH / 2;


wire signed [INPUT_WIDTH - 1 : HALF_PRODUCT_INPUT] w_b_hi;
wire signed [HALF_PRODUCT_INPUT - 1 : 0] w_b_lo;
wire signed [INPUT_WIDTH -1 : 0] w_b_hi_ext;
wire signed [INPUT_WIDTH -1 : 0] w_b_lo_ext;

wire signed [PRODUCT_WIDTH - 1:0] w_product_hi;
wire signed [PRODUCT_WIDTH - 1:0] w_product_lo;
reg signed [PRODUCT_WIDTH - 1:0] r_product_hi;
reg signed [PRODUCT_WIDTH - 1:0] r_product_lo;

wire signed [PRODUCT_WIDTH - 1:0] w_product_hi_shifted;
wire signed [PRODUCT_WIDTH - 1:0] w_product_sum;
reg signed [PRODUCT_WIDTH - 1:0] r_product_sum;

wire signed [ACC_WIDTH - 1:0] w_product_sum_ext;
reg signed [ACC_WIDTH - 1:0] r_acc;
wire signed [ACC_WIDTH - 1:0] w_acc;

reg signed [INPUT_WIDTH - 1:0] r_a_out;
reg signed [INPUT_WIDTH - 1:0] r_b_out;
reg r_a_valid_out;
reg r_b_valid_out;

wire mac_en_1;
reg mac_en_2;
reg mac_en_3;

assign mac_en_1 = a_valid_in & b_valid_in;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        r_a_out       <= '0;
        r_b_out       <= '0;
        r_a_valid_out <= 1'b0;
        r_b_valid_out <= 1'b0;
    end else begin
        r_a_out       <= a_in;
        r_b_out       <= b_in;
        r_a_valid_out <= a_valid_in;
        r_b_valid_out <= b_valid_in;
    end
end

assign a_valid_out = r_a_valid_out;
assign b_valid_out = r_b_valid_out;
assign a_out = r_a_out;
assign b_out = r_b_out;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        mac_en_2 <= 1'b0;
        mac_en_3 <= 1'b0;
    end else begin
        mac_en_2 <= mac_en_1;
        mac_en_3 <= mac_en_2;
    end
end

// CYCLE 1

assign w_b_hi = b_in[INPUT_WIDTH - 1 : HALF_PRODUCT_INPUT];
assign w_b_lo = b_in[HALF_PRODUCT_INPUT - 1 : 0];
assign w_b_hi_ext = {{HALF_PRODUCT_INPUT{w_b_hi[INPUT_WIDTH-1]}}, w_b_hi};
assign w_b_lo_ext = {{(INPUT_WIDTH-HALF_PRODUCT_INPUT){1'b0}}, w_b_lo};

assign w_product_hi = a_in * w_b_hi_ext;
assign w_product_lo = a_in * w_b_lo_ext;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        r_product_hi <= '0;
        r_product_lo <= '0;
    end else begin
        if (mac_en_1) begin
            r_product_hi <= w_product_hi;
            r_product_lo <= w_product_lo;
        end else begin
            r_product_hi <= '0;
            r_product_lo <= '0;
        end
    end
end

// CYCLE 2

assign w_product_hi_shifted = r_product_hi <<< HALF_PRODUCT_INPUT;
assign w_product_sum = w_product_hi_shifted + r_product_lo;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        r_product_sum <= '0;
    end else begin
        if (mac_en_2) begin
            r_product_sum <= w_product_sum;
        end else begin
            r_product_sum <= '0;
        end
    end
end

// CYCLE 3

assign w_product_sum_ext = {{(ACC_WIDTH - PRODUCT_WIDTH){r_product_sum[PRODUCT_WIDTH - 1]}}, r_product_sum};
assign w_acc = r_acc + w_product_sum_ext;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        r_acc <= '0;
    end else begin
        if (clear_acc) begin
            r_acc <= '0;
        end else if (mac_en_3) begin
            r_acc <= w_acc;
        end else begin
            r_acc <= r_acc;
        end
    end
end

assign c_out = r_acc;

endmodule