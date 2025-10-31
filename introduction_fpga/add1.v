module add1(
    input a, b,
    output sum, carry
);
assign sum = a ^ b;
assign carry = a & b;
endmodule
