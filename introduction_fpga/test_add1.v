`timescale 1ns / 1ps
module stimulus;
	// Inputs
	reg a;
	reg b;
	// Outputs
	wire sum;
    wire carry;
	// Instantiate the Unit Under Test (UUT)
	add1 uut (
		a,
        b,   
		sum,
        carry	
	);
 
	initial begin
	$dumpfile("test.vcd");
    $dumpvars(0,stimulus);
		// Initialize Inputs
		a = 0;
		b = 0;
 
	#20 a = 1;
	#20 b = 1;
	#20 b = 0;	
	#20 a = 1;	  
	#40 ;
 
	end  
 
		initial begin
		 $monitor("t=%3d a=%d,b=%d,sum=%d,carry=%d \n",$time,a,b,sum,carry);
		 end
 
endmodule
 
 
