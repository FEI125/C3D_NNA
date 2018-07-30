module C3D_NNA(	clk, reset, cnn_en, data_in, res_index, data_out, data_valid, cnn_load_en, res_index_valid);

/*-----------------------PARAMETER DECLARATION-----------------------*/
parameter WK = 8;	//原始输入为8bits,fi(8,9),输入后马上缓存为9bits,fi(9,9)
parameter WK_T = 10;
parameter WI = 12;		//输入位宽
parameter WI_T = 14;	//ifmap转换后的位宽
parameter WB = 7;		//conv_bias位宽
parameter WO = 12;		//输出位宽，也是ifmap转换前的位宽
parameter WFI1 = 12;
parameter WFK1 = 8;
parameter WFB1 = 9;
parameter WFK2 = 9;
parameter WFB2 = 9;
parameter INDEX = 3;	//7个分类, 3bit表示0-6

/*--------------------------PORT DECLARATION--------------------------*/
input wire clk;
input wire reset;
input wire cnn_en;
input wire [WI*16-1:0] data_in;
output wire [INDEX-1:0] res_index;
output wire [WO*4-1:0] data_out;
output wire data_valid;
output wire res_index_valid;
output wire cnn_load_en;

/*--------------------------WIRE DECLARATION--------------------------*/
//1 module MAIN_CONTROL
//for 2 module BUFF_FMAP
wire pre_load;
wire load_en;
wire [7:0] addr_load;
wire [7:0] addr_ifmp;
wire [3:0] row_valid;
//for 3 module IFMP_TRANS(NO CONTROL SIGNAL)
//for 4 module BUFF_CONV_WEIGHT
wire [9:0] addr_conv_weight;
//for 5 module WEIGHT_TRANS(NO CONTROL SIGNAL)
//for 6 module PE
wire conv_chan_start;
wire pooling_mode;
wire out_fix_mode;
//for 7 module OFMP_CONTROL
wire data_valid_in;
wire [1:0] data_out_index;
wire fc1_store;
wire fc1_start;
//for 8 module BUFF_FC_WEIGHT_BIAS
wire [11:0] addr_fc1_weight;
wire [9:0] addr_fc2_weight;
wire [9:0] addr_conv_bias;
//for 9 module FC_LAYER
wire fc1_chan_start;	
wire fc2_chan_start;	
wire fc1_cut_act;		
wire fc2_cut_act;
wire [6:0] fc2_chan_cnt;
//2 module BUFF_FMAP
wire [WI*16-1:0] frame1;
wire [WI*16-1:0] frame2;
wire [WI*16-1:0] frame3;
wire [WI*16-1:0] frame4;
//3 module IFMP_TRANS
wire [WI_T*16-1:0] frame1t;
wire [WI_T*16-1:0] frame2t;
wire [WI_T*16-1:0] frame3t;
wire [WI_T*16-1:0] frame4t;
//4 module BUFFER_CONV_WEIGHT
wire [WK*9-1:0] weight1_A;
wire [WK*9-1:0] weight2_A;
wire [WK*9-1:0] weight3_A;
wire [WK*9-1:0] weight1_B;
wire [WK*9-1:0] weight2_B;
wire [WK*9-1:0] weight3_B;
wire [WK*9-1:0] weight1_C;
wire [WK*9-1:0] weight2_C;
wire [WK*9-1:0] weight3_C;
wire [WK*9-1:0] weight1_D;
wire [WK*9-1:0] weight2_D;
wire [WK*9-1:0] weight3_D;
//5 module WEIGHT_TRANS
wire [WK_T*16-1:0] weight1t_A;
wire [WK_T*16-1:0] weight2t_A;
wire [WK_T*16-1:0] weight3t_A;
wire [WK_T*16-1:0] weight1t_B;
wire [WK_T*16-1:0] weight2t_B;
wire [WK_T*16-1:0] weight3t_B;
wire [WK_T*16-1:0] weight1t_C;
wire [WK_T*16-1:0] weight2t_C;
wire [WK_T*16-1:0] weight3t_C;
wire [WK_T*16-1:0] weight1t_D;
wire [WK_T*16-1:0] weight2t_D;
wire [WK_T*16-1:0] weight3t_D;
//6 module PE
wire signed [WO-1:0] ofmp_frame1_A;
wire signed [WO-1:0] ofmp_frame2_A;
wire signed [WO-1:0] ofmp_frame1_B;
wire signed [WO-1:0] ofmp_frame2_B;
wire signed [WO-1:0] ofmp_frame1_C;
wire signed [WO-1:0] ofmp_frame2_C;
wire signed [WO-1:0] ofmp_frame1_D;
wire signed [WO-1:0] ofmp_frame2_D;
//7 module OFMP_CONTROL
wire signed [WFI1-1:0] fc1_ifmp;
//8 module BUFF_FC_WEIGHT_BIAS
wire [WFK1*16-1:0] fc1_weight1;
wire [WFK1*16-1:0] fc1_weight2;
wire [WFK1*16-1:0] fc1_weight3;
wire [WFK1*16-1:0] fc1_weight4;
wire [WFK1*16-1:0] fc1_weight5;
wire [WFK1*16-1:0] fc1_weight6;
wire [WFK1*16-1:0] fc1_weight7;
wire [WFK1*16-1:0] fc1_weight8;
wire [WFK2*7-1:0] fc2_weight;
wire signed [WB-1:0] conv_bias_A;
wire signed [WB-1:0] conv_bias_B;
wire signed [WB-1:0] conv_bias_C;
wire signed [WB-1:0] conv_bias_D;
wire [WFB1*16-1:0] fc1_bias1;
wire [WFB1*16-1:0] fc1_bias2;
wire [WFB1*16-1:0] fc1_bias3;
wire [WFB1*16-1:0] fc1_bias4;
wire [WFB1*16-1:0] fc1_bias5;
wire [WFB1*16-1:0] fc1_bias6;
wire [WFB1*16-1:0] fc1_bias7;
wire [WFB1*16-1:0] fc1_bias8;
wire [WFB2*7-1:0] fc2_bias;
//9 module FC_LAYER(NO FIRST_DEFINED_WIRE)


/*MODULE INSTANTIATION*/
//1
MAIN_CONTROL	MAIN_CONTROL_A(	.clk(clk), .reset(reset), .cnn_en(cnn_en),
								.pre_load(pre_load), .load_en(load_en), .cnn_load_en(cnn_load_en), .addr_load(addr_load), .addr_ifmp(addr_ifmp), .row_valid(row_valid),
								.addr_conv_weight(addr_conv_weight),
								.conv_chan_start(conv_chan_start), .pooling_mode(pooling_mode), .out_fix_mode(out_fix_mode),
								.data_valid_in(data_valid_in), .data_out_index(data_out_index), .fc1_store(fc1_store), .fc1_start(fc1_start),
								.addr_fc1_weight(addr_fc1_weight), .addr_fc2_weight(addr_fc2_weight), .addr_conv_bias(addr_conv_bias), 
								.fc1_chan_start(fc1_chan_start), .fc2_chan_start(fc2_chan_start), .fc1_cut_act(fc1_cut_act), .fc2_cut_act(fc2_cut_act), .fc2_chan_cnt(fc2_chan_cnt));
//2 
BUFF_FMAP BUFF_FMAP_A(.clk(clk), .reset(reset), .data_in(data_in),
					  .pre_load(pre_load), .load_en(load_en), .addr_load(addr_load), .addr_ifmp(addr_ifmp), .row_valid(row_valid), 
					  .frame1(frame1), .frame2(frame2), .frame3(frame3), .frame4(frame4));
//3	OK						
IFMP_TRANS IFMP_TRANS_A(.clk(clk), .reset(reset), 
						.frame1(frame1), .frame2(frame2), .frame3(frame3), .frame4(frame4),
						.frame1t(frame1t), .frame2t(frame2t), .frame3t(frame3t), .frame4t(frame4t));
//4				
BUFF_CONV_WEIGHT	BUFF_CONV_WEIGHT_A(.clk(clk), .addr_conv_weight(addr_conv_weight),
									   .weight1_A(weight1_A), .weight2_A(weight2_A), .weight3_A(weight3_A), 
									   .weight1_B(weight1_B), .weight2_B(weight2_B), .weight3_B(weight3_B),
									   .weight1_C(weight1_C), .weight2_C(weight2_C), .weight3_C(weight3_C),
									   .weight1_D(weight1_D), .weight2_D(weight2_D), .weight3_D(weight3_D));			
//5	OK			 
WEIGHT_TRANS WEIGHT_TRANS_A(.clk(clk), .reset(reset),
							.weight1_A(weight1_A), .weight2_A(weight2_A), .weight3_A(weight3_A), 
							.weight1_B(weight1_B), .weight2_B(weight2_B), .weight3_B(weight3_B),
							.weight1_C(weight1_C), .weight2_C(weight2_C), .weight3_C(weight3_C),
							.weight1_D(weight1_D), .weight2_D(weight2_D), .weight3_D(weight3_D),
							.weight1t_A(weight1t_A), .weight2t_A(weight2t_A), .weight3t_A(weight3t_A),
							.weight1t_B(weight1t_B), .weight2t_B(weight2t_B), .weight3t_B(weight3t_B), 
							.weight1t_C(weight1t_C), .weight2t_C(weight2t_C), .weight3t_C(weight3t_C), 
							.weight1t_D(weight1t_D), .weight2t_D(weight2t_D), .weight3t_D(weight3t_D));
//6-A OK
PE 	PE_A(.clk(clk), .reset(reset), 
		 .conv_chan_start(conv_chan_start), .pooling_mode(pooling_mode), .out_fix_mode(out_fix_mode), 
		 .conv_bias(conv_bias_A),
		 .frame1t(frame1t), .frame2t(frame2t), .frame3t(frame3t), .frame4t(frame4t),
		 .weight1t(weight1t_A), .weight2t(weight2t_A), .weight3t(weight3t_A), 
		 .ofmp_frame1(ofmp_frame1_A), .ofmp_frame2(ofmp_frame2_A));
//6-B OK
PE 	PE_B(.clk(clk), .reset(reset), 
		 .conv_chan_start(conv_chan_start), .pooling_mode(pooling_mode), .out_fix_mode(out_fix_mode), 
		 .conv_bias(conv_bias_B),
		 .frame1t(frame1t), .frame2t(frame2t), .frame3t(frame3t), .frame4t(frame4t),
		 .weight1t(weight1t_B), .weight2t(weight2t_B), .weight3t(weight3t_B), 
		 .ofmp_frame1(ofmp_frame1_B), .ofmp_frame2(ofmp_frame2_B));
//6-C OK		
PE 	PE_C(.clk(clk), .reset(reset), 
		 .conv_chan_start(conv_chan_start), .pooling_mode(pooling_mode), .out_fix_mode(out_fix_mode), 
		 .conv_bias(conv_bias_C),
		 .frame1t(frame1t), .frame2t(frame2t), .frame3t(frame3t), .frame4t(frame4t),
		 .weight1t(weight1t_C), .weight2t(weight2t_C), .weight3t(weight3t_C), 
		 .ofmp_frame1(ofmp_frame1_C), .ofmp_frame2(ofmp_frame2_C));
//6-D OK		
PE 	PE_D(.clk(clk), .reset(reset), 
		 .conv_chan_start(conv_chan_start), .pooling_mode(pooling_mode), .out_fix_mode(out_fix_mode), 
		 .conv_bias(conv_bias_D),
		 .frame1t(frame1t), .frame2t(frame2t), .frame3t(frame3t), .frame4t(frame4t),
		 .weight1t(weight1t_D), .weight2t(weight2t_D), .weight3t(weight3t_D), 
		 .ofmp_frame1(ofmp_frame1_D), .ofmp_frame2(ofmp_frame2_D));
//7 
OFMP_CONTROL OFMP_CONTROL_A(.clk(clk), .reset(reset), 
							.data_valid_in(data_valid_in), .data_out_index(data_out_index), .fc1_store(fc1_store), .fc1_start(fc1_start), 
							.ofmp_frame1_A(ofmp_frame1_A), .ofmp_frame2_A(ofmp_frame2_A), .ofmp_frame1_B(ofmp_frame1_B), .ofmp_frame2_B(ofmp_frame2_B),
							.ofmp_frame1_C(ofmp_frame1_C), .ofmp_frame2_C(ofmp_frame2_C), .ofmp_frame1_D(ofmp_frame1_D), .ofmp_frame2_D(ofmp_frame2_D),
							.fc1_ifmp(fc1_ifmp), .data_out(data_out), .data_valid(data_valid));

//8		 
BUFF_FC_WEIGHT_BIAS BUFF_FC_WEIGHT_BIAS_A(.clk(clk), .reset(reset), .cnn_en(cnn_en),
										  .addr_fc1_weight(addr_fc1_weight), .addr_fc2_weight(addr_fc2_weight), .addr_conv_bias(addr_conv_bias), 
										  .fc1_weight1(fc1_weight1), .fc1_weight2(fc1_weight2), .fc1_weight3(fc1_weight3), .fc1_weight4(fc1_weight4), 
										  .fc1_weight5(fc1_weight5), .fc1_weight6(fc1_weight6), .fc1_weight7(fc1_weight7), .fc1_weight8(fc1_weight8),
										  .fc2_weight(fc2_weight), 
										  .conv_bias_A(conv_bias_A), .conv_bias_B(conv_bias_B), .conv_bias_C(conv_bias_C), .conv_bias_D(conv_bias_D),
										  .fc1_bias1(fc1_bias1), .fc1_bias2(fc1_bias2), .fc1_bias3(fc1_bias3), .fc1_bias4(fc1_bias4), 
										  .fc1_bias5(fc1_bias5), .fc1_bias6(fc1_bias6), .fc1_bias7(fc1_bias7), .fc1_bias8(fc1_bias8), .fc2_bias(fc2_bias));
//9 							
 FC_LAYER FC_LAYER_A(.clk(clk), .reset(reset), 
					 .fc1_chan_start(fc1_chan_start), .fc2_chan_start(fc2_chan_start), .fc1_cut_act(fc1_cut_act), .fc2_cut_act(fc2_cut_act), .fc2_chan_cnt(fc2_chan_cnt),
					 .fc1_weight1(fc1_weight1), .fc1_weight2(fc1_weight2), .fc1_weight3(fc1_weight3), .fc1_weight4(fc1_weight4), 
					 .fc1_weight5(fc1_weight5), .fc1_weight6(fc1_weight6), .fc1_weight7(fc1_weight7), .fc1_weight8(fc1_weight8), 
					 .fc1_bias1(fc1_bias1), .fc1_bias2(fc1_bias2), .fc1_bias3(fc1_bias3), .fc1_bias4(fc1_bias4), 
					 .fc1_bias5(fc1_bias5), .fc1_bias6(fc1_bias6), .fc1_bias7(fc1_bias7), .fc1_bias8(fc1_bias8),
					 .fc1_ifmp(fc1_ifmp), .fc2_weight(fc2_weight), .fc2_bias(fc2_bias), 
					 .res_index(res_index), .res_index_valid(res_index_valid));
					 
endmodule
