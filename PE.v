module PE(	clk, reset, conv_chan_start, pooling_mode, out_fix_mode, conv_bias,
			frame1t, frame2t, frame3t, frame4t, weight1t, weight2t, weight3t,   
			ofmp_frame1, ofmp_frame2);

/*PARAMETER DECLARATION*/
parameter WI_T = 14;	//输入位宽,ifmap转换后的位宽
parameter WK_T = 10;		//weight位宽
parameter WM_O = 20;	//原始乘积需先保留的位宽
parameter WO_T = 12;	//mul_cut [20:9] -> [WM_O-1:WM_O-WO_T]=>[11:0]; 乘积截取后的位宽，也是OFMP输出转换的位宽
parameter WB = 7;		//conv_bias位宽
parameter WFR = 13;		//三个frame相加的位宽
parameter WCH = 14;		//所有channel相加的位宽
parameter WO = 12;		//输出位宽，也是ifmap转换前的位宽

/*PORT DECLARATION*/
input wire clk;
input wire reset;

input wire conv_chan_start;		//1 指示channel_sum加和conv_bias与oft2输出;	 0 指示channel_sum加和自身与frame_sum输出
input wire pooling_mode;		//1 指示conv1_pooling,frames不减少;		  	 0 指示other_conv,frame参与pooling
input wire out_fix_mode;		//1 指示conv1&2, relu截取[13:0]->[12:1]; 	 0 指示con3&4&5,relu截取[13:0]->[13:2]

input wire signed [WB-1:0] conv_bias;

input wire [WI_T*16-1:0] frame1t; //frame1  high->low (1,1)16, (1,2)15, (1,3)14, (1,4)13
input wire [WI_T*16-1:0] frame2t; //		  high->low (2,1)12, (2,2)11, (2,3)10, (2,4)9
input wire [WI_T*16-1:0] frame3t; //		  high->low (3,1)8,  (3,2)7,  (3,3)6,  (3,4)5
input wire [WI_T*16-1:0] frame4t; //		  high->low (4,1)4,  (4,2)3,  (4,3)2,  (4,4)1

input wire [WK_T*16-1:0] weight1t;	//frame1_weight   high->low (1,1)16, (1,2)15, (1,3)14, (1,4)13	
input wire [WK_T*16-1:0] weight2t;	//f2	 		  high->low (2,1)12, (2,2)11, (2,3)10, (2,4)9
input wire [WK_T*16-1:0] weight3t;	//f3			  high->low (3,1)8,  (3,2)7,  (3,3)6,  (3,4)5
									//				  high->low (4,1)4,  (4,2)3,  (4,3)2,  (4,4)1
									
output reg signed [WO-1:0] ofmp_frame1;	//pooling_mode = 0/1,都有效; 	存储ofmap_frame1,根据out_fix_mode做相应截取
output reg signed [WO-1:0] ofmp_frame2;	//pooling_mode = 0时,无效,置零; 存储ofmap_frame2,根据out_fix_mode做相应截取

/*REGISTER DECLARATION*/
reg signed [WM_O-1:0] mul_ori1 [0:15];	//f1*w1		|(1,1)0  (1,2)1  (1,3)2  (1,4)3 |
reg signed [WM_O-1:0] mul_ori2 [0:15];	//f2*w2     |(2,1)4  (2,2)5  (2,3)6  (2,4)7 |
reg signed [WM_O-1:0] mul_ori3 [0:15];	//f3*w3     |(3,1)8  (3,2)9  (3,3)10 (3,4)11|
reg signed [WM_O-1:0] mul_ori4 [0:15];	//f2*w1     |(4,1)12 (4,2)13 (4,3)14 (4,4)15|
reg signed [WM_O-1:0] mul_ori5 [0:15];	//f3*w2
reg signed [WM_O-1:0] mul_ori6 [0:15];	//f4*w3

reg signed [WO_T-1:0] mul_cut1 [0:15];	//f1*w1 cut		|(1,1)0  (1,2)1  (1,3)2  (1,4)3 |
reg signed [WO_T-1:0] mul_cut2 [0:15];  //f2*w2 cut     |(2,1)4  (2,2)5  (2,3)6  (2,4)7 |
reg signed [WO_T-1:0] mul_cut3 [0:15];  //f3*w3 cut     |(3,1)8  (3,2)9  (3,3)10 (3,4)11|
reg signed [WO_T-1:0] mul_cut4 [0:15];  //f2*w1 cut     |(4,1)12 (4,2)13 (4,3)14 (4,4)15|
reg signed [WO_T-1:0] mul_cut5 [0:15];  //f3*w2 cut
reg signed [WO_T-1:0] mul_cut6 [0:15];  //f4*w3 cut

reg signed [WO_T-1:0] oft1_1 [0:7];		//f1*w1 ofmp_trans1		|(1,1)0  (1,2)1  (1,3)2  (1,4)3|
reg signed [WO_T-1:0] oft1_2 [0:7];     //f2*w2 ofmp_trans1 	|(2,1)4  (2,2)5  (2,3)6  (2,4)7|
reg signed [WO_T-1:0] oft1_3 [0:7];     //f3*w3 ofmp_trans1 
reg signed [WO_T-1:0] oft1_4 [0:7];     //f2*w1 ofmp_trans1 
reg signed [WO_T-1:0] oft1_5 [0:7];     //f3*w2 ofmp_trans1 
reg signed [WO_T-1:0] oft1_6 [0:7];     //f4*w3 ofmp_trans1 

reg signed [WO_T-1:0] oft2_1 [0:3];		//f1*w1 ofmp_trans2		|(1,1)0  (1,2)1|
reg signed [WO_T-1:0] oft2_2 [0:3];     //f2*w2 ofmp_trans2 	|(2,1)2  (2,2)3|
reg signed [WO_T-1:0] oft2_3 [0:3];     //f3*w3 ofmp_trans2 
reg signed [WO_T-1:0] oft2_4 [0:3];     //f2*w1 ofmp_trans2 
reg signed [WO_T-1:0] oft2_5 [0:3];     //f3*w2 ofmp_trans2 
reg signed [WO_T-1:0] oft2_6 [0:3];     //f4*w3 ofmp_trans2 

reg signed [WFR-1:0] frame1_sum[0:3];	//f1w1_t+f2w2_t+f3w3_t	  |(1,1)0  (1,2)1|
reg signed [WFR-1:0] frame2_sum[0:3];	//f2w1_t+f3w3_t+f4w3_t    |(2,1)2  (2,2)3|

reg signed [WCH-1:0] chan_sum1[0:3];	//frame1_sum	|(1,1)0  (1,2)1|
reg signed [WCH-1:0] chan_sum2[0:3];	//frame2_sum    |(2,1)2  (2,2)3|

reg signed [WCH-1:0] relu1[0:3];		//channel_sum1 relu		|(1,1)0  (1,2)1|
reg signed [WCH-1:0] relu2[0:3];		//channel_sum2 relu     |(2,1)2  (2,2)3|

reg signed [WCH-1:0] pooling1;			//relu1 pooling 2*2->1*1	由其根据out_fix_mode做相应截取，结果传递给输出ofmp_frame1;
reg signed [WCH-1:0] pooling2;			//relu2 pooling 2*2->1*1    由其根据out_fix_mode做相应截取，结果传递给输出ofmp_frame2;

reg signed [WCH-1:0] pool_fr1_1;		//  |relu1(1,1)  relu1(1,2)  relu1(2,1)  relu1(2,2)|  **COMBINATIONAL LOGIC REGISTER**	
reg signed [WCH-1:0] pool_fr1_2;		//  |	   	 fr1_1			   		 fr1_2	   	   |
reg signed [WCH-1:0] pool_fr2_1;		//  |relu2(1,1)  relu2(1,2)  relu2(2,1)  relu2(2,2)|  **COMBINATIONAL LOGIC REGISTER**
reg signed [WCH-1:0] pool_fr2_2;        //  |	   	 fr2_1			   		 fr2_2	   	   |
reg signed [WCH-1:0] pool_fr1;			//  |fr1_1	fr1_2| -> pool_fr1
reg signed [WCH-1:0] pool_fr2;			//  |fr2_1	fr2_2| -> pool_fr2

//信号延迟
reg conv_chan_start_d1;
reg conv_chan_start_d2;
reg conv_chan_start_d3;
reg conv_chan_start_d4;
reg conv_chan_start_d5;
reg conv_chan_start_d6;
reg conv_chan_start_d7;
reg conv_chan_start_d8;
reg conv_chan_start_d9;
reg conv_chan_start_d10;
reg conv_chan_start_d11;
reg conv_chan_start_d12;
reg conv_chan_start_d13;
reg conv_chan_start_d14;
reg conv_chan_start_d15;
reg conv_chan_start_cur;

//conv_chan_start+3
reg pooling_mode_d1;
reg pooling_mode_d2;
reg pooling_mode_d3;
reg pooling_mode_d4;
reg pooling_mode_d5;
reg pooling_mode_d6;
reg pooling_mode_d7;
reg pooling_mode_d8;
reg pooling_mode_d9;
reg pooling_mode_d10;
reg pooling_mode_d11;
reg pooling_mode_d12;
reg pooling_mode_d13;
reg pooling_mode_d14;
reg pooling_mode_d15;
reg pooling_mode_d16;
reg pooling_mode_d17;
reg pooling_mode_d18;
reg pooling_mode_cur;

//conv_chan_start+4
reg out_fix_mode_d1;
reg out_fix_mode_d2;
reg out_fix_mode_d3;
reg out_fix_mode_d4;
reg out_fix_mode_d5;
reg out_fix_mode_d6;
reg out_fix_mode_d7;
reg out_fix_mode_d8;
reg out_fix_mode_d9;
reg out_fix_mode_d10;
reg out_fix_mode_d11;
reg out_fix_mode_d12;
reg out_fix_mode_d13;
reg out_fix_mode_d14;
reg out_fix_mode_d15;
reg out_fix_mode_d16;
reg out_fix_mode_d17;
reg out_fix_mode_d18;
reg out_fix_mode_d19;
reg out_fix_mode_cur;

//conv_chan_start -1
reg signed [WB-1:0] conv_bias_d1;
reg signed [WB-1:0] conv_bias_d2;
reg signed [WB-1:0] conv_bias_d3;
reg signed [WB-1:0] conv_bias_d4;
reg signed [WB-1:0] conv_bias_d5;
reg signed [WB-1:0] conv_bias_d6;
reg signed [WB-1:0] conv_bias_d7;
reg signed [WB-1:0] conv_bias_d8;
reg signed [WB-1:0] conv_bias_d9;
reg signed [WB-1:0] conv_bias_d10;
reg signed [WB-1:0] conv_bias_d11;
reg signed [WB-1:0] conv_bias_d12;
reg signed [WB-1:0] conv_bias_d13;
reg signed [WB-1:0] conv_bias_d14;
reg signed [WB-1:0] conv_bias_cur;

/*OPERATION*/
//信号延迟逻辑
always@(posedge clk or negedge reset)
begin
if(!reset)
	begin
	conv_chan_start_d1 <= 0;
	conv_chan_start_d2 <= 0;
	conv_chan_start_d3 <= 0;
	conv_chan_start_d4 <= 0;
	conv_chan_start_d5 <= 0;
	conv_chan_start_d6 <= 0;
	conv_chan_start_d7 <= 0;
	conv_chan_start_d8 <= 0;
	conv_chan_start_d9 <= 0;
	conv_chan_start_d10 <= 0;
	conv_chan_start_d11 <= 0;
	conv_chan_start_d12 <= 0;
	conv_chan_start_d13 <= 0;
	conv_chan_start_d14 <= 0;
	conv_chan_start_d15 <= 0;
	conv_chan_start_cur <= 0;
	
	pooling_mode_d1 <= 0;
	pooling_mode_d2 <= 0;
	pooling_mode_d3 <= 0;
	pooling_mode_d4 <= 0;
	pooling_mode_d5 <= 0;
	pooling_mode_d6 <= 0;
	pooling_mode_d7 <= 0;
	pooling_mode_d8 <= 0;
	pooling_mode_d9 <= 0;
	pooling_mode_d10 <= 0;
	pooling_mode_d11 <= 0;
	pooling_mode_d12 <= 0;
	pooling_mode_d13 <= 0;
	pooling_mode_d14 <= 0;
	pooling_mode_d15 <= 0;
	pooling_mode_d16 <= 0;
	pooling_mode_d17 <= 0;
	pooling_mode_d18 <= 0;
	pooling_mode_cur <= 0;
	
	out_fix_mode_d1 <= 0;
	out_fix_mode_d2 <= 0;
	out_fix_mode_d3 <= 0;
	out_fix_mode_d4 <= 0;
	out_fix_mode_d5 <= 0;
	out_fix_mode_d6 <= 0;
	out_fix_mode_d7 <= 0;
	out_fix_mode_d8 <= 0;
	out_fix_mode_d9 <= 0;
	out_fix_mode_d10 <= 0;
	out_fix_mode_d11 <= 0;
	out_fix_mode_d12 <= 0;
	out_fix_mode_d13 <= 0;
	out_fix_mode_d14 <= 0;
	out_fix_mode_d15 <= 0;
	out_fix_mode_d16 <= 0;
	out_fix_mode_d17 <= 0;
	out_fix_mode_d18 <= 0;
	out_fix_mode_d19 <= 0;
	out_fix_mode_cur <= 0;
	
	conv_bias_d1 <= 0;
	conv_bias_d2 <= 0;
	conv_bias_d3 <= 0;
	conv_bias_d4 <= 0;
	conv_bias_d5 <= 0;
	conv_bias_d6 <= 0;
	conv_bias_d7 <= 0;
	conv_bias_d8 <= 0;
	conv_bias_d9 <= 0;
	conv_bias_d10 <= 0;
	conv_bias_d11 <= 0;
	conv_bias_d12 <= 0;
	conv_bias_d13 <= 0;
	conv_bias_d14 <= 0;
	conv_bias_cur <= 0;
	end
else
	begin
	conv_chan_start_d1  <= conv_chan_start;
	conv_chan_start_d2  <= conv_chan_start_d1;
	conv_chan_start_d3  <= conv_chan_start_d2;
	conv_chan_start_d4  <= conv_chan_start_d3;
	conv_chan_start_d5  <= conv_chan_start_d4;
	conv_chan_start_d6  <= conv_chan_start_d5;
	conv_chan_start_d7  <= conv_chan_start_d6;
	conv_chan_start_d8  <= conv_chan_start_d7;
	conv_chan_start_d9  <= conv_chan_start_d8;
	conv_chan_start_d10 <= conv_chan_start_d9;
	conv_chan_start_d11 <= conv_chan_start_d10;
	conv_chan_start_d12 <= conv_chan_start_d11;
	conv_chan_start_d13 <= conv_chan_start_d12;
	conv_chan_start_d14 <= conv_chan_start_d13;
	conv_chan_start_d15 <= conv_chan_start_d14;
	conv_chan_start_cur <= conv_chan_start_d15;
	
	pooling_mode_d1  <= pooling_mode;
	pooling_mode_d2  <= pooling_mode_d1;
	pooling_mode_d3  <= pooling_mode_d2;
	pooling_mode_d4  <= pooling_mode_d3;
	pooling_mode_d5  <= pooling_mode_d4;
	pooling_mode_d6  <= pooling_mode_d5;
	pooling_mode_d7  <= pooling_mode_d6;
	pooling_mode_d8  <= pooling_mode_d7;
	pooling_mode_d9  <= pooling_mode_d8;
	pooling_mode_d10 <= pooling_mode_d9;
	pooling_mode_d11 <= pooling_mode_d10;
	pooling_mode_d12 <= pooling_mode_d11;
	pooling_mode_d13 <= pooling_mode_d12;
	pooling_mode_d14 <= pooling_mode_d13;
	pooling_mode_d15 <= pooling_mode_d14;
	pooling_mode_d16 <= pooling_mode_d15;
	pooling_mode_d17 <= pooling_mode_d16;
	pooling_mode_d18 <= pooling_mode_d17;
	pooling_mode_cur <= pooling_mode_d18;
	
	out_fix_mode_d1  <= out_fix_mode;
	out_fix_mode_d2  <= out_fix_mode_d1;
	out_fix_mode_d3  <= out_fix_mode_d2;
	out_fix_mode_d4  <= out_fix_mode_d3;
	out_fix_mode_d5  <= out_fix_mode_d4;
	out_fix_mode_d6  <= out_fix_mode_d5;
	out_fix_mode_d7  <= out_fix_mode_d6;
	out_fix_mode_d8  <= out_fix_mode_d7;
	out_fix_mode_d9  <= out_fix_mode_d8;
	out_fix_mode_d10 <= out_fix_mode_d9;
	out_fix_mode_d11 <= out_fix_mode_d10;
	out_fix_mode_d12 <= out_fix_mode_d11;
	out_fix_mode_d13 <= out_fix_mode_d12;
	out_fix_mode_d14 <= out_fix_mode_d13;
	out_fix_mode_d15 <= out_fix_mode_d14;
	out_fix_mode_d16 <= out_fix_mode_d15;
	out_fix_mode_d17 <= out_fix_mode_d16;
	out_fix_mode_d18 <= out_fix_mode_d17;
	out_fix_mode_d19 <= out_fix_mode_d18;
	out_fix_mode_cur <= out_fix_mode_d19;
	
	conv_bias_d1  <= conv_bias;
	conv_bias_d2  <= conv_bias_d1;
	conv_bias_d3  <= conv_bias_d2;
	conv_bias_d4  <= conv_bias_d3;
	conv_bias_d5  <= conv_bias_d4;
	conv_bias_d6  <= conv_bias_d5;
	conv_bias_d7  <= conv_bias_d6;
	conv_bias_d8  <= conv_bias_d7;
	conv_bias_d9  <= conv_bias_d8;
	conv_bias_d10 <= conv_bias_d9;
	conv_bias_d11 <= conv_bias_d10;
	conv_bias_d12 <= conv_bias_d11;
	conv_bias_d13 <= conv_bias_d12;
	conv_bias_d14 <= conv_bias_d13;
	conv_bias_cur <= conv_bias_d14;
	end
end

//矩阵对应元素相乘
always@(posedge clk)
begin
//f1*w1
mul_ori1[0]  <= $signed(frame1t[WI_T*16-1:WI_T*15]) * $signed(weight1t[WK_T*16-1:WK_T*15]);
mul_ori1[1]  <= $signed(frame1t[WI_T*15-1:WI_T*14]) * $signed(weight1t[WK_T*15-1:WK_T*14]);
mul_ori1[2]  <= $signed(frame1t[WI_T*14-1:WI_T*13]) * $signed(weight1t[WK_T*14-1:WK_T*13]);
mul_ori1[3]  <= $signed(frame1t[WI_T*13-1:WI_T*12]) * $signed(weight1t[WK_T*13-1:WK_T*12]);
mul_ori1[4]  <= $signed(frame1t[WI_T*12-1:WI_T*11]) * $signed(weight1t[WK_T*12-1:WK_T*11]);
mul_ori1[5]  <= $signed(frame1t[WI_T*11-1:WI_T*10]) * $signed(weight1t[WK_T*11-1:WK_T*10]);
mul_ori1[6]  <= $signed(frame1t[WI_T*10-1:WI_T*9])  * $signed(weight1t[WK_T*10-1:WK_T*9]);
mul_ori1[7]  <= $signed(frame1t[WI_T*9-1:WI_T*8]) * $signed(weight1t[WK_T*9-1:WK_T*8]);
mul_ori1[8]  <= $signed(frame1t[WI_T*8-1:WI_T*7]) * $signed(weight1t[WK_T*8-1:WK_T*7]);
mul_ori1[9]  <= $signed(frame1t[WI_T*7-1:WI_T*6]) * $signed(weight1t[WK_T*7-1:WK_T*6]);
mul_ori1[10] <= $signed(frame1t[WI_T*6-1:WI_T*5]) * $signed(weight1t[WK_T*6-1:WK_T*5]);
mul_ori1[11] <= $signed(frame1t[WI_T*5-1:WI_T*4]) * $signed(weight1t[WK_T*5-1:WK_T*4]);
mul_ori1[12] <= $signed(frame1t[WI_T*4-1:WI_T*3]) * $signed(weight1t[WK_T*4-1:WK_T*3]);
mul_ori1[13] <= $signed(frame1t[WI_T*3-1:WI_T*2]) * $signed(weight1t[WK_T*3-1:WK_T*2]);
mul_ori1[14] <= $signed(frame1t[WI_T*2-1:WI_T*1]) * $signed(weight1t[WK_T*2-1:WK_T*1]);
mul_ori1[15] <= $signed(frame1t[WI_T*1-1:WI_T*0]) * $signed(weight1t[WK_T*1-1:WK_T*0]);
//f2*w2
mul_ori2[0]  <= $signed(frame2t[WI_T*16-1:WI_T*15]) * $signed(weight2t[WK_T*16-1:WK_T*15]);
mul_ori2[1]  <= $signed(frame2t[WI_T*15-1:WI_T*14]) * $signed(weight2t[WK_T*15-1:WK_T*14]);
mul_ori2[2]  <= $signed(frame2t[WI_T*14-1:WI_T*13]) * $signed(weight2t[WK_T*14-1:WK_T*13]);
mul_ori2[3]  <= $signed(frame2t[WI_T*13-1:WI_T*12]) * $signed(weight2t[WK_T*13-1:WK_T*12]);
mul_ori2[4]  <= $signed(frame2t[WI_T*12-1:WI_T*11]) * $signed(weight2t[WK_T*12-1:WK_T*11]);
mul_ori2[5]  <= $signed(frame2t[WI_T*11-1:WI_T*10]) * $signed(weight2t[WK_T*11-1:WK_T*10]);
mul_ori2[6]  <= $signed(frame2t[WI_T*10-1:WI_T*9])  * $signed(weight2t[WK_T*10-1:WK_T*9]);
mul_ori2[7]  <= $signed(frame2t[WI_T*9-1:WI_T*8]) * $signed(weight2t[WK_T*9-1:WK_T*8]);
mul_ori2[8]  <= $signed(frame2t[WI_T*8-1:WI_T*7]) * $signed(weight2t[WK_T*8-1:WK_T*7]);
mul_ori2[9]  <= $signed(frame2t[WI_T*7-1:WI_T*6]) * $signed(weight2t[WK_T*7-1:WK_T*6]);
mul_ori2[10] <= $signed(frame2t[WI_T*6-1:WI_T*5]) * $signed(weight2t[WK_T*6-1:WK_T*5]);
mul_ori2[11] <= $signed(frame2t[WI_T*5-1:WI_T*4]) * $signed(weight2t[WK_T*5-1:WK_T*4]);
mul_ori2[12] <= $signed(frame2t[WI_T*4-1:WI_T*3]) * $signed(weight2t[WK_T*4-1:WK_T*3]);
mul_ori2[13] <= $signed(frame2t[WI_T*3-1:WI_T*2]) * $signed(weight2t[WK_T*3-1:WK_T*2]);
mul_ori2[14] <= $signed(frame2t[WI_T*2-1:WI_T*1]) * $signed(weight2t[WK_T*2-1:WK_T*1]);
mul_ori2[15] <= $signed(frame2t[WI_T*1-1:WI_T*0]) * $signed(weight2t[WK_T*1-1:WK_T*0]);
//f3*w3
mul_ori3[0]  <= $signed(frame3t[WI_T*16-1:WI_T*15]) * $signed(weight3t[WK_T*16-1:WK_T*15]);
mul_ori3[1]  <= $signed(frame3t[WI_T*15-1:WI_T*14]) * $signed(weight3t[WK_T*15-1:WK_T*14]);
mul_ori3[2]  <= $signed(frame3t[WI_T*14-1:WI_T*13]) * $signed(weight3t[WK_T*14-1:WK_T*13]);
mul_ori3[3]  <= $signed(frame3t[WI_T*13-1:WI_T*12]) * $signed(weight3t[WK_T*13-1:WK_T*12]);
mul_ori3[4]  <= $signed(frame3t[WI_T*12-1:WI_T*11]) * $signed(weight3t[WK_T*12-1:WK_T*11]);
mul_ori3[5]  <= $signed(frame3t[WI_T*11-1:WI_T*10]) * $signed(weight3t[WK_T*11-1:WK_T*10]);
mul_ori3[6]  <= $signed(frame3t[WI_T*10-1:WI_T*9])  * $signed(weight3t[WK_T*10-1:WK_T*9]);
mul_ori3[7]  <= $signed(frame3t[WI_T*9-1:WI_T*8]) * $signed(weight3t[WK_T*9-1:WK_T*8]);
mul_ori3[8]  <= $signed(frame3t[WI_T*8-1:WI_T*7]) * $signed(weight3t[WK_T*8-1:WK_T*7]);
mul_ori3[9]  <= $signed(frame3t[WI_T*7-1:WI_T*6]) * $signed(weight3t[WK_T*7-1:WK_T*6]);
mul_ori3[10] <= $signed(frame3t[WI_T*6-1:WI_T*5]) * $signed(weight3t[WK_T*6-1:WK_T*5]);
mul_ori3[11] <= $signed(frame3t[WI_T*5-1:WI_T*4]) * $signed(weight3t[WK_T*5-1:WK_T*4]);
mul_ori3[12] <= $signed(frame3t[WI_T*4-1:WI_T*3]) * $signed(weight3t[WK_T*4-1:WK_T*3]);
mul_ori3[13] <= $signed(frame3t[WI_T*3-1:WI_T*2]) * $signed(weight3t[WK_T*3-1:WK_T*2]);
mul_ori3[14] <= $signed(frame3t[WI_T*2-1:WI_T*1]) * $signed(weight3t[WK_T*2-1:WK_T*1]);
mul_ori3[15] <= $signed(frame3t[WI_T*1-1:WI_T*0]) * $signed(weight3t[WK_T*1-1:WK_T*0]);
//f2*w1
mul_ori4[0]  <= $signed(frame2t[WI_T*16-1:WI_T*15]) * $signed(weight1t[WK_T*16-1:WK_T*15]);
mul_ori4[1]  <= $signed(frame2t[WI_T*15-1:WI_T*14]) * $signed(weight1t[WK_T*15-1:WK_T*14]);
mul_ori4[2]  <= $signed(frame2t[WI_T*14-1:WI_T*13]) * $signed(weight1t[WK_T*14-1:WK_T*13]);
mul_ori4[3]  <= $signed(frame2t[WI_T*13-1:WI_T*12]) * $signed(weight1t[WK_T*13-1:WK_T*12]);
mul_ori4[4]  <= $signed(frame2t[WI_T*12-1:WI_T*11]) * $signed(weight1t[WK_T*12-1:WK_T*11]);
mul_ori4[5]  <= $signed(frame2t[WI_T*11-1:WI_T*10]) * $signed(weight1t[WK_T*11-1:WK_T*10]);
mul_ori4[6]  <= $signed(frame2t[WI_T*10-1:WI_T*9])  * $signed(weight1t[WK_T*10-1:WK_T*9]);
mul_ori4[7]  <= $signed(frame2t[WI_T*9-1:WI_T*8]) * $signed(weight1t[WK_T*9-1:WK_T*8]);
mul_ori4[8]  <= $signed(frame2t[WI_T*8-1:WI_T*7]) * $signed(weight1t[WK_T*8-1:WK_T*7]);
mul_ori4[9]  <= $signed(frame2t[WI_T*7-1:WI_T*6]) * $signed(weight1t[WK_T*7-1:WK_T*6]);
mul_ori4[10] <= $signed(frame2t[WI_T*6-1:WI_T*5]) * $signed(weight1t[WK_T*6-1:WK_T*5]);
mul_ori4[11] <= $signed(frame2t[WI_T*5-1:WI_T*4]) * $signed(weight1t[WK_T*5-1:WK_T*4]);
mul_ori4[12] <= $signed(frame2t[WI_T*4-1:WI_T*3]) * $signed(weight1t[WK_T*4-1:WK_T*3]);
mul_ori4[13] <= $signed(frame2t[WI_T*3-1:WI_T*2]) * $signed(weight1t[WK_T*3-1:WK_T*2]);
mul_ori4[14] <= $signed(frame2t[WI_T*2-1:WI_T*1]) * $signed(weight1t[WK_T*2-1:WK_T*1]);
mul_ori4[15] <= $signed(frame2t[WI_T*1-1:WI_T*0]) * $signed(weight1t[WK_T*1-1:WK_T*0]);
//f3*w2
mul_ori5[0]  <= $signed(frame3t[WI_T*16-1:WI_T*15]) * $signed(weight2t[WK_T*16-1:WK_T*15]);
mul_ori5[1]  <= $signed(frame3t[WI_T*15-1:WI_T*14]) * $signed(weight2t[WK_T*15-1:WK_T*14]);
mul_ori5[2]  <= $signed(frame3t[WI_T*14-1:WI_T*13]) * $signed(weight2t[WK_T*14-1:WK_T*13]);
mul_ori5[3]  <= $signed(frame3t[WI_T*13-1:WI_T*12]) * $signed(weight2t[WK_T*13-1:WK_T*12]);
mul_ori5[4]  <= $signed(frame3t[WI_T*12-1:WI_T*11]) * $signed(weight2t[WK_T*12-1:WK_T*11]);
mul_ori5[5]  <= $signed(frame3t[WI_T*11-1:WI_T*10]) * $signed(weight2t[WK_T*11-1:WK_T*10]);
mul_ori5[6]  <= $signed(frame3t[WI_T*10-1:WI_T*9])  * $signed(weight2t[WK_T*10-1:WK_T*9]);
mul_ori5[7]  <= $signed(frame3t[WI_T*9-1:WI_T*8]) * $signed(weight2t[WK_T*9-1:WK_T*8]);
mul_ori5[8]  <= $signed(frame3t[WI_T*8-1:WI_T*7]) * $signed(weight2t[WK_T*8-1:WK_T*7]);
mul_ori5[9]  <= $signed(frame3t[WI_T*7-1:WI_T*6]) * $signed(weight2t[WK_T*7-1:WK_T*6]);
mul_ori5[10] <= $signed(frame3t[WI_T*6-1:WI_T*5]) * $signed(weight2t[WK_T*6-1:WK_T*5]);
mul_ori5[11] <= $signed(frame3t[WI_T*5-1:WI_T*4]) * $signed(weight2t[WK_T*5-1:WK_T*4]);
mul_ori5[12] <= $signed(frame3t[WI_T*4-1:WI_T*3]) * $signed(weight2t[WK_T*4-1:WK_T*3]);
mul_ori5[13] <= $signed(frame3t[WI_T*3-1:WI_T*2]) * $signed(weight2t[WK_T*3-1:WK_T*2]);
mul_ori5[14] <= $signed(frame3t[WI_T*2-1:WI_T*1]) * $signed(weight2t[WK_T*2-1:WK_T*1]);
mul_ori5[15] <= $signed(frame3t[WI_T*1-1:WI_T*0]) * $signed(weight2t[WK_T*1-1:WK_T*0]);
//f4*w3
mul_ori6[0]  <= $signed(frame4t[WI_T*16-1:WI_T*15]) * $signed(weight3t[WK_T*16-1:WK_T*15]);
mul_ori6[1]  <= $signed(frame4t[WI_T*15-1:WI_T*14]) * $signed(weight3t[WK_T*15-1:WK_T*14]);
mul_ori6[2]  <= $signed(frame4t[WI_T*14-1:WI_T*13]) * $signed(weight3t[WK_T*14-1:WK_T*13]);
mul_ori6[3]  <= $signed(frame4t[WI_T*13-1:WI_T*12]) * $signed(weight3t[WK_T*13-1:WK_T*12]);
mul_ori6[4]  <= $signed(frame4t[WI_T*12-1:WI_T*11]) * $signed(weight3t[WK_T*12-1:WK_T*11]);
mul_ori6[5]  <= $signed(frame4t[WI_T*11-1:WI_T*10]) * $signed(weight3t[WK_T*11-1:WK_T*10]);
mul_ori6[6]  <= $signed(frame4t[WI_T*10-1:WI_T*9])  * $signed(weight3t[WK_T*10-1:WK_T*9]);
mul_ori6[7]  <= $signed(frame4t[WI_T*9-1:WI_T*8]) * $signed(weight3t[WK_T*9-1:WK_T*8]);
mul_ori6[8]  <= $signed(frame4t[WI_T*8-1:WI_T*7]) * $signed(weight3t[WK_T*8-1:WK_T*7]);
mul_ori6[9]  <= $signed(frame4t[WI_T*7-1:WI_T*6]) * $signed(weight3t[WK_T*7-1:WK_T*6]);
mul_ori6[10] <= $signed(frame4t[WI_T*6-1:WI_T*5]) * $signed(weight3t[WK_T*6-1:WK_T*5]);
mul_ori6[11] <= $signed(frame4t[WI_T*5-1:WI_T*4]) * $signed(weight3t[WK_T*5-1:WK_T*4]);
mul_ori6[12] <= $signed(frame4t[WI_T*4-1:WI_T*3]) * $signed(weight3t[WK_T*4-1:WK_T*3]);
mul_ori6[13] <= $signed(frame4t[WI_T*3-1:WI_T*2]) * $signed(weight3t[WK_T*3-1:WK_T*2]);
mul_ori6[14] <= $signed(frame4t[WI_T*2-1:WI_T*1]) * $signed(weight3t[WK_T*2-1:WK_T*1]);
mul_ori6[15] <= $signed(frame4t[WI_T*1-1:WI_T*0]) * $signed(weight3t[WK_T*1-1:WK_T*0]);
end

//矩阵相乘结果截取	
//14b*9b=23b	[19:0]([21:0])->[19:8]; [19:8]=>[WM_O-1:WM_O-WO_T]=>[11:0]
always@(posedge clk)
begin
//f1*w1 cut
mul_cut1[0]  <= mul_ori1[0][WM_O-1:WM_O-WO_T];
mul_cut1[1]  <= mul_ori1[1][WM_O-1:WM_O-WO_T];
mul_cut1[2]  <= mul_ori1[2][WM_O-1:WM_O-WO_T];
mul_cut1[3]  <= mul_ori1[3][WM_O-1:WM_O-WO_T];
mul_cut1[4]  <= mul_ori1[4][WM_O-1:WM_O-WO_T];
mul_cut1[5]  <= mul_ori1[5][WM_O-1:WM_O-WO_T];
mul_cut1[6]  <= mul_ori1[6][WM_O-1:WM_O-WO_T];
mul_cut1[7]  <= mul_ori1[7][WM_O-1:WM_O-WO_T];
mul_cut1[8]  <= mul_ori1[8][WM_O-1:WM_O-WO_T];
mul_cut1[9]  <= mul_ori1[9][WM_O-1:WM_O-WO_T];
mul_cut1[10] <= mul_ori1[10][WM_O-1:WM_O-WO_T];
mul_cut1[11] <= mul_ori1[11][WM_O-1:WM_O-WO_T];
mul_cut1[12] <= mul_ori1[12][WM_O-1:WM_O-WO_T];
mul_cut1[13] <= mul_ori1[13][WM_O-1:WM_O-WO_T];
mul_cut1[14] <= mul_ori1[14][WM_O-1:WM_O-WO_T];
mul_cut1[15] <= mul_ori1[15][WM_O-1:WM_O-WO_T];
//f2*w2 cut
mul_cut2[0]  <= mul_ori2[0][WM_O-1:WM_O-WO_T];
mul_cut2[1]  <= mul_ori2[1][WM_O-1:WM_O-WO_T];
mul_cut2[2]  <= mul_ori2[2][WM_O-1:WM_O-WO_T];
mul_cut2[3]  <= mul_ori2[3][WM_O-1:WM_O-WO_T];
mul_cut2[4]  <= mul_ori2[4][WM_O-1:WM_O-WO_T];
mul_cut2[5]  <= mul_ori2[5][WM_O-1:WM_O-WO_T];
mul_cut2[6]  <= mul_ori2[6][WM_O-1:WM_O-WO_T];
mul_cut2[7]  <= mul_ori2[7][WM_O-1:WM_O-WO_T];
mul_cut2[8]  <= mul_ori2[8][WM_O-1:WM_O-WO_T];
mul_cut2[9]  <= mul_ori2[9][WM_O-1:WM_O-WO_T];
mul_cut2[10] <= mul_ori2[10][WM_O-1:WM_O-WO_T];
mul_cut2[11] <= mul_ori2[11][WM_O-1:WM_O-WO_T];
mul_cut2[12] <= mul_ori2[12][WM_O-1:WM_O-WO_T];
mul_cut2[13] <= mul_ori2[13][WM_O-1:WM_O-WO_T];
mul_cut2[14] <= mul_ori2[14][WM_O-1:WM_O-WO_T];
mul_cut2[15] <= mul_ori2[15][WM_O-1:WM_O-WO_T];
//f3*w3 cut
mul_cut3[0]  <= mul_ori3[0][WM_O-1:WM_O-WO_T];
mul_cut3[1]  <= mul_ori3[1][WM_O-1:WM_O-WO_T];
mul_cut3[2]  <= mul_ori3[2][WM_O-1:WM_O-WO_T];
mul_cut3[3]  <= mul_ori3[3][WM_O-1:WM_O-WO_T];
mul_cut3[4]  <= mul_ori3[4][WM_O-1:WM_O-WO_T];
mul_cut3[5]  <= mul_ori3[5][WM_O-1:WM_O-WO_T];
mul_cut3[6]  <= mul_ori3[6][WM_O-1:WM_O-WO_T];
mul_cut3[7]  <= mul_ori3[7][WM_O-1:WM_O-WO_T];
mul_cut3[8]  <= mul_ori3[8][WM_O-1:WM_O-WO_T];
mul_cut3[9]  <= mul_ori3[9][WM_O-1:WM_O-WO_T];
mul_cut3[10] <= mul_ori3[10][WM_O-1:WM_O-WO_T];
mul_cut3[11] <= mul_ori3[11][WM_O-1:WM_O-WO_T];
mul_cut3[12] <= mul_ori3[12][WM_O-1:WM_O-WO_T];
mul_cut3[13] <= mul_ori3[13][WM_O-1:WM_O-WO_T];
mul_cut3[14] <= mul_ori3[14][WM_O-1:WM_O-WO_T];
mul_cut3[15] <= mul_ori3[15][WM_O-1:WM_O-WO_T];
//f2*w1 cut
mul_cut4[0]  <= mul_ori4[0][WM_O-1:WM_O-WO_T];
mul_cut4[1]  <= mul_ori4[1][WM_O-1:WM_O-WO_T];
mul_cut4[2]  <= mul_ori4[2][WM_O-1:WM_O-WO_T];
mul_cut4[3]  <= mul_ori4[3][WM_O-1:WM_O-WO_T];
mul_cut4[4]  <= mul_ori4[4][WM_O-1:WM_O-WO_T];
mul_cut4[5]  <= mul_ori4[5][WM_O-1:WM_O-WO_T];
mul_cut4[6]  <= mul_ori4[6][WM_O-1:WM_O-WO_T];
mul_cut4[7]  <= mul_ori4[7][WM_O-1:WM_O-WO_T];
mul_cut4[8]  <= mul_ori4[8][WM_O-1:WM_O-WO_T];
mul_cut4[9]  <= mul_ori4[9][WM_O-1:WM_O-WO_T];
mul_cut4[10] <= mul_ori4[10][WM_O-1:WM_O-WO_T];
mul_cut4[11] <= mul_ori4[11][WM_O-1:WM_O-WO_T];
mul_cut4[12] <= mul_ori4[12][WM_O-1:WM_O-WO_T];
mul_cut4[13] <= mul_ori4[13][WM_O-1:WM_O-WO_T];
mul_cut4[14] <= mul_ori4[14][WM_O-1:WM_O-WO_T];
mul_cut4[15] <= mul_ori4[15][WM_O-1:WM_O-WO_T];
//f3*w2 cut
mul_cut5[0]  <= mul_ori5[0][WM_O-1:WM_O-WO_T];
mul_cut5[1]  <= mul_ori5[1][WM_O-1:WM_O-WO_T];
mul_cut5[2]  <= mul_ori5[2][WM_O-1:WM_O-WO_T];
mul_cut5[3]  <= mul_ori5[3][WM_O-1:WM_O-WO_T];
mul_cut5[4]  <= mul_ori5[4][WM_O-1:WM_O-WO_T];
mul_cut5[5]  <= mul_ori5[5][WM_O-1:WM_O-WO_T];
mul_cut5[6]  <= mul_ori5[6][WM_O-1:WM_O-WO_T];
mul_cut5[7]  <= mul_ori5[7][WM_O-1:WM_O-WO_T];
mul_cut5[8]  <= mul_ori5[8][WM_O-1:WM_O-WO_T];
mul_cut5[9]  <= mul_ori5[9][WM_O-1:WM_O-WO_T];
mul_cut5[10] <= mul_ori5[10][WM_O-1:WM_O-WO_T];
mul_cut5[11] <= mul_ori5[11][WM_O-1:WM_O-WO_T];
mul_cut5[12] <= mul_ori5[12][WM_O-1:WM_O-WO_T];
mul_cut5[13] <= mul_ori5[13][WM_O-1:WM_O-WO_T];
mul_cut5[14] <= mul_ori5[14][WM_O-1:WM_O-WO_T];
mul_cut5[15] <= mul_ori5[15][WM_O-1:WM_O-WO_T];
//f4*w3 cut
mul_cut6[0]  <= mul_ori6[0][WM_O-1:WM_O-WO_T];
mul_cut6[1]  <= mul_ori6[1][WM_O-1:WM_O-WO_T];
mul_cut6[2]  <= mul_ori6[2][WM_O-1:WM_O-WO_T];
mul_cut6[3]  <= mul_ori6[3][WM_O-1:WM_O-WO_T];
mul_cut6[4]  <= mul_ori6[4][WM_O-1:WM_O-WO_T];
mul_cut6[5]  <= mul_ori6[5][WM_O-1:WM_O-WO_T];
mul_cut6[6]  <= mul_ori6[6][WM_O-1:WM_O-WO_T];
mul_cut6[7]  <= mul_ori6[7][WM_O-1:WM_O-WO_T];
mul_cut6[8]  <= mul_ori6[8][WM_O-1:WM_O-WO_T];
mul_cut6[9]  <= mul_ori6[9][WM_O-1:WM_O-WO_T];
mul_cut6[10] <= mul_ori6[10][WM_O-1:WM_O-WO_T];
mul_cut6[11] <= mul_ori6[11][WM_O-1:WM_O-WO_T];
mul_cut6[12] <= mul_ori6[12][WM_O-1:WM_O-WO_T];
mul_cut6[13] <= mul_ori6[13][WM_O-1:WM_O-WO_T];
mul_cut6[14] <= mul_ori6[14][WM_O-1:WM_O-WO_T];
mul_cut6[15] <= mul_ori6[15][WM_O-1:WM_O-WO_T];
end

//OFMAP_TRANS1	(A')*M
//A' = [1 1 1 0; 0 1 -1 -1]
always@(posedge clk)
begin
//f1*w1 ofmp_trans1
oft1_1[0] <= mul_cut1[0] + mul_cut1[4]  + mul_cut1[8];
oft1_1[1] <= mul_cut1[1] + mul_cut1[5]  + mul_cut1[9];
oft1_1[2] <= mul_cut1[2] + mul_cut1[6]  + mul_cut1[10];
oft1_1[3] <= mul_cut1[3] + mul_cut1[7]  + mul_cut1[11];
oft1_1[4] <= mul_cut1[4] - mul_cut1[8]  - mul_cut1[12];
oft1_1[5] <= mul_cut1[5] - mul_cut1[9]  - mul_cut1[13];
oft1_1[6] <= mul_cut1[6] - mul_cut1[10] - mul_cut1[14];
oft1_1[7] <= mul_cut1[7] - mul_cut1[11] - mul_cut1[15];
//f2*w2 ofmp_trans1
oft1_2[0] <= mul_cut2[0] + mul_cut2[4]  + mul_cut2[8];
oft1_2[1] <= mul_cut2[1] + mul_cut2[5]  + mul_cut2[9];
oft1_2[2] <= mul_cut2[2] + mul_cut2[6]  + mul_cut2[10];
oft1_2[3] <= mul_cut2[3] + mul_cut2[7]  + mul_cut2[11];
oft1_2[4] <= mul_cut2[4] - mul_cut2[8]  - mul_cut2[12];
oft1_2[5] <= mul_cut2[5] - mul_cut2[9]  - mul_cut2[13];
oft1_2[6] <= mul_cut2[6] - mul_cut2[10] - mul_cut2[14];
oft1_2[7] <= mul_cut2[7] - mul_cut2[11] - mul_cut2[15];
//f3*w3 ofmp_trans1
oft1_3[0] <= mul_cut3[0] + mul_cut3[4]  + mul_cut3[8];
oft1_3[1] <= mul_cut3[1] + mul_cut3[5]  + mul_cut3[9];
oft1_3[2] <= mul_cut3[2] + mul_cut3[6]  + mul_cut3[10];
oft1_3[3] <= mul_cut3[3] + mul_cut3[7]  + mul_cut3[11];
oft1_3[4] <= mul_cut3[4] - mul_cut3[8]  - mul_cut3[12];
oft1_3[5] <= mul_cut3[5] - mul_cut3[9]  - mul_cut3[13];
oft1_3[6] <= mul_cut3[6] - mul_cut3[10] - mul_cut3[14];
oft1_3[7] <= mul_cut3[7] - mul_cut3[11] - mul_cut3[15];
//f2*w1 ofmp_trans1
oft1_4[0] <= mul_cut4[0] + mul_cut4[4]  + mul_cut4[8];
oft1_4[1] <= mul_cut4[1] + mul_cut4[5]  + mul_cut4[9];
oft1_4[2] <= mul_cut4[2] + mul_cut4[6]  + mul_cut4[10];
oft1_4[3] <= mul_cut4[3] + mul_cut4[7]  + mul_cut4[11];
oft1_4[4] <= mul_cut4[4] - mul_cut4[8]  - mul_cut4[12];
oft1_4[5] <= mul_cut4[5] - mul_cut4[9]  - mul_cut4[13];
oft1_4[6] <= mul_cut4[6] - mul_cut4[10] - mul_cut4[14];
oft1_4[7] <= mul_cut4[7] - mul_cut4[11] - mul_cut4[15];
//f3*w2 ofmp_trans1
oft1_5[0] <= mul_cut5[0] + mul_cut5[4]  + mul_cut5[8];
oft1_5[1] <= mul_cut5[1] + mul_cut5[5]  + mul_cut5[9];
oft1_5[2] <= mul_cut5[2] + mul_cut5[6]  + mul_cut5[10];
oft1_5[3] <= mul_cut5[3] + mul_cut5[7]  + mul_cut5[11];
oft1_5[4] <= mul_cut5[4] - mul_cut5[8]  - mul_cut5[12];
oft1_5[5] <= mul_cut5[5] - mul_cut5[9]  - mul_cut5[13];
oft1_5[6] <= mul_cut5[6] - mul_cut5[10] - mul_cut5[14];
oft1_5[7] <= mul_cut5[7] - mul_cut5[11] - mul_cut5[15];
//f4*w3 ofmp_trans1
oft1_6[0] <= mul_cut6[0] + mul_cut6[4]  + mul_cut6[8];
oft1_6[1] <= mul_cut6[1] + mul_cut6[5]  + mul_cut6[9];
oft1_6[2] <= mul_cut6[2] + mul_cut6[6]  + mul_cut6[10];
oft1_6[3] <= mul_cut6[3] + mul_cut6[7]  + mul_cut6[11];
oft1_6[4] <= mul_cut6[4] - mul_cut6[8]  - mul_cut6[12];
oft1_6[5] <= mul_cut6[5] - mul_cut6[9]  - mul_cut6[13];
oft1_6[6] <= mul_cut6[6] - mul_cut6[10] - mul_cut6[14];
oft1_6[7] <= mul_cut6[7] - mul_cut6[11] - mul_cut6[15];
end

//OFMAP_TRANS2	(A')*M*A
//A = [1 0; 1 1; 1 -1; 0 -1]
always@(posedge clk)
begin
//f1*w1 ofmp_trans2
oft2_1[0] <= oft1_1[0] + oft1_1[1] + oft1_1[2];
oft2_1[1] <= oft1_1[1] - oft1_1[2] - oft1_1[3];
oft2_1[2] <= oft1_1[4] + oft1_1[5] + oft1_1[6];
oft2_1[3] <= oft1_1[5] - oft1_1[6] - oft1_1[7];
//f2*w2 ofmp_trans2
oft2_2[0] <= oft1_2[0] + oft1_2[1] + oft1_2[2];
oft2_2[1] <= oft1_2[1] - oft1_2[2] - oft1_2[3];
oft2_2[2] <= oft1_2[4] + oft1_2[5] + oft1_2[6];
oft2_2[3] <= oft1_2[5] - oft1_2[6] - oft1_2[7];
//f3*w3 ofmp_trans2
oft2_3[0] <= oft1_3[0] + oft1_3[1] + oft1_3[2];
oft2_3[1] <= oft1_3[1] - oft1_3[2] - oft1_3[3];
oft2_3[2] <= oft1_3[4] + oft1_3[5] + oft1_3[6];
oft2_3[3] <= oft1_3[5] - oft1_3[6] - oft1_3[7];
//f2*w1 ofmp_trans2
oft2_4[0] <= oft1_4[0] + oft1_4[1] + oft1_4[2];
oft2_4[1] <= oft1_4[1] - oft1_4[2] - oft1_4[3];
oft2_4[2] <= oft1_4[4] + oft1_4[5] + oft1_4[6];
oft2_4[3] <= oft1_4[5] - oft1_4[6] - oft1_4[7];
//f3*w2 ofmp_trans2
oft2_5[0] <= oft1_5[0] + oft1_5[1] + oft1_5[2];
oft2_5[1] <= oft1_5[1] - oft1_5[2] - oft1_5[3];
oft2_5[2] <= oft1_5[4] + oft1_5[5] + oft1_5[6];
oft2_5[3] <= oft1_5[5] - oft1_5[6] - oft1_5[7];
//f4*w3 ofmp_trans2
oft2_6[0] <= oft1_6[0] + oft1_6[1] + oft1_6[2];
oft2_6[1] <= oft1_6[1] - oft1_6[2] - oft1_6[3];
oft2_6[2] <= oft1_6[4] + oft1_6[5] + oft1_6[6];
oft2_6[3] <= oft1_6[5] - oft1_6[6] - oft1_6[7];
end

//FRAME_SUM ifmp三帧相加得ofmp一帧
always@(posedge clk or negedge reset)
begin
if(!reset)
	begin
	frame1_sum[0] <= 0;
	frame1_sum[1] <= 0;
	frame1_sum[2] <= 0;
	frame1_sum[3] <= 0;
	//f2w1_t+f3w3_t+f4w3_t
	frame2_sum[0] <= 0;
	frame2_sum[1] <= 0;
	frame2_sum[2] <= 0;
	frame2_sum[3] <= 0;
	end
else
	begin
	//f1w1_t+f2w2_t+f3w3_t
	frame1_sum[0] <= oft2_1[0] + oft2_2[0] + oft2_3[0];
	frame1_sum[1] <= oft2_1[1] + oft2_2[1] + oft2_3[1];
	frame1_sum[2] <= oft2_1[2] + oft2_2[2] + oft2_3[2];
	frame1_sum[3] <= oft2_1[3] + oft2_2[3] + oft2_3[3];
	//f2w1_t+f3w3_t+f4w3_t
	frame2_sum[0] <= oft2_4[0] + oft2_5[0] + oft2_6[0];
	frame2_sum[1] <= oft2_4[1] + oft2_5[1] + oft2_6[1];
	frame2_sum[2] <= oft2_4[2] + oft2_5[2] + oft2_6[2];
	frame2_sum[3] <= oft2_4[3] + oft2_5[3] + oft2_6[3];
	end
end

//channel累加	
//第一个channel与conv_bias相加,忽略自身保存的原值; 后续channel将frame_sum的值与自身保存的原值相累加
always@(posedge clk or negedge reset)
begin
if(!reset)
	begin
	chan_sum1[0] <= 0;
	chan_sum1[1] <= 0;
	chan_sum1[2] <= 0;
	chan_sum1[3] <= 0;
	
	chan_sum2[0] <= 0;
	chan_sum2[1] <= 0;
	chan_sum2[2] <= 0;
	chan_sum2[3] <= 0;
	end
else
	begin
	if(conv_chan_start_cur)
		begin
		chan_sum1[0] <= frame1_sum[0] + conv_bias_cur;
		chan_sum1[1] <= frame1_sum[1] + conv_bias_cur;
		chan_sum1[2] <= frame1_sum[2] + conv_bias_cur;
		chan_sum1[3] <= frame1_sum[3] + conv_bias_cur;
		
		chan_sum2[0] <= frame2_sum[0] + conv_bias_cur;
		chan_sum2[1] <= frame2_sum[1] + conv_bias_cur;	
		chan_sum2[2] <= frame2_sum[2] + conv_bias_cur;
		chan_sum2[3] <= frame2_sum[3] + conv_bias_cur;
		end
	else
		begin
		chan_sum1[0] <= frame1_sum[0] + chan_sum1[0];
		chan_sum1[1] <= frame1_sum[1] + chan_sum1[1];
		chan_sum1[2] <= frame1_sum[2] + chan_sum1[2];
		chan_sum1[3] <= frame1_sum[3] + chan_sum1[3];
		
		chan_sum2[0] <= frame2_sum[0] + chan_sum2[0];
		chan_sum2[1] <= frame2_sum[1] + chan_sum2[1];
		chan_sum2[2] <= frame2_sum[2] + chan_sum2[2];
		chan_sum2[3] <= frame2_sum[3] + chan_sum2[3];
		end
	end
end

//relu
always@(posedge clk)
begin
if(chan_sum1[0] > 0)	relu1[0] <= chan_sum1[0];
else	relu1[0] <= 0;
if(chan_sum1[1] > 0)	relu1[1] <= chan_sum1[1];
else	relu1[1] <= 0;
if(chan_sum1[2] > 0)	relu1[2] <= chan_sum1[2];
else	relu1[2] <= 0;
if(chan_sum1[3] > 0)	relu1[3] <= chan_sum1[3];
else	relu1[3] <= 0;

if(chan_sum2[0] > 0)	relu2[0] <= chan_sum2[0];
else	relu2[0] <= 0;
if(chan_sum2[1] > 0)	relu2[1] <= chan_sum2[1];
else	relu2[1] <= 0;
if(chan_sum2[2] > 0)	relu2[2] <= chan_sum2[2];
else	relu2[2] <= 0;
if(chan_sum2[3] > 0)	relu2[3] <= chan_sum2[3];
else	relu2[3] <= 0;
end

//pooling
//8->4->2->1
//组合逻辑完成第一级pooling
always@(*)
begin
if(relu1[0] > relu1[1])	pool_fr1_1 <= relu1[0];
else	pool_fr1_1 <= relu1[1];
if(relu1[2] > relu1[3])	pool_fr1_2 <= relu1[2];
else	pool_fr1_2 <= relu1[3];

if(relu2[0] > relu2[1])	pool_fr2_1 <= relu2[0];
else	pool_fr2_1 <= relu2[1];
if(relu2[2] > relu2[3])	pool_fr2_2 <= relu2[2];
else	pool_fr2_2 <= relu2[3];
end

//时序逻辑完成第二级pooling
always@(posedge clk)
begin
if(pool_fr1_1 > pool_fr1_2)	pool_fr1 <= pool_fr1_1;
else	pool_fr1 <= pool_fr1_2;

if(pool_fr2_1 > pool_fr2_2)	pool_fr2 <= pool_fr2_1;
else	pool_fr2 <= pool_fr2_2;
end

//时序逻辑完成第三级pooling,根据pooling_mode赋值
always@(posedge clk)
begin
if(pooling_mode_cur)
	begin
	pooling1 <= pool_fr1;
	pooling2 <= pool_fr2;
	end
else
	begin
	if(pool_fr1 > pool_fr2)	
		begin
		pooling1 <= pool_fr1;
		pooling2 <= 0;
		end
	else
		begin
		pooling1 <= pool_fr2;
		pooling2 <= 0;
		end
	end
end

//时序逻辑完成pooling结果的截取，根据out_fix_mode做截取，输出reset复位
always@(posedge clk or negedge reset)
begin
if(!reset)
	begin
	ofmp_frame1 <= 0;
	ofmp_frame2 <= 0;
	end
else
	begin
	if(out_fix_mode_cur)
		begin
		ofmp_frame1 <= pooling1[WCH-2:1];
		ofmp_frame2 <= pooling2[WCH-2:1];
		end
	else//layer2
		begin
		ofmp_frame1 <= pooling1[WCH-1:2];
		ofmp_frame2 <= pooling2[WCH-1:2];
		end
	end
end

endmodule
