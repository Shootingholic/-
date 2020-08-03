#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_math.h"
#include<ap_fixed.h>
//typedef ap_fixed<8,0,AP_RND, AP_SAT> data_t;
typedef ap_fixed<16,8,AP_RND, AP_SAT> data_t;
typedef ap_axiu<32,1,1,1> AXI_VAL;
// --------------------------------------------------------------------
// function to be accelerated in HW wrapped with AXI4-Stream interface

// --------------------------------------------------------
// functions to insert and extract elements from an axi stream
// includes conversion to correct data type

int Axi_Transfer(AXI_VAL* in_data, AXI_VAL* out_data, int value, int loop)
{
	int Temproray;
	Temproray= in_data->data;
	if (loop==1)
	{
		out_data->data=Temproray;
	}else
		out_data->data=value;
	out_data->dest = in_data->dest;
	out_data->id = in_data->id;
	out_data->keep = in_data->keep;
	out_data->last = in_data->last;
	out_data->strb = in_data->strb;
	out_data->user = in_data->user;
	return Temproray;
}

//-----------------------------------------------------------------
// Main function

void cnn (AXI_VAL* in_data, AXI_VAL* out_data)
{
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis      port=in_data
#pragma HLS INTERFACE axis      port=out_data

cnn_label0:while(true)
{
	data_t Input[120000],Weight[40000],Bias[1280];
	float Temproray,Precision,Transfer_value,Convolve_value,Pool_Value, BN_Value;
	int H_Result,W_Result,Index,Index2,Parameters[18],Counter,R_Plane,R_Row,\
	Relu_Activation,Load_Input,Load_Weight,Stride_Size[2],Window_Size[2],\
	Filter_size[4],Input_Size[3],Bias_Activation,Pooling_Mode,dilia_rate;

	// Get Module initial parameters

	for(int idx=0; idx< (18); idx++)
		{
		Transfer_value=0;
		Parameters[idx]= Axi_Transfer(in_data, out_data,Transfer_value,1);
	}


	//Convoulution

	// Parameters In convolution { 0-Module selection,1-Input size, 2-Input D, 3-Input H, 4-Input W, 5-Filter N,6-
	// Filter D,7- Filter H,8- Filter W,9- stride H,10- Stride w,11- padding,12- bias size,13- Relu_Activation, 14-precision 15-Load_Input 16-Load_Weight 17-dilia_rate}

	if(Parameters[0]==0)
	{
		Relu_Activation=Parameters[13]; // Relu Activation
		Bias_Activation=Parameters[12]; // Bias Activation
		Load_Input=Parameters[15]; // Activate receiver to get new Input
		Load_Weight=Parameters[16]; // Activate receiver to get new Weights
		Stride_Size[0]=Parameters[9]; // Stride H
		Stride_Size[1]=Parameters[10]; // Stride W
		Filter_size[0]=Parameters[5];  // Number of Filters (output planes)
		Filter_size[1]=Parameters[6];  // Filter Depth
		Filter_size[2]=Parameters[7];  // Filter H
		Filter_size[3]=Parameters[8];  // Filter W
		Input_Size[0]=Parameters[2];  // Input Depth
		Input_Size[1]=Parameters[3];  // Input H
		Input_Size[2]=Parameters[4];  // Input W
		dilia_rate = Parameters[17];	// dilia_rate for a_conv, default=1


		Precision=Parameters[14];
		// Get Input Tensor
		if(Load_Input==1)
		{
		for (int idx=0; idx<Parameters[1];idx++)
		{
			Temproray= Axi_Transfer(in_data, out_data,1,0);
			Input[idx]= (data_t)(Temproray/Precision);
		}
		}
		if(Load_Weight==1)
		{
		// Get Bias (if there is Bias)
		if(Bias_Activation==1)
		{
			for (int idx=0; idx<Filter_size[0];idx++)
			{
				Transfer_value=2;
				Temproray= Axi_Transfer(in_data, out_data,Transfer_value,0);
				Bias[idx]= (data_t)(Temproray/Precision);
			}
		}

		// Layer weights load
		for (int idx=0; idx<(Filter_size[0]*Filter_size[1]*Filter_size[2]*Filter_size[3]);idx++)
		{
			Transfer_value=3;
			Temproray= Axi_Transfer(in_data, out_data,Transfer_value,0);;
			Weight[idx]= (data_t)(Temproray/Precision);

		}
		}

		H_Result=int(((Input_Size[1]-Filter_size[2])/Stride_Size[0])+1); // calculate Output dimension
		W_Result=int(((Input_Size[2]-Filter_size[3])/Stride_Size[1])+1); // calculate Output dimension

		// Send output Result to CPU
	    Temproray=(Filter_size[0]*W_Result*H_Result);
	    Axi_Transfer(in_data, out_data,Temproray,0);
	    Axi_Transfer(in_data, out_data,H_Result,0);
	    Axi_Transfer(in_data, out_data,W_Result,0);


	    // Main Convolution
		// Parameters In convolution { 0-Module selection,1-Input size, 2-Input D, 3-Input H, 4-Input W, 5-Filter N,6-
		// Filter D,7- Filter H,8- Filter W,9- stride H,10- Stride w,11- padding,12- bias size,13- Relu_Activation, 14-precision 15-Load_Input 16-Load_Weight}
	    for (int idx=0; idx<Filter_size[0];idx++)
	        {
	        for(int idx2=0; idx2<H_Result;idx2++)
	            {
	            for(int idx3=0; idx3<W_Result;idx3++)
	                {
	            	Index2= (idx*(H_Result*W_Result))+(idx2*W_Result)+idx3; // store location of convolution result
	            	Convolve_value=0;
#pragma HLS PIPELINE II=1
	            	for(int k=0; k<Filter_size[1];k++)
	            	{
	            		R_Plane=(k*(Input_Size[1]*Input_Size[2])); // find plane of element for multiplication
						for(int i=0; i<Filter_size[2];i++)
							{
							// dilia_rate default = 1, else is a_covn
							R_Row=R_Plane+(((idx2*(Stride_Size[0]))+i*dilia_rate)*Input_Size[2]);  // find Row of element for multiplication
							for(int j=0; j<Filter_size[3];j++)
								{
									Index=R_Row+(idx3*Stride_Size[1])+j*dilia_rate;  // find Input element for multiplication
									// Multiply and accumulate
									Convolve_value=Convolve_value+(float)Input[Index]*(float)Weight[((idx*(Filter_size[1]*Filter_size[2]*Filter_size[3]))+(k*Filter_size[2]*Filter_size[3])+(i*Filter_size[3])+j)];

								}
							}
	            	}
	            	if(Bias_Activation!=0)
	            	{
						//bias

	            		Convolve_value=Convolve_value+(float)Bias[idx];
	            	}
	            	if(Relu_Activation==1)
	            	{
	            		if (Convolve_value < 0) Convolve_value=0;
	            	}

	            	Convolve_value=Convolve_value*Precision;
	            	Axi_Transfer(in_data, out_data,ap_uint<32>(Convolve_value),0); // Return Result to CPU
	                }
	            }
	        }
	       }// End of Convolution

	    //POOLING

	    // Parameters In POOLING{ 0-Module selection,1-Input size, 2-Input D, 3-Input H, 4-Input W, 5-Pooling window H,6-
	    	// Pooling window W,7- stride H,8- Stride W,9- Pooling Type {0:max , 1: Average},10- padding,11- Relu_Activation, 12-precision, 13-Load_Input }


	if(Parameters[0]==1)
	{


		Input_Size[0]=Parameters[2];  // Input Depth
		Input_Size[1]=Parameters[3];  // Input H
		Input_Size[2]=Parameters[4];  // Input W
		Window_Size[0]=Parameters[5];  // pooling window H
		Window_Size[1]=Parameters[6];  // pooling window W
		Stride_Size[0]=Parameters[7]; // Stride H
		Stride_Size[1]=Parameters[8]; // Stride W
		Pooling_Mode=Parameters[9]; // pooling Mode 0: Max , 1:Average
		Relu_Activation=Parameters[11]; // Relu Activation
		Precision=Parameters[12];
		Load_Input=Parameters[13]; // Activate receiver to get new Input


		// Get Input Tensor
		if(Load_Input==1)
		{
		for (int idx=0; idx<Parameters[1];idx++)
		{
			Temproray= Axi_Transfer(in_data, out_data,1,0);
			Input[idx]= (data_t)Temproray;
		}
		}

		H_Result=int(((Input_Size[1])/Stride_Size[0])); // calculate Output dimension
		W_Result=int(((Input_Size[2])/Stride_Size[1])); // calculate Output dimension

		// Send output Result to CPU
	    Temproray=(Input_Size[0]*W_Result*H_Result);
	    Axi_Transfer(in_data, out_data,Temproray,0);
	    Axi_Transfer(in_data, out_data,H_Result,0);
	    Axi_Transfer(in_data, out_data,W_Result,0);

	    // Pooling Function
	    for (int idx=0; idx<Input_Size[0];idx++)
	        {
	        for(int idx2=0; idx2<H_Result;idx2++)
	            {
	            for(int idx3=0; idx3<W_Result;idx3++)
	                {
	            	Pool_Value=0;
#pragma HLS PIPELINE II=1
	            	for(int k=0; k<Window_Size[0];k++)
	            	{
						for(int i=0; i<Window_Size[1];i++)
							{
							// Maximum Pooling Function
							if(Pooling_Mode==0)
							{
							if(k==0 && i==0)
								{
								Pool_Value=(float)Input[(idx*(Input_Size[1]*Input_Size[2]))+(idx2*(Stride_Size[0])*Input_Size[2])+ (idx3*Stride_Size[1])];
								}
								else
								{

									Temproray=(float)Input[(idx*(Input_Size[1]*Input_Size[2]))+((idx2*(Stride_Size[0])+k)*Input_Size[2])+(idx3*Stride_Size[1])+i];
									if (Temproray>Pool_Value) Pool_Value=Temproray;
								}
							}
							// Average Pooling Function
							if(Pooling_Mode==1)
							{
							if(k==0 && i==0)
								{
								Pool_Value=Pool_Value+(float)Input[(idx*(Input_Size[1]*Input_Size[2]))+((idx2*(Stride_Size[0])+k)*Input_Size[2])+(idx3*Stride_Size[1])+i];
								}
							}
	            	}
	            	}
					if(Pooling_Mode==1)
					{
						Pool_Value=(Pool_Value/(Window_Size[0]*Window_Size[0]));
					}
	            	if(Relu_Activation==1)
	            	{
	            		if (Pool_Value < 0) Pool_Value=0;
	            	}
	            	Axi_Transfer(in_data, out_data,ap_uint<32>(Pool_Value),0); // Return Result to CPU
	                }
	            }
	        }

			}// End of Pooling

	//Fully Connected
    // Parameters In Fully Connected{ 0-Module selection,1-Input size, 2-Output size 3- Relu_Activation, 4-precision, 5-Load_Input, 6- Bias Activation }
	if(Parameters[0]==2)
		{



			Input_Size[0]=Parameters[1];  // Input Depth
			Relu_Activation=Parameters[3]; // Relu Activation
			Precision=Parameters[4];
			Load_Input=Parameters[5]; // Activate receiver to get new Input
			Bias_Activation=Parameters[6]; // Bias Activation


			// Get Input Tensor
			if(Load_Input==1)
			{
			for (int idx=0; idx<Input_Size[0];idx++)
			{
				Temproray= Axi_Transfer(in_data, out_data,Input_Size[0],0);
				Input[idx]= (float)(Temproray/Precision);
			}
			}

			// Get Bias (if there is Bias)
			if(Bias_Activation==1)
			{
				for (int idx=0; idx<Parameters[2];idx++)
				{
					Transfer_value=2;
					Temproray= Axi_Transfer(in_data, out_data,Transfer_value,1);
					Bias[idx]= (data_t)(Temproray/Precision);
				}
			}

		    for (int idx=0; idx<Parameters[2];idx++)
		        {
		    	Transfer_value=0;
		        for(int idx2=0; idx2<Input_Size[0];idx2++)
		            {
		        		Temproray= Axi_Transfer(in_data, out_data,4,0);
		        		Temproray=Temproray/Precision;
		        		Transfer_value=Transfer_value+ (float)Input[idx2]*Temproray;
		            }
				if(Relu_Activation==1)
				{
					if (Transfer_value < 0) Transfer_value=0;
				}
				if(Bias_Activation==1)
				{
					Transfer_value=Transfer_value+(float)Bias[idx];
				}
				Transfer_value=Transfer_value*Precision;
				Weight[idx]= (data_t)Transfer_value;
				}
		    for(int idx=0; idx<Parameters[2];idx++)
		    {
		    	Axi_Transfer(in_data, out_data,ap_uint<32>((float)Weight[idx]),0); // Return Result to CPU
		    }
		       }// End of Fully Connected Layer

		// BN-Layer
		// Parameters in BN {0-Module selection,1-Input size, 2-Input D, 3-Input H, 4-Input W(dont need to change shape after bn),
		// 5-Load_input, 6-Load_Paras(save to Bias[1500], size upper bound = Input D * 5, max for fingernet = 256*5), 7-Precision, 8-pReLu}

		if(Parameters[0]==3){
			Input_Size[0] = Parameters[2];	// D
			Input_Size[1] = Parameters[3];	// H
			Input_Size[2] = Parameters[4];	// W
			Load_Input = Parameters[5];
			Load_Weight = Parameters[6];
			Precision = Parameters[7];
			Relu_Activation=Parameters[8];

			if(Load_Input == 1){
				for(int idx=0; idx < Parameters[1]; idx++){
					Temproray = Axi_Transfer(in_data, out_data, 1, 0);
					Input[idx] = (data_t)Temproray;
				}
			}
			if(Load_Weight == 1){
				// store gamma[0~D-1], beta[D~(2D-1)], mean[2D~3D-1] and variance[3D~4D-1] plus p[4D~4D-1]in Bias[] sequentially
				for(int idx=0; idx < Input_Size[0]*5; idx++){
					Transfer_value = 2; // indicate writing Bias array
					Temproray = Axi_Transfer(in_data, out_data, Transfer_value, 1);
					Bias[idx] = (data_t)(Temproray / Precision);
				}
			}

			// Send output Result to CPU
			Temproray=Parameters[1];
			H_Result = Input_Size[1];
			W_Result = Input_Size[2];
			Axi_Transfer(in_data, out_data, (ap_uint<32>)Temproray,0);
			Axi_Transfer(in_data, out_data, H_Result,0);
			Axi_Transfer(in_data, out_data, W_Result,0);

			// actual Batch-Norm
			for(int d = 0; d < Input_Size[0]; d ++){
				BN_Value = 0;
				for(int h = 0; h < Input_Size[1]; h++){
					for(int w = 0; w < Input_Size[2]; w++){
						BN_Value = (float)Bias[d] * (((float)Input[d*(Input_Size[0]*Input_Size[1]*Input_Size[2])+h*(Input_Size[1]*Input_Size[2])+w*Input_Size[2]]-(float)Bias[2*Input_Size[0]+d])/(sqrt(float(Bias[3*Input_Size[0]+d]+(data_t)0.1)))) \
								+ (float)Bias[Input_Size[0]+d];
						if(Relu_Activation == 1)
							BN_Value = BN_Value > 0 ? BN_Value : BN_Value * (float)Bias[4*Input_Size[0]+d];
						BN_Value = BN_Value * Precision;
						Axi_Transfer(in_data, out_data, ap_uint<32>((float)BN_Value), 0);
					}
				}

//				Weight[((idx*(Filter_size[1]*Filter_size[2]*Filter_size[3]))+(k*Filter_size[2]*Filter_size[3])+(i*Filter_size[3])+j)];
			}

		} // end of BN

		} // End of Main While

}







