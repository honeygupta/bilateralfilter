__kernel void hello_kernel( __global float *Fbar ,__global float *F, __global int *a,__global float *g_s)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
  int x = 256;
	int y = 256;
  
	float accumulation = 0;
	float	weightsum = 0;
	int row,col;
	float k;

			for (row = 0; row < a[0] - 1; row++)
				{
					for (col = 0; col < a[0] - 1; col++)
					{
						if (((i + row)< x) && ((j + col)<y))
						{
							k = F[(i + row)*x + (j + col)];
							accumulation += k * g_s[(row*a[0])+col];
							weightsum += g_s[(row*a[0]) + col];
						}
					}
				}
				Fbar[(i*x) + j] = accumulation / weightsum;
}
