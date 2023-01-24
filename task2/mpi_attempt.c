#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <signal.h>
#include "mpi.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

typedef struct pixel_struct {
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
} pixel;

const int MAX_STR_LEN = 10;


//--------------------------------------------------------------------------------------------------
//--------------------------bilinear interpolation--------------------------------------------------
//--------------------------------------------------------------------------------------------------
void bilinear(pixel* Im, float row, float col, pixel* pix, int width, int height)
{
	int cm, cn, fm, fn;
	double alpha, beta;

	cm = (int)ceil(row);
	fm = (int)floor(row);
	cn = (int)ceil(col);
	fn = (int)floor(col);
	alpha = ceil(row) - row;
	beta = ceil(col) - col;

	pix->r = (unsigned char)(alpha*beta*Im[fm*width+fn].r
			+ (1-alpha)*beta*Im[cm*width+fn].r
			+ alpha*(1-beta)*Im[fm*width+cn].r
			+ (1-alpha)*(1-beta)*Im[cm*width+cn].r );
	pix->g = (unsigned char)(alpha*beta*Im[fm*width+fn].g
			+ (1-alpha)*beta*Im[cm*width+fn].g
			+ alpha*(1-beta)*Im[fm*width+cn].g
			+ (1-alpha)*(1-beta)*Im[cm*width+cn].g );
	pix->b = (unsigned char)(alpha*beta*Im[fm*width+fn].b
			+ (1-alpha)*beta*Im[cm*width+fn].b
			+ alpha*(1-beta)*Im[fm*width+cn].b
			+ (1-alpha)*(1-beta)*Im[cm*width+cn].b );
	pix->a = 255;
}
//---------------------------------------------------------------------------

//Helper function to locate the source of errors
void
SEGVFunction( int sig_num)
{
	printf ("\n Signal %d received\n",sig_num);
	exit(sig_num);
} 

int main(int argc, char** argv)
{
	signal(SIGSEGV, SEGVFunction);
	stbi_set_flip_vertically_on_load(true);
	stbi_flip_vertically_on_write(true);

//TODO 1 - init
    int comm_size;
    int rank;
	char msg[MAX_STR_LEN];
	int ranks[comm_size];

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(rank != 0){
		sprintf(msg, "%d", rank);
		MPI_Send(msg, strlen(msg)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	} else{
		for(int i = 1; i < comm_size; i++){
			MPI_Recv(msg, MAX_STR_LEN, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

//TODO END


	pixel* pixels_in;

	int in_width;
	int in_height;
	int channels;


//TODO 2 - broadcast

	const int pixel_attributes = 4;
	int blocklens[4] = {1,1,1,1};
	MPI_Aint offsets[4];
	MPI_Datatype types[4] = {MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR};
	offsets[0] = offsetof(pixel, r);
	offsets[1] = offsetof(pixel, g);
	offsets[2] = offsetof(pixel, b);
	offsets[3] = offsetof(pixel, a);
	
	MPI_Datatype mpi_pixel_type;
	MPI_Type_create_struct(pixel_attributes, blocklens, offsets, types, &mpi_pixel_type);
    MPI_Type_commit(&mpi_pixel_type);

	if(rank == 0){
		pixels_in = (pixel *) stbi_load(argv[1], &in_width, &in_height, &channels, STBI_rgb_alpha);
		if (pixels_in == NULL) {
			exit(1);
		}
		printf("Image dimensions: %dx%d\n", in_width, in_height);
	}
	
	// This could probably be compacted into one bcast with an array / tuple for the img dimensions...
	MPI_Bcast(&in_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&in_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(rank != 0)
		pixels_in = (pixel *) malloc(in_height*in_width*sizeof(pixel));
	MPI_Bcast(pixels_in, (in_width*in_height), mpi_pixel_type, 0, MPI_COMM_WORLD);	

//TODO END


	double scale_x = argc > 2 ? atof(argv[2]): 2;
	double scale_y = argc > 3 ? atof(argv[3]): 8;

	int out_width = in_width * scale_x;
	int out_height = in_height * scale_y;
	
//TODO 3 - partitioning
	int local_out_height = out_height/comm_size;
	pixel* local_out = (pixel *) malloc(sizeof(pixel) * (out_width * local_out_height));
//TODO END


//TODO 4 - computation
	int loc_out_h_start = local_out_height*rank;
	for(int i = loc_out_h_start; i < loc_out_h_start+local_out_height; i++) {
		for(int j = 0; j < out_width; j++) {
			pixel new_pixel;

			float row = i * (in_height) / (float)out_height;
			float col = j * (in_width) / (float)out_width;
			bilinear(pixels_in, row, col, &new_pixel, in_width, in_height);

			int px_x = ((i-loc_out_h_start)*out_width) + j;
			local_out[px_x] = new_pixel;
		}
	}
//TODO END



//TODO 5 - gather
	pixel* pixels_out;
	if(rank == 0){
		pixels_out = (pixel *) malloc(out_height*out_width*sizeof(pixel));	}

	MPI_Gather(local_out, out_width*local_out_height, mpi_pixel_type, pixels_out, out_width*local_out_height, mpi_pixel_type, 0, MPI_COMM_WORLD);
	MPI_Type_free(&mpi_pixel_type);

	if(rank == 0){
		stbi_write_png("outputye.png", out_width, out_height, STBI_rgb_alpha, pixels_out, sizeof(pixel) * out_width);
		printf("Image with dimensions x: %i, y: %i\nUpscaled to new dimensions x: %i, y: %i\n\n", in_width, in_height, out_width, out_height);
	}


//TODO END


//TODO 1 - init
//TODO END
	free(local_out);
	free(pixels_in);
	if(rank == 0)
		free(pixels_out);

	MPI_Finalize();
	
	return 0;
}
