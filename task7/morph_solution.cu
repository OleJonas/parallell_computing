#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <cmath>

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glut.h>
#endif

using namespace std;

//the pixel
typedef struct pix{
  unsigned char r,g,b,a;
} pixel;

typedef struct SimplePoint_struct {
        float x, y;
        
} SimplePoint;


typedef struct SimpleFeatureLine_struct {
      SimplePoint startPoint;
        SimplePoint endPoint;
        
} SimpleFeatureLine;


template <typename T> 
__host__ __device__ T CLAMP(T value, T low, T high)
{
        return (value < low) ? low : ((value > high) ? high : value);
}

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int imgWidthOrig, imgHeightOrig, imgWidthDest, imgHeightDest;
int steps;
float p,a,b,t;
pixel* hSrcImgMap;
pixel* hDstImgMap;

//The name of input and output files
string inputFileSrc;
string inputFileDest;
string inputFileLines;
string outputPath;
string tempFile;
string dataPath;
string stepsStr;
string pStr, aStr, bStr, tStr;

void imgRead(string filename, pixel * &map, int &imgW, int &imgH){
    stbi_set_flip_vertically_on_load(true);

    int x, y, componentsPerPixel;
    if( ! filename.empty()  ){
        map = (pixel *) stbi_load(filename.c_str(), &x, &y, &componentsPerPixel, STBI_rgb_alpha);
    } else{
        cout<<"The input file name cannot be empty"<<endl;
        exit(1);
    }

    // get the current image columns and rows
    imgW = x;
    imgH = y;

    cout<<"Read the image file \""<<filename<<"\" successfully ."<<endl;
}

void imgWrite(string filename, pixel * map, int imgW, int imgH){
    if(filename.empty()){
        cout<<"The output file name cannot be empty"<<endl;
        exit(1);
    }
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filename.c_str(), imgW, imgH, STBI_rgb_alpha, map, sizeof(pixel) * imgW);
    cout<<"Write the image into \""<<filename<<"\" file successfully."<<endl;
}

void loadLines(SimpleFeatureLine** linesSrc, SimpleFeatureLine** linesDst, int* linesLen, const char* name) {
	FILE *f = fopen(name, "r");
	if (f == NULL)
	{
		printf("Error opening file %s! \n", name);
		exit(1);
	}
	fscanf(f, "%d", linesLen);
	SimpleFeatureLine* srcLines = (SimpleFeatureLine*) malloc(sizeof(SimpleFeatureLine)*(*linesLen));
	SimpleFeatureLine* dstLines = (SimpleFeatureLine*) malloc(sizeof(SimpleFeatureLine)*(*linesLen));
	SimpleFeatureLine* line;
	int fac = 2;

	for(int i = 0; i < (*linesLen)*fac; i++) {
		line = (i % fac) ? &dstLines[(i-1)/fac] : &srcLines[i/fac];
		fscanf(f, "%f,%f,%f,%f", 
				&(line->startPoint.x), &(line->startPoint.y), 
				&(line->endPoint.x), &(line->endPoint.y));
	}

	*linesSrc = srcLines;
	*linesDst = dstLines;
}


// Parse commandline arguments
void parse(int argc, char *argv[]) {
    p = 0;
    a = 1;
    b = 2;
    t = 0.5;
    steps = 90;
    switch(argc){
        case 2:
            dataPath = argv[1];
        case 6:
            inputFileSrc = argv[1];
            inputFileDest = argv[2];
            inputFileLines = argv[3];
            outputPath = argv[4];
            stepsStr = argv[5];
            istringstream ( stepsStr ) >> steps;
            imgRead(inputFileSrc, hSrcImgMap, imgWidthOrig, imgHeightOrig);
            imgRead(inputFileDest, hDstImgMap, imgWidthDest, imgHeightDest);
            break;

        case 9:
            inputFileSrc = argv[1];
            inputFileDest = argv[2];
            inputFileLines = argv[3];
            outputPath = argv[4];
            stepsStr = argv[5];
            pStr = argv[6];
            aStr = argv[7];
            bStr = argv[8];
            istringstream ( stepsStr ) >> steps;
            istringstream ( pStr ) >> p;
            istringstream ( aStr ) >> a;
            istringstream ( bStr ) >> b;
            imgRead(inputFileSrc, hSrcImgMap, imgWidthOrig, imgHeightOrig);
            imgRead(inputFileDest, hDstImgMap, imgWidthDest, imgHeightDest);
            break;

        default:
            cout<<"Usage:"<<endl;
            cout<<"./morph srcImg.png destImg.png lines.txt outputPath steps [p] [a] [b]"<<endl;
            exit(1); 
    }

}


void simpleLineInterpolate(SimpleFeatureLine* sourceLines, 
                     SimpleFeatureLine* destLines , SimpleFeatureLine** morphLines, int linesLen, float t)
{
	SimpleFeatureLine* interLines = (SimpleFeatureLine*) malloc(sizeof(SimpleFeatureLine)*linesLen);
	for(int i=0; i<linesLen; i++){
		interLines[i].startPoint.x = (1-t)*(sourceLines[i].startPoint.x) + t*(destLines[i].startPoint.x);
		interLines[i].startPoint.y = (1-t)*(sourceLines[i].startPoint.y) + t*(destLines[i].startPoint.y);
		interLines[i].endPoint.x = (1-t)*(sourceLines[i].endPoint.x) + t*(destLines[i].endPoint.x);
		interLines[i].endPoint.y = (1-t)*(sourceLines[i].endPoint.y) + t*(destLines[i].endPoint.y);
	}
	*morphLines = interLines;
}




/* warping function (backward mapping)
   input:
   interPt = the point in the intermediary image
   interLines = given line in the intermediary image
   srcLines = given line in the source image
   p, a, b = parameters of the weight function
   output:
   src = the corresponding point */
__host__ __device__ void warp(const SimplePoint* interPt, SimpleFeatureLine* interLines,
          SimpleFeatureLine* sourceLines, const int sourceLinesSize, SimplePoint* src)
{
	int i;
	float interLength, srcLength;
	float weight, weightSum, dist;
	float sum_x, sum_y; // weighted sum of the coordination of the point "src"
	float u, v;
	SimplePoint pd, pq, qd;
	float X, Y;

	sum_x = 0;
	sum_y = 0;
	weightSum = 0;

	for (i=0; i<sourceLinesSize; i++) {
		pd.x = interPt->x - interLines[i].startPoint.x;
		pd.y = interPt->y - interLines[i].startPoint.y;
		pq.x = interLines[i].endPoint.x - interLines[i].startPoint.x;
		pq.y = interLines[i].endPoint.y - interLines[i].startPoint.y;
		interLength = pq.x * pq.x + pq.y * pq.y;
		u = (pd.x * pq.x + pd.y * pq.y) / interLength;

		interLength = sqrt(interLength); // length of the vector PQ

		v = (pd.x * pq.y - pd.y * pq.x) / interLength;

		pq.x = sourceLines[i].endPoint.x - sourceLines[i].startPoint.x;
		pq.y = sourceLines[i].endPoint.y - sourceLines[i].startPoint.y;

		srcLength = sqrt(pq.x * pq.x + pq.y * pq.y); // length of the vector P'Q'
		// corresponding point based on the ith line
		X = sourceLines[i].startPoint.x + u * pq.x + v * pq.y / srcLength;
		Y = sourceLines[i].startPoint.y + u * pq.y - v * pq.x / srcLength;

		// the distance from the corresponding point to the line P'Q'
		if (u < 0)
			dist = sqrt(pd.x * pd.x + pd.y * pd.y);
		else if (u > 1) {
			qd.x = interPt->x - interLines[i].endPoint.x;
			qd.y = interPt->y - interLines[i].endPoint.y;
			dist = sqrt(qd.x * qd.x + qd.y * qd.y);
		}else{
			dist = abs(v);
		}

		weight = pow(1.0f / (1.0f + dist), 2.0f);
		sum_x += X * weight;
		sum_y += Y * weight;
		weightSum += weight;
	}

	src->x = sum_x / weightSum;
	src->y = sum_y / weightSum;
}

__host__ __device__ void bilinear(pixel* Im, float row, float col, pixel* pix, int dImgWidth)
{
	int cm, cn, fm, fn;
	double alpha, beta;

	cm = (int)ceil(row);
	fm = (int)floor(row);
	cn = (int)ceil(col);
	fn = (int)floor(col);
	alpha = ceil(row) - row;
	beta = ceil(col) - col;

	pix->r = (unsigned int)( alpha*beta*Im[fm*dImgWidth+fn].r
			+ (1-alpha)*beta*Im[cm*dImgWidth+fn].r 
			+ alpha*(1-beta)*Im[fm*dImgWidth+cn].r
			+ (1-alpha)*(1-beta)*Im[cm*dImgWidth+cn].r );
	pix->g = (unsigned int)( alpha*beta*Im[fm*dImgWidth+fn].g
			+ (1-alpha)*beta*Im[cm*dImgWidth+fn].g 
			+ alpha*(1-beta)*Im[fm*dImgWidth+cn].g
			+ (1-alpha)*(1-beta)*Im[cm*dImgWidth+cn].g );
	pix->b = (unsigned int)( alpha*beta*Im[fm*dImgWidth+fn].b
			+ (1-alpha)*beta*Im[cm*dImgWidth+fn].b 
			+ alpha*(1-beta)*Im[fm*dImgWidth+cn].b
			+ (1-alpha)*(1-beta)*Im[cm*dImgWidth+cn].b );
	pix->a = 255;
}

__host__ __device__ void ColorInterPolate(const SimplePoint* Src_P, 
                      const SimplePoint* Dest_P, float t, 
                      pixel* imgSrc, pixel* imgDest, pixel* rgb, int dImgWidth)
{
    pixel srcColor, destColor;

    bilinear(imgSrc, Src_P->y, Src_P->x, &srcColor, dImgWidth);
    bilinear(imgDest, Dest_P->y, Dest_P->x, &destColor, dImgWidth);

    rgb->b = srcColor.b*(1-t)+ destColor.b*t;
    rgb->g = srcColor.g*(1-t)+ destColor.g*t;
    rgb->r = srcColor.r*(1-t)+ destColor.r*t;
    rgb->a = 255;
}


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///// DO NOT TOUCH CODE ABOVE. YOU ONLY NEED TO CHANGE THE FUNCTIONS BELOW ////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

// TODO 1 b: Change to kernel
__global__
void morphKernel(SimpleFeatureLine* dSrcLines, SimpleFeatureLine* dDstLines, SimpleFeatureLine* dMorphLines, 
		pixel* dSrcImgMap, pixel* dDstImgMap,  pixel* dMorphMap, 
		int linesLen, int dImgWidth, int dImgHeight, float dT) {
	// TODO 2 c: Implement a shared memory solution.
	// Make space for feature lines in shared memory since this are often accessed data -> speedup :)
	extern __shared__ SimpleFeatureLine shared_mem[];
	SimpleFeatureLine* dMLines = &shared_mem[0];
	SimpleFeatureLine* dSLines = &shared_mem[linesLen];
	SimpleFeatureLine* dDLines = &shared_mem[linesLen*2];

	// TODO 1 c: Parallelize kernel

	pixel interColor;
	SimplePoint dest;
	SimplePoint src;
	SimplePoint q;

	// Kernel is parallelised by replacing the double for-loop with a thread index for every thread.
	// Because we have n_threads >= n_pixels, we can reach every index of the pixel array
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= dImgHeight || j >= dImgWidth) // Check if overshooting image dims
		return;
	q.x = j;
	q.y = i;

	// Let the linesLen first threads of each block copy feature lines into shared memory
	int blockTID = threadIdx.x * blockDim.x + threadIdx.y;
	if(blockTID < linesLen){
		dSLines[blockTID] = dSrcLines[blockTID];
		dDLines[blockTID] = dDstLines[blockTID];
		dMLines[blockTID] = dMorphLines[blockTID];
	}
	__syncthreads();

	warp(&q, dMLines, dSLines, linesLen, &src);
	warp(&q, dMLines, dDLines, linesLen, &dest);

	src.x = CLAMP<double>(src.x, 0, dImgWidth-1);
	src.y = CLAMP<double>(src.y, 0, dImgHeight-1);
	dest.x = CLAMP<double>(dest.x, 0, dImgWidth-1);
	dest.y = CLAMP<double>(dest.y, 0, dImgHeight-1);

	// color interpolation
	ColorInterPolate(&src, &dest, dT, dSrcImgMap, dDstImgMap, &interColor, dImgWidth);

	dMorphMap[i*dImgWidth+j].r = interColor.r;
	dMorphMap[i*dImgWidth+j].g = interColor.g;
	dMorphMap[i*dImgWidth+j].b = interColor.b;
	dMorphMap[i*dImgWidth+j].a = interColor.a;
}

int main(int argc,char *argv[]){

	// Setup ////////////////// 
	parse(argc, argv);
	tempFile = outputPath;
	float stepSize = 1.0/steps;

	int linesLen;
	SimpleFeatureLine *hSrcLines, *hDstLines;
	loadLines(&hSrcLines, &hDstLines, &linesLen, inputFileLines.c_str());
	printf("Loaded %d lines\n", linesLen);

	pixel** hMorphMapArr = (pixel**) malloc(sizeof(pixel*) * (steps+1));
	SimpleFeatureLine** hMorphLinesArr = (SimpleFeatureLine**) malloc(sizeof(SimpleFeatureLine*)*(steps+1));

	for (int i = 0; i < steps+1; i++) {
		hMorphMapArr[i] = (pixel*) malloc(sizeof(pixel)*imgHeightOrig*imgWidthOrig);
		simpleLineInterpolate(hSrcLines, hDstLines, &(hMorphLinesArr[i]), linesLen, t);
	}
	///////////////////////////

	int dImgWidth = imgHeightOrig; // 1024
	int dImgHeight = imgWidthOrig; // 1024
	int imgDimsOrig = dImgHeight * dImgWidth;

	// TODO: 1 e: CUDA setup
	// Allocate space for feature lines
	int linesToBytes = sizeof(SimpleFeatureLine)*linesLen;

	SimpleFeatureLine *dSrcLines, *dDstLines, *dMorphLines;
	cudaMalloc(&dSrcLines, linesToBytes);
	cudaMalloc(&dDstLines, linesToBytes);
	cudaMalloc(&dMorphLines, linesToBytes);


	// Allocate space for image maps
	int pixelsToBytes = sizeof(pixel)*imgDimsOrig;

	pixel *dSrcImgMap, *dDstImgMap, *dMorphMap;
	cudaMalloc(&dSrcImgMap, pixelsToBytes);
	cudaMalloc(&dDstImgMap, pixelsToBytes);
	cudaMalloc(&dMorphMap, pixelsToBytes);


	// Copy data from host to device
	cudaMemcpy(dSrcLines, hSrcLines, linesToBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dDstLines, hDstLines, linesToBytes, cudaMemcpyHostToDevice);

	cudaMemcpy(dSrcImgMap, hSrcImgMap, pixelsToBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dDstImgMap, hDstImgMap, pixelsToBytes, cudaMemcpyHostToDevice);


	// TODO: 3 a: Occupancy API call
	int bSize;   // The launch configurator returned block size 
	int minGSize; // The minimum grid size needed to achieve the 
					// maximum occupancy for a full device launch 

	cudaOccupancyMaxPotentialBlockSize( &minGSize, &bSize, 
											morphKernel, 0, 0); 

	cudaDeviceSynchronize();
	printf("blocksize: %d, minGSize: %d\n", bSize, minGSize); // This prints a bSize of 1024 -> max number of threads per block. Testing with this shows that 8x8 is still faster than 1024 = 32x32 (1.5ms faster)
	// TODO: 3 b: Define the 2D block size
	// int blockDim = ceil(sqrt(bSize));
	// printf("blockDim: %d\n", blockDim);
	
	// TODO: 2 a: Define shared-memory size
	// Define launch params for kernel
	int blockDim = 8;
	dim3 blockSize(blockDim, blockDim);
	dim3 gridSize((dImgWidth+blockDim-1)/blockDim, (dImgHeight+blockDim-1)/blockDim);
	
	int shared_mem_size = sizeof(SimpleFeatureLine)*linesLen*3; // Space for the three sets of feature lines


	// Timing code
	cudaEvent_t start_total, stop_total;
	cudaErrorCheck(cudaEventCreate(&start_total));
	cudaErrorCheck(cudaEventCreate(&stop_total));
	cudaErrorCheck( cudaEventRecord(start_total, 0));

	// Computes a morphed image for each step based on hMorphLinesArr[i]. 
	// The morphed image is saved in hMorphMapArr[i];
	for (int i = 0; i < steps+1; i++) {
		t = stepSize*i;
		float dT = t;

		/* TODO: 1 a: CUDA malloc and memcpy */
		
		// Copy variables from host to device
		// Feature-lines
		cudaMemcpy(dMorphLines, hMorphLinesArr[i], linesToBytes, cudaMemcpyHostToDevice);

		// Img-maps
		cudaMemcpy(dMorphMap, hMorphMapArr[i], pixelsToBytes, cudaMemcpyHostToDevice);

		/* TODO: 1 a: END */

		// TODO: 1 b: block and grid size definition
		// Moved all this to 2a instead

		// Timing code
		float elapsed=0;
		cudaEvent_t start, stop;
		cudaErrorCheck(cudaEventCreate(&start));
		cudaErrorCheck(cudaEventCreate(&stop));
		cudaErrorCheck(cudaEventRecord(start, 0));


		// TODO 1 b: Launch kernel. 
		// For 2 b you will need to change the launch parameters.
		// Launching kernel with defined grid- and blocksize. Also setting a dynamic shared memory size.
		morphKernel<<<gridSize, blockSize, shared_mem_size>>>(
			dSrcLines, dDstLines, dMorphLines, 
			dSrcImgMap, dDstImgMap, dMorphMap,
			linesLen, dImgWidth, dImgHeight, dT
		);

		// Timing code
		cudaErrorCheck(cudaEventRecord(stop, 0));
		cudaErrorCheck(cudaEventSynchronize (stop) );
		cudaErrorCheck(cudaEventElapsedTime(&elapsed, start, stop) );
		cudaErrorCheck(cudaEventDestroy(start));
		cudaErrorCheck(cudaEventDestroy(stop));
		printf("Time in morphKernel (step %d): %.2f ms\n", i, elapsed);

		// TODO 1 d: Copy data back to host from GPU. Save the morphed image to hMorphMapArr[i]. 
		// Do not write the image out to file. This is done afterwards.
		cudaMemcpy(hMorphMapArr[i], dMorphMap, pixelsToBytes, cudaMemcpyDeviceToHost);

	} 

	// Timing code
	float elapsed_total = 0;
	cudaErrorCheck(cudaDeviceSynchronize());
	cudaErrorCheck(cudaEventRecord(stop_total, 0));
	cudaErrorCheck(cudaEventSynchronize (stop_total) );
	cudaErrorCheck(cudaEventElapsedTime(&elapsed_total, start_total, stop_total) );
	cudaErrorCheck(cudaEventDestroy(start_total));
	cudaErrorCheck(cudaEventDestroy(stop_total));
	printf("Total time in GPU: %.2f ms\n", elapsed_total);
	// Serial solution takes 38388ms 10 steps, 104481ms 30 steps
	// Current best: 120ms 10 steps, 340ms 30 steps
	// Seems to be speedup of around 470 times faster in the kernel itself, but the entire program has a speedup of ca. 310-320 times varying between executions

	// Write morphed images to files
	for (int i = 0; i < steps+1; i++) {
		t = stepSize*i;
    		outputPath = tempFile + "output-" + to_string(t) + "-cuda.png";
    		imgWrite(outputPath, hMorphMapArr[i], imgWidthOrig, imgHeightOrig);
		free(hMorphMapArr[i]);
		free(hMorphLinesArr[i]);
	}

	free(hMorphMapArr);
	free(hMorphLinesArr);

	// TODO 1 d: cudaFree the heap-allocated memory
	cudaFree(dDstImgMap);
	cudaFree(dDstLines);
	cudaFree(dSrcImgMap);
	cudaFree(dSrcLines);

	return 0;
}

