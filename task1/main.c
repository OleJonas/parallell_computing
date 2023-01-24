#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

//#include <windows.h>
//#include <magick_wand.h>

typedef struct{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} pixel;

int main(int argc, char** argv)
{
    stbi_set_flip_vertically_on_load(true);
	stbi_flip_vertically_on_write(true);

	int width;
	int height;
	int channels;
    unsigned char* char_pixels_1 = stbi_load(argv[1], &width, &height, &channels, STBI_rgb_alpha);
    unsigned char* char_pixels_2 = stbi_load(argv[2], &width, &height, &channels, STBI_rgb_alpha);

    //TODO 2 - typecast pointer
    pixel* pixels_1 = (pixel*)char_pixels_1;
    pixel* pixels_2 = (pixel*)char_pixels_2;

    printf("height:%d, width: %d\n", height, width);
    if (pixels_1 == NULL || pixels_2 == NULL)
    {
        exit(1);
    }

    int dimensions = height*width;
    //TODO 3 - malloc
    pixel* pixels_out = (pixel*)malloc(dimensions*sizeof(pixel));

    //TODO 4 - loop
    //Write your loop here
    pixel avg_pixel(pixel one, pixel two){
        pixel out_pixel;
        out_pixel.r = (one.r + two.r)/2;
        out_pixel.g = (one.g + two.g)/2;
        out_pixel.b = (one.b + two.b)/2;
        out_pixel.a = 100;

        return out_pixel;
    }

    for(int i = 0; i < dimensions; i++){
        pixels_out[i] = avg_pixel(pixels_1[i], pixels_2[i]);
    }

    stbi_write_png("output.png", width, height, STBI_rgb_alpha, pixels_out, sizeof(pixel) * width);

    //TODO 5 - free
    free(pixels_out);
    free(char_pixels_1);
    free(char_pixels_2);

    return 0;
}
