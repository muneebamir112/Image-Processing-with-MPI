#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "ppmFile.h"

int main(int argc, char *argv[]) {
    int rank, size;
    Image *image = NULL;
    int width, height;
    unsigned char *chunk, *fullData;
    int chunkSize, fullSize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        image = ImageRead("input.ppm");
        width = ImageWidth(image);
        height = ImageHeight(image);
        fullData = image->data;
        fullSize = width * height * 3;
        if (fullSize % size != 0) {
            fprintf(stderr, "Image size must be divisible by number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast width and height to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    chunkSize = (width * height * 3) / size;
    chunk = (unsigned char *) malloc(chunkSize);

    // Scatter image data to all processes
    MPI_Scatter(fullData, chunkSize, MPI_UNSIGNED_CHAR,
                chunk, chunkSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Example operation: clear each chunk to red (255, 0, 0)
    for (int i = 0; i < chunkSize; i += 3) {
        chunk[i] = 255;      // Red
        chunk[i + 1] = 0;    // Green
        chunk[i + 2] = 0;    // Blue
    }

    // Gather the processed chunks back
    MPI_Gather(chunk, chunkSize, MPI_UNSIGNED_CHAR,
               fullData, chunkSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        ImageWrite(image, "output.ppm");
    }

    free(chunk);
    if (rank == 0) {
        free(image->data);
        free(image);
    }

    MPI_Finalize();
    return 0;
}
