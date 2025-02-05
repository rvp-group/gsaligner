#include "aligner_utils.cuh"

__global__ void init_photo_image_from_buffer_kernel(const float* buf, photo::Image* img, const int rows, const int cols) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= rows || col >= cols)
    return;

  const float dval = buf[row * cols + col];
  if (dval > 1e-5)
    img->at<1>(row, col).setMasked(false);
  img->at<1>(row, col).setDepth(dval);

  // printf("row: %d, col: %d, dval: %f, masked=%d\t", row, col, dval, img->at<1>(row, col).masked());
}

void init_photo_image_from_buffer(const torch::Tensor buf, photo::Image& img, const int rows, const int cols) {
  init_photo_image_from_buffer_kernel<<<dim3((cols + 16 - 1) / 16, (rows + 16 - 1) / 16), dim3(16, 16)>>>(
    buf.contiguous().data<float>(), img.deviceInstance(), rows, cols);
}