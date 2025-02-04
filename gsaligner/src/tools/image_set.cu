#include "image_set.cuh"

namespace photo {
  
  void ImageSet::getIntensity(Matrixf& intensity) const {
    intensity.resize(rows(), cols());
    intensity.fill(0);
    for (size_t k = 0; k < matrix_.size(); ++k)
      intensity.at(k) = matrix_.at(k).intensity();
  }

  void ImageSet::getDepth(Matrixf& depth) const {
    depth.resize(rows(), cols());
    depth.fill(0);
    for (size_t k = 0; k < matrix_.size(); ++k)
      depth.at(k) = matrix_.at(k).depth();
  }



  // TODO fix redundant arguments
  __global__ void toCloud_kernel(MatrixCloud* target,
                                 const Image* mat,
                                 const Eigen::Isometry3f sensor_offset,
                                 const Eigen::Matrix3f inv_K,
                                 const CameraType cam_type,
                                 const float ifx,
                                 const float ify,
                                 const float cx,
                                 const float cy,
                                 const float min_depth,
                                 const float max_depth) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (!target->inside(row, col))
      return;

    auto& dst = target->at<1>(row, col);
    const ImageEntry src        = mat->at<1>(row, col);

    float w = src.depth();
    if (src.masked() || w < min_depth || w > max_depth)
      return;
#ifdef _MD_ENABLE_SUPERRES_
    const float& r = src.r;
    const float& c = src.c;
#else
    const float r = row;
    const float c = col;
#endif
    dst.status = PointStatusFlag::Good;
    switch (cam_type) {
      case Pinhole: {
        dst.coordinates = inv_K * Eigen::Vector3f(c * w, r * w, w);
      } break;
      case Spherical: {
        float azimuth        = ifx * (c - cx);
        float elevation      = ify * (r - cy);
        float s0             = sinf(azimuth);
        float c0             = cosf(azimuth);
        float s1             = sinf(elevation);
        float c1             = cosf(elevation);
        dst.coordinates(0) = c0 * c1 * w;
        dst.coordinates(1) = s0 * c1 * w;
        dst.coordinates(2) = s1 * w;
      } break;
      default:;
    }
    dst.intensity   = src.intensity();
    dst.normal      = sensor_offset.linear() * src.normal();
    dst.coordinates = sensor_offset * dst.coordinates;
  }

  void ImageSet::toCloudDevice(MatrixCloud* target) const {
    target->resize(rows(), cols());

    Pointf p; // initialize to zeros
    p.status = PointStatusFlag::Invalid;

    // TODO only in device?
    target->fill(p);

    const auto& camera_matrix = sensor_->cameraMatrix();
    const float ifx           = 1.f / camera_matrix(0, 0);
    const float ify           = 1.f / camera_matrix(1, 1);
    const float cx            = camera_matrix(0, 2);
    const float cy            = camera_matrix(1, 2);
    const Eigen::Matrix3f inv_K      = camera_matrix.inverse();

    // init bidimensional kernel since we move in image space
    dim3 n_blocks(16, 16);
    dim3 n_threads;
    n_threads.x = (cols() + n_blocks.x - 1) / n_blocks.x;
    n_threads.y = (rows() + n_blocks.y - 1) / n_blocks.y;

    toCloud_kernel<<<n_blocks, n_threads>>>(target->deviceInstance(),
                                            matrix_.deviceInstance(),
                                            sensor_->sensorOffset(),
                                            inv_K,
                                            sensor_->cameraType(),
                                            ifx,
                                            ify,
                                            cx,
                                            cy,
                                            attributes_->min_depth,
                                            attributes_->max_depth);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void ImageSet::fromCloud(const MatrixCloud& src_cloud) {
    ImageEntry zero_entry;
    zero_entry.setDepth(attributes_->max_depth + 1);
    matrix_.fill(zero_entry);

    Eigen::Isometry3f inv_sensor_offset = sensor_->sensorOffset().inverse();
    Eigen::Vector3f sperical_point;
    Eigen::Vector3f coordinates;
    Eigen::Vector3f camera_point = Eigen::Vector3f::Zero();
    const float fx        = sensor_->cameraMatrix()(0, 0);
    const float fy        = sensor_->cameraMatrix()(1, 1);
    const float cx        = sensor_->cameraMatrix()(0, 2);
    const float cy        = sensor_->cameraMatrix()(1, 2);
    float w               = 0;

    for (size_t i = 0; i < src_cloud.size(); ++i){
    auto& src = src_cloud[i];
      if (src.status != PointStatusFlag::Good)
        continue;
      coordinates    = inv_sensor_offset * src.coordinates;
      const float x = coordinates.x();
      const float y = coordinates.y();
      const float z = coordinates.z();
      switch (sensor_->cameraType()) {
        case Pinhole: {
          w = coordinates(2);
          if (w < attributes_->min_depth || w > attributes_->max_depth)
            continue;
          camera_point = sensor_->cameraMatrix() * coordinates;
          camera_point.block<2, 1>(0, 0) *= 1. / w;
        } break;
        case Spherical: {
          w = coordinates.norm();
          if (w < attributes_->min_depth || w > attributes_->max_depth)
            continue;
          sperical_point.x() = atan2(y, x);
          sperical_point.y() = atan2(coordinates.z(), sqrt(x * x + y * y));
          sperical_point.z() = z;
          camera_point.x()   = fx * sperical_point.x() + cx;
          camera_point.y()   = fy * sperical_point.y() + cy;
          camera_point.z()   = w;
        } break;
        default:;
      }
      
      int c = (int) (camera_point.y() + (camera_point.y() >= 0 ? 0.5f : -0.5f));
      int r = (int) (camera_point.x() + (camera_point.x() >= 0 ? 0.5f : -0.5f));
      
      if (!matrix_.inside(r, c))
        continue;
      ImageEntry& entry = matrix_.at(r, c);

      if (w < entry.depth()) {
        entry.setIntensity(src.intensity);
        entry.setDepth(w);
        entry.setNormal(inv_sensor_offset.linear() * src.normal);
#ifdef _MD_ENABLE_SUPERRES_
        entry.c = camera_point.x();
        entry.r = camera_point.y();
#endif
        entry.setMasked(false);
      }
    }

    growMask();
    updateDerivatives();
  }


  void ImageSet::toCloud(MatrixCloud& target) const {
    target.resize(rows(), cols());
    Pointf p;
    p.status = PointStatusFlag::Invalid;
    target.fill(p);

    const float ifx = 1. / sensor_->cameraMatrix()(0, 0);
    const float ify = 1. / sensor_->cameraMatrix()(1, 1);
    const float cx  = sensor_->cameraMatrix()(0, 2);
    const float cy  = sensor_->cameraMatrix()(1, 2);

    Eigen::Matrix3f inv_K = sensor_->cameraMatrix().inverse();
    for (int r = 0; r < rows(); ++r) {
      for (int c = 0; c < cols(); ++c) {
        const ImageEntry& src        = matrix_.at(r, c);
        Pointf& dest = target.at(r, c);
        float w                      = src.depth();
        if (src.masked() || w < attributes_->min_depth || w > attributes_->max_depth)
          continue;
#ifdef _MD_ENABLE_SUPERRES_
        const float row = src.r;
        const float col = src.c;
#else
        const float row = r;
        const float col = c;
#endif
        dest.status = PointStatusFlag::Good;
        switch (sensor_->cameraType()) {
          case Pinhole: {
            Eigen::Vector3f p         = inv_K * Eigen::Vector3f(col * w, row * w, w);
            dest.coordinates = p;
          } break;
          case Spherical: {
            float azimuth         = ifx * (col - cx);
            float elevation       = ify * (row - cy);
            float s0              = sin(azimuth);
            float c0              = cos(azimuth);
            float s1              = sin(elevation);
            float c1              = cos(elevation);
            dest.coordinates(0) = c0 * c1 * w;
            dest.coordinates(1) = s0 * c1 * w;
            dest.coordinates(2) = s1 * w;
          } break;
          default:;
        }

        dest.intensity = src.intensity();
        dest.coordinates = sensor_->sensorOffset() * dest.coordinates;
        dest.normal    = sensor_->sensorOffset().linear() * src.normal();

      }
    }
  }


} // namespace bundlee
