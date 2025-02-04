// Copyright 2024 R(obots) V(ision) and P(erception) group
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <registration/geometric.h>
#include <registration/photometric.cuh>
#include <sys/time.h>
#include <tools/image_set.cuh>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/opencv.hpp>

namespace py = pybind11;

class GSAligner {
public:
  GSAligner(const int num_threads) : num_threads_(num_threads_) {
    max_parallel_levels_ = static_cast<int>(std::log2(num_threads_));
    omp_set_num_threads(num_threads_);
  }

  void setQueryCloud(ContainerType query, const double b_max, const double b_min) {
    ContainerType* query_ptr = &query;
    query_tree_.reset(
      new MADtree(query_ptr, query_ptr->begin(), query_ptr->end(), b_max, b_min, 0, max_parallel_levels_, nullptr, nullptr));
    query_tree_->getLeafs(std::back_insert_iterator<LeafList>(query_leaves_));
  }

  void setReferenceCloud(ContainerType reference, const double b_max, const double b_min) {
    ContainerType* reference_ptr = &reference;
    ref_b_max_                   = b_max;
    ref_tree_.reset(new MADtree(
      reference_ptr, reference_ptr->begin(), reference_ptr->end(), ref_b_max_, b_min, 0, max_parallel_levels_, nullptr, nullptr));
  }

  void setReferenceCameraMatrix(const Eigen::Matrix3f& cam_matrix_) {
    cameraMatrix_ = cam_matrix_;
  }

  void setImages(const py::array_t<float> query_depth, const py::array_t<float> ref_depth) {
    setDepthImg(imgReference_, ref_depth);
    setDepthImg(imgQuery_, query_depth);
  }

  void setDepthImg(photo::Image& outimg, const py::array_t<float> input_depth_array) {
    // request a buffer descriptor from Python
    py::buffer_info buf_info = input_depth_array.request();

    // check the dimensions of the input array
    if (buf_info.ndim != 2) {
      throw std::runtime_error("GSAligner::setDepthImage|input should be a 2D numpy array");
    }

    // get the dimensions of the array
    const size_t rows = buf_info.shape[0];
    const size_t cols = buf_info.shape[1];

    // get a pointer to the data as a float*
    float* ptr = static_cast<float*>(buf_info.ptr);

    // resize the matrix to match the input array dimensions
    outimg = photo::Image(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        const float dvalue = ptr[r * cols + c];
        if (dvalue > dtol)
          outimg.at(r, c).setMasked(false);
        outimg.at(r, c).setDepth(dvalue);
        // you can set intensity and normals
      }
    }
  }

  inline Eigen::Matrix4d compute(const Eigen::Matrix4d& T,
                                 const size_t max_icp_iterations,
                                 const double rho_ker,
                                 double b_ratio,
                                 const bool print_stats) {
    mad_icp_.reset(new GeometricFac(ref_b_max_, rho_ker, b_ratio, 1));
    mad_icp_->setMoving(query_leaves_);
    // make initial guess in right format
    Eigen::Isometry3d dX = Eigen::Isometry3d::Identity();
    dX.linear()          = T.block<3, 3>(0, 0);
    dX.translation()     = T.block<3, 1>(0, 3);
    mad_icp_->init(dX);

    float icp_time = 0;
    struct timeval icp_start, icp_end, icp_delta;
    gettimeofday(&icp_start, nullptr);

    // icp loop
    for (size_t icp_iteration = 0; icp_iteration < max_icp_iterations; ++icp_iteration) {
      if (icp_iteration == max_icp_iterations - 1) {
        for (MADtree* l : query_leaves_) {
          l->matched_ = false;
        }
      }
      mad_icp_->resetAdders();
      mad_icp_->update(ref_tree_.get());
      mad_icp_->updateState();
    }

    gettimeofday(&icp_end, nullptr);
    timersub(&icp_end, &icp_start, &icp_delta);
    icp_time = float(icp_delta.tv_sec) * 1000. + 1e-3 * icp_delta.tv_usec;

    if (print_stats) {
      int matched_leaves = 0;
      for (MADtree* l : query_leaves_) {
        if (l->matched_) {
          matched_leaves++;
        }
      }
      const double inliers_ratio = double(matched_leaves) / double(query_leaves_.size());
      std::cout << "geometricFac|compute time " << icp_time << " [ms] " << std::endl;
      std::cout << "geometricFac|inliers ratio " << inliers_ratio << std::endl;
      std::cout << "--geometricFac|matched leaves " << matched_leaves << std::endl;
      std::cout << "--geometricFac|total num leaves " << query_leaves_.size() << std::endl;
    }

    // // start photometric
    photo::SensorPtr sensor = std::make_shared<photo::Sensor>();
    sensor->rows()          = imgReference_.rows();
    sensor->cols()          = imgReference_.cols();
    sensor->sensorOffset()  = Eigen::Isometry3f::Identity();
    sensor->cameraMatrix()  = cameraMatrix_;
    sensor->cameraType()    = photo::CameraType::Spherical;
    sensor->setDepthScale(1);

    // // TODO some params to take out
    const float min_depth = 0.2f;
    const float max_depth = 80.f;
    int mask_grow_radius  = 1;
    // derivative thresholds
    float intensity_derivative_threshold = 10.0f;
    float depth_derivative_threshold     = 0.5f;
    float normals_derivative_threshold   = 0.3f;
    // filtering policies
    uint8_t intensity_policy = 0; // {0 Ignore, 1 Suppress, 2 Clamp}
    uint8_t depth_policy     = 2; // {0 Ignore, 1 Suppress, 2 Clamp}
    uint8_t normals_policy   = 0; // 2;    // {0 Ignore, 1 Suppress, 2 Clamp}

    photo::ImageAttributesPtr attributes     = std::make_shared<photo::ImageAttributes>();
    attributes->min_depth                    = min_depth;
    attributes->max_depth                    = max_depth;
    attributes->policies[photo::Intensity]   = (photo::FilterPolicy) intensity_policy;
    attributes->thresholds[photo::Intensity] = intensity_derivative_threshold;
    attributes->policies[photo::Depth]       = (photo::FilterPolicy) depth_policy;
    attributes->thresholds[photo::Depth]     = depth_derivative_threshold;
    attributes->policies[photo::Normal]      = (photo::FilterPolicy) normals_policy;
    attributes->thresholds[photo::Normal]    = normals_derivative_threshold;
    attributes->mask_grow_radius             = mask_grow_radius;

    photo::ImageSet set_query;
    set_query.matrix_     = imgQuery_;
    set_query.attributes_ = attributes;
    set_query.sensor_     = sensor;

    photo::ImageSet set_reference;
    set_reference.matrix_     = imgReference_;
    set_reference.attributes_ = attributes;
    set_reference.sensor_     = sensor;
    set_reference.growMask();
    set_reference.updateDerivatives();

    std::cerr << mad_icp_->X_.matrix() << std::endl;

    photo::PhotometricFac factor;
    factor.setFixedData(&set_reference);
    set_query.matrix_.toDevice();
    factor.setMovingData(set_query);
    factor.setTransform(mad_icp_->X_.cast<float>());
    factor.compute(50, print_stats);

    return factor.transform().matrix().cast<double>();
  }

protected:
  photo::Image imgReference_;
  photo::Image imgQuery_;
  Eigen::Matrix3f cameraMatrix_;

  std::unique_ptr<GeometricFac> mad_icp_ = nullptr;
  std::unique_ptr<MADtree> ref_tree_     = nullptr;
  std::unique_ptr<MADtree> query_tree_   = nullptr;
  LeafList query_leaves_;
  double ref_b_max_;
  int max_parallel_levels_;
  int num_threads_;

  float dtol = 1e-5; // tolerance to mask 0 depth
};
