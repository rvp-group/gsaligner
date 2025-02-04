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

#include <Eigen/Dense>
#include <iterator>
#include <memory>

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

template <typename IteratorType, typename PredicateType>
IteratorType split(const IteratorType& begin, const IteratorType& end, const PredicateType& predicate) {
  IteratorType lower                        = begin;
  std::reverse_iterator<IteratorType> upper = std::make_reverse_iterator(end);
  while (lower != upper.base()) {
    Eigen::Vector3d& v_lower = *lower;
    Eigen::Vector3d& v_upper = *upper;
    if (predicate(v_lower)) {
      ++lower;
    } else {
      std::swap(v_lower, v_upper);
      ++upper;
    }
  }
  return upper.base();
}

template <typename IteratorType>
int computeMeanAndCovariance(Eigen::Vector3d& mean, Eigen::Matrix3d& cov, const IteratorType& begin, const IteratorType& end) {
  // mean computed as 1/(end-start) Sum_k={start..end} x_k
  // cov computed as  (1/(end-start) Sum_k={start..end} x_k* x_k^T ) - mean*mean.transpose();
  mean.setZero();
  cov.setZero();
  int k = 0;
  for (IteratorType it = begin; it != end; ++it) {
    const Eigen::Vector3d& v = *it;
    mean += v;
    cov += v * v.transpose();
    ++k;
  }
  mean *= (1. / k);
  cov *= (1. / k);
  cov -= mean * mean.transpose();
  cov *= double(k) / double(k - 1);

  return k;
}

template <typename IteratorType>
int computeBoundingBox(Eigen::Vector3d& b_max,
                       const Eigen::Vector3d& center,
                       const Eigen::Matrix3d& R,
                       const IteratorType& begin,
                       const IteratorType& end) {
  b_max.setZero();
  int k = 0;

  Eigen::Vector3d bbox_neg = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_pos = Eigen::Vector3d::Zero();
  for (IteratorType it = begin; it != end; ++it) {
    const Eigen::Vector3d v = R * (*it - center);
    for (size_t i = 0; i < 3; ++i) {
      bbox_neg(i) = std::min<double>(bbox_neg(i), v(i));
      bbox_pos(i) = std::max<double>(bbox_pos(i), v(i));
    }
    ++k;
  }

  b_max = bbox_pos - bbox_neg;
  return k;
}

namespace photo {

  enum CameraType { Pinhole = 0, Spherical = 1, Unknown = -1 };

  struct Sensor {
    // clang-format off
    inline const Eigen::Isometry3f& sensorOffset() const { return sOffset; }
    inline const Eigen::Matrix3f& cameraMatrix() const { return camMatrix; }
    inline const CameraType cameraType() const { return camType; }
    inline const int rows() const { return rws; }
    inline const int cols() const { return cls; }
    inline const float depthScale() const { return dScale; }

    inline Eigen::Isometry3f& sensorOffset() { return sOffset; }
    inline Eigen::Matrix3f& cameraMatrix() { return camMatrix; }
    inline CameraType& cameraType() { return camType; }
    inline void setDepthScale(float dScale) { dScale = dScale; }
    inline int& rows() { return rws; }
    inline int& cols() { return cls; }
    // clang-format on

  protected:
    Eigen::Isometry3f sOffset = Eigen::Isometry3f::Identity();
    Eigen::Matrix3f camMatrix = Eigen::Matrix3f::Zero();
    CameraType camType        = CameraType::Pinhole;
    float dScale              = 0.f;
    int rws                   = 0;
    int cls                   = 0;
  };

  using SensorPtr = std::shared_ptr<Sensor>;

  __host__ __device__ inline bool project(Eigen::Vector2f& image_point,
                                          Eigen::Vector3f& camera_point,
                                          float& depth,
                                          const Eigen::Vector3f& point,
                                          const CameraType& camera_type,
                                          const Eigen::Matrix3f& camera_mat,
                                          const float min_depth,
                                          const float max_depth) {
    switch (camera_type) {
      case CameraType::Pinhole:
        depth = point.z();
        if (depth < min_depth || depth > max_depth) {
          return false;
        }
        camera_point = camera_mat * point;
        image_point  = camera_point.head(2) * 1.f / depth;
        break;
      case CameraType::Spherical:
        depth = point.norm();
        if (depth < min_depth || depth > max_depth) {
          return false;
        }
        camera_point.x() = atan2(point.y(), point.x());
        camera_point.y() = atan2(point.z(), point.head<2>().norm());
        camera_point.z() = depth;

        image_point.x() = camera_mat(0, 0) * camera_point.x() + camera_mat(0, 2);
        image_point.y() = camera_mat(1, 1) * camera_point.y() + camera_mat(1, 2);
        break;
        // default:
        //   throw std::runtime_error("utils::project | unknown camera type");
    }
    return true;
  }
} // namespace photo