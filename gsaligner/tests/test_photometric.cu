#include <bits/stdc++.h>
#include <random>

#include <iostream>
#include <vector>

#include <registration/photometric.cuh>
#include <tools/image_set.cuh>
#include <tools/utils.cuh>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#ifndef GSTEST_DATA_FOLDER
#error "NO TEST DATA FOLDER"
#endif

const std::string test_path = GSTEST_DATA_FOLDER;
const std::string depth_filename     = test_path + "camera.depth.image_raw_00002102.pgm";
const std::string intensity_filename = test_path + "camera.rgb.image_raw_00002102.png";

// sensor/data dependent
float min_depth = 0.3f;
float max_depth = 5.0f;
const float scale = 0.001f;

// mask grow radius settings
int mask_grow_radius = 1;

// derivative thresholds
float intensity_derivative_threshold = 10.0f;
float depth_derivative_threshold = 0.5f;
float normals_derivative_threshold = 0.3f;
// filtering policies
uint8_t intensity_policy = 0;  // {0 Ignore, 1 Suppress, 2 Clamp}
uint8_t depth_policy = 2;      // {0 Ignore, 1 Suppress, 2 Clamp}
uint8_t normals_policy = 0; //2;    // {0 Ignore, 1 Suppress, 2 Clamp}

// normal blurring parameters
unsigned int normals_scaled_blur_multiplier = 1;
unsigned int normals_blur_region_size = 3;

using namespace photo;

TEST(DUMMY, PhotometricFac) {

  cv::Mat intensity_cv = cv::imread(intensity_filename, cv::IMREAD_GRAYSCALE);
  cv::Mat depth_cv = cv::imread(depth_filename, cv::IMREAD_ANYDEPTH);

  const size_t rows = intensity_cv.rows;
  const size_t cols = intensity_cv.cols;

  Matrixf depth(rows, cols);
  Matrixf intensity(rows, cols);


  cv::Mat intensity_float;
  intensity_cv.convertTo(intensity_float, CV_32FC1, 1/255.0);

  Image img(rows, cols);
  for (size_t r = 0; r < rows; ++r){
    for (size_t c = 0; c < cols; ++c){
      const uint16_t& dvalue = depth_cv.at<uint16_t>(r, c);
      if(dvalue != 0)
        img.at(r, c).setMasked(false);
      img.at(r, c).setDepth(dvalue * scale);
      img.at(r, c).setIntensity(intensity_float.at<float>(r, c));
    }
  }

  // intrinsics and extrinsics
  Eigen::Matrix3f pinhole_camera_matrix;
  pinhole_camera_matrix << 481.2f, 0.f, 319.5f, 0.f, 481.f, 239.5f, 0.f, 0.f, 1.f;
  Eigen::Isometry3f forced_offset    = Eigen::Isometry3f::Identity();
  forced_offset.translation() = Eigen::Vector3f(0.2, 0.3, 0.1);
  forced_offset.linear()      = Eigen::AngleAxisf(1, Eigen::Vector3f(1, 0, 0)).toRotationMatrix();

  // create a fixed simulated measurement
  Image img_fixed = img;

  // configuring some constant components to propagate along images
  SensorPtr sensor = std::make_shared<Sensor>();
  sensor->rows() = rows;
  sensor->cols() = cols;
  sensor->sensorOffset() = forced_offset;
  sensor->cameraMatrix() = pinhole_camera_matrix;
  sensor->cameraType() = CameraType::Pinhole;
  sensor->setDepthScale(scale);
  
  ImageAttributesPtr attributes = std::make_shared<ImageAttributes>();
  attributes->min_depth = min_depth;
  attributes->max_depth = max_depth;
  attributes->policies[Intensity]   = (FilterPolicy) intensity_policy;
  attributes->thresholds[Intensity] = intensity_derivative_threshold;
  attributes->policies[Depth]       = (FilterPolicy) depth_policy;
  attributes->thresholds[Depth]     = depth_derivative_threshold;
  attributes->policies[Normal]      = (FilterPolicy) normals_policy;
  attributes->thresholds[Normal]    = normals_derivative_threshold;
  attributes->mask_grow_radius      = mask_grow_radius;

  ImageSet set;
  set.matrix_ = img;
  set.attributes_ = attributes;
  set.sensor_ = sensor;

  ImageSet set_fixed;
  set_fixed.matrix_ = img_fixed;
  set_fixed.attributes_ = attributes;
  set_fixed.sensor_ = sensor;
  set_fixed.growMask();
  set_fixed.updateDerivatives();

  PhotometricFac factor;
  factor.setFixedData(&set_fixed);
  set.matrix_.toDevice();
  factor.setMovingData(set);

  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
  T.translation() << 0.01, 0.01, 0.01;

  Eigen::AngleAxisf rotX(0.02, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf rotY(0.02, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rotZ(0.02, Eigen::Vector3f::UnitZ());

  Eigen::Matrix3f rotation_matrix = rotX.toRotationMatrix() * rotY.toRotationMatrix() * rotZ.toRotationMatrix();
  T.linear() = rotation_matrix;

  factor.setTransform(T);
  factor.compute(100, true);

  std::cerr << factor.transform().matrix() << std::endl;

}
