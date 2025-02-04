#include "aligner.h"

#include "registration/geometric.h"
#include <Eigen/Core>
#include <iterator>
#include <torch/extension.h>
#include <vector>

#define CHECK_INPUT(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
// TODO: Controllare
constexpr int MAX_PARALLEL_LEVELS = 4;

std::vector<Eigen::Vector3f> TensorToVector3f(const torch::Tensor& tensor) {
  std::vector<Eigen::Vector3f> vec(tensor.size(0));
  for (int i = 0; i < tensor.size(0); ++i) {
    vec[i] = Eigen::Map<Eigen::Vector3f>(tensor[i].data<float>());
  }
  return vec;
}
std::vector<Eigen::Vector3d> TensorToVector3d(const torch::Tensor& tensor) {
  std::vector<Eigen::Vector3d> vec(tensor.size(0));
  for (int i = 0; i < tensor.size(0); ++i) {
    vec[i] = (Eigen::Map<Eigen::Vector3f>(tensor[i].data<float>())).cast<double>();
  }
  return vec;
}

std::tuple<torch::Tensor, float, float> RegistrationGeomPhoto(const torch::Tensor& ref_T_query_init,
                                                              const torch::Tensor& reference_depth,
                                                              const torch::Tensor& query_depth,
                                                              const torch::Tensor& reference_cloud,
                                                              const torch::Tensor& query_cloud,
                                                              const torch::Tensor& reference_projmat,
                                                              const int image_height,
                                                              const int image_width,
                                                              const float geom_b_min,
                                                              const float geom_b_max,
                                                              const float geom_b_ratio,
                                                              const float geom_rho_ker,
                                                              const int geom_iterations,
                                                              const float photo_omega_depth,
                                                              const float photo_depth_rejection_threshold,
                                                              const float photo_rho_ker,
                                                              const float photo_min_depth,
                                                              const float photo_max_depth,
                                                              const int photo_iterations) {
  // const int H = image_height;
  // const int W = image_width;

  CHECK_INPUT(ref_T_query_init);
  CHECK_INPUT(reference_depth);
  CHECK_INPUT(query_depth);
  CHECK_INPUT_CPU(reference_cloud);
  CHECK_INPUT_CPU(query_cloud);
  CHECK_INPUT(reference_projmat);

  if (reference_depth.ndimension() != 3) {
    AT_ERROR("reference_depth must be a (1xHxW) tensor");
  }
  if (query_depth.ndimension() != 3) {
    AT_ERROR("reference_depth must be a (1xHxW) tensor");
  }

  if (reference_cloud.ndimension() != 2 || reference_cloud.size(1) != 3) {
    AT_ERROR("reference_cloud must be a (Nx3) tensor");
  }

  if (query_cloud.ndimension() != 2 || query_cloud.size(1) != 3) {
    AT_ERROR("reference_cloud must be a (Nx3) tensor");
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  torch::Tensor ref_T_query = torch::eye(4, options);
  float align_fitness       = 0.0;
  float align_rmse          = 0.0;

  /** This is the merging point between torch and gsalign
   * To do that, we now have to carefully pass shit to the appropriate stuff
   */

  // Geometric
  auto std_ref_cloud   = TensorToVector3d(reference_cloud);
  auto std_query_cloud = TensorToVector3d(query_cloud);
  // TODO: get T from ref_T_query_init
  auto T = Eigen::Matrix4d::Identity();

  std::unique_ptr<MADtree> ref_tree;
  ref_tree.reset(new MADtree(&std_ref_cloud,
                             std_ref_cloud.begin(),
                             std_ref_cloud.end(),
                             geom_b_max,
                             geom_b_min,
                             0,
                             MAX_PARALLEL_LEVELS,
                             nullptr,
                             nullptr));

  std::unique_ptr<MADtree> query_tree;
  query_tree.reset(new MADtree(&std_query_cloud,
                               std_query_cloud.begin(),
                               std_query_cloud.end(),
                               geom_b_max,
                               geom_b_min,
                               0,
                               MAX_PARALLEL_LEVELS,
                               nullptr,
                               nullptr));
  LeafList query_leaves_;
  query_tree->getLeafs(std::back_insert_iterator<LeafList>(query_leaves_));

  std::unique_ptr<GeometricFac> mad_icp_;

  mad_icp_.reset(new GeometricFac(geom_b_max, geom_rho_ker, geom_b_ratio, 1));
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
  for (size_t icp_iteration = 0; icp_iteration < geom_iterations; ++icp_iteration) {
    if (icp_iteration == geom_iterations - 1) {
      for (MADtree* l : query_leaves_) {
        l->matched_ = false;
      }
    }
    mad_icp_->resetAdders();
    mad_icp_->update(ref_tree.get());
    mad_icp_->updateState();
  }

  gettimeofday(&icp_end, nullptr);
  timersub(&icp_end, &icp_start, &icp_delta);
  icp_time = float(icp_delta.tv_sec) * 1000. + 1e-3 * icp_delta.tv_usec;

  // if (print_stats) {
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
  std::cout << mad_icp_->X_.matrix() << std::endl;
  // }

  return std::make_tuple(ref_T_query, align_fitness, align_rmse);
}