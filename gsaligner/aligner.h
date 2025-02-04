#pragma once
#include <torch/extension.h>

#include <tuple>

/**
 * @brief The following function should act as the main interface for the total
 * registration process. It takes the following inputs:
 * - ref_T_query: a torch tensor of shape (4, 4) containing the initial guess
 * - reference_depth: A torch tensor of shape (1, H, W) containing the depth
 * image of the reference frame.
 * - query_depth: A torch tensor of shape (1, H, W) containing the depth image
 * of the query frame
 * - reference_projmat: A torch tensor of shape (4, 4) containing the projection
 * matrix of the reference frame.
 * - Geometric:
 * - b_min: when a node is flatten than this param, propagate normal
 * - b_max: max size of kd leaves
 * - b_ratio: the increase factor of search radius needed in data association
 * - rho_ker: huber threshold in mad-icp
 * - iterations: number of iterations
 * - Photometric:
 * - omegaIntensity <- Discard for now
 * - omegaDepth
 * - omegaNormals <- Discard for now
 * - depthRejectionThreshold
 * - kernelChiThreshold
 * - minDepth
 * - maxDepth
 * - iterations
 * - intensity_derivative_threshold <- Discard for now
 * - depth_derivative_threshold
 * - normals_derivative_threshold <- Discard for now
 *
 * The function should return:
 * - ref_T_query: a torch tensor of shape (4, 4) containing the final
 * transformation
 * - fitness: a float containing the fitness of the registration (no. inliers /
 * no. points)
 * - rmse: a float containing the root mean squared error of the registration
 */

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
                                                              const int photo_iterations);