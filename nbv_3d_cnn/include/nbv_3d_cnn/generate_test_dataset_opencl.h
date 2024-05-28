#ifndef GENERATE_TEST_DATASET_OPENCL_H
#define GENERATE_TEST_DATASET_OPENCL_H

// OpenCL
#include <CL/cl2.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// STL
#include <string>
#include <stdint.h>
#include <vector>
#include <cmath>

#include <nbv_3d_cnn/voxelgrid.h>

#include "opencl_program.h"

class GenerateTestDatasetOpenCL: OpenCLProgram
{
  public:
  typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > Vector2iVector;
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;
  typedef std::vector<float> FloatVector;
  typedef std::vector<Voxelgrid> VoxelgridVector;
  typedef std::vector<uint64> Uint64Vector;

  GenerateTestDatasetOpenCL(ros::NodeHandle & nh);

  void SimulateView(const Voxelgrid & environment, const Eigen::Vector3f & origin,
                    const Vector3fVector ray_orientations,
                    const float max_range, const float min_range,
                    FloatVector & nearest_dist, Vector3iVector & observed_points);

  void SimulateMultiRay(const Voxelgrid &environment, const Vector3fVector &origins,
                        const Vector3fVector &bearings, const float max_range, const float min_range,
                        FloatVector & nearest_dist, Vector3iVector &observed_points);
  // accepts invalid bearings
  void SimulateMultiRayWI(const Voxelgrid &environment, const Vector3fVector &origins,
                          const Vector3fVector &bearings, const float max_range, const float min_range,
                          FloatVector & nearest_dist, Vector3iVector &observed_points);

  void SimulateMultiRayWithInformationGain(const Voxelgrid & known_empty,
                                           const Voxelgrid & known_occupied,
                                           const Vector3fVector & origins,
                                           const Vector3fVector &bearings,
                                           const float sensor_f,
                                           const float max_range,
                                           const float min_range,
                                           const bool stop_at_first_hit,
                                           const float a_priori_occupied_prob,
                                           FloatVector & hits,
                                           FloatVector & miss);

  void SimulateMultiRayWithInformationGainBatched(const Voxelgrid & known_empty,
                                                  const Voxelgrid & known_occupied,
                                                  const Vector3fVector & origins,
                                                  const Vector3fVector & bearings,
                                                  const float sensor_f,
                                                  const float max_range,
                                                  const float min_range,
                                                  const bool stop_at_first_hit,
                                                  const float a_priori_occupied_prob,
                                                  FloatVector & hits,
                                                  FloatVector & miss);

  void FillEnvironmentFromView(const Voxelgrid & input_empty,
                               const Eigen::Vector3f & origin,
                               const Eigen::Quaternionf &orientation,
                               const float sensor_f,
                               const Eigen::Vector2i &resolution,
                               const FloatVector & nearest_dist,
                               const Vector3iVector & observed_points,
                               Voxelgrid & filled_empty
                               );

  void FillEnvironmentFromViewCube(const Voxelgrid & input_empty,
                                   const Eigen::Vector3f & origin,
                                   const Eigen::Quaternionf &orientation,
                                   const Eigen::Vector2f &sensor_hfov,
                                   const Eigen::Vector3i & view_cube_resolution,
                                   const FloatVector & nearest_dist,
                                   const Vector3iVector & observed_points,
                                   Voxelgrid & filled_empty
                                   );

  void ComputeGainFromViewCube(const Voxelgrid & input_empty,
                               const Eigen::Vector3f & origin,
                               const Eigen::Vector3i &view_cube_resolution,
                               const float max_range,
                               const FloatVector & nearest_dist,
                               Uint64Vector & hits,
                               Uint64Vector & miss
                               );

  void EvaluateSensorOrientationsOnViewCube(const Voxelgrid &view_cube_hits,
                                            const Voxelgrid &view_cube_miss,
                                            const Eigen::Vector2f & hfov,
                                            const QuaternionfVector & orientations,
                                            FloatVector &hits, FloatVector &miss);

  float GetLastUploadTime() const {return m_last_upload_time; }
  float GetLastSimulationTime() const {return m_last_simulation_time; }

  private:
  void UploadEnv(const Voxelgrid &known_empty,
                 const Voxelgrid &known_occupied);

  void SimulateMultiRayWithInformationGainNoUpEnv(const Vector3fVector &origins,
                                                  const Vector3fVector &bearings,
                                                  const float sensor_f,
                                                  const float max_range,
                                                  const float min_range,
                                                  const bool stop_at_first_hit,
                                                  const float a_priori_occupied_prob,
                                                  FloatVector & hits,
                                                  FloatVector & miss);

  ros::NodeHandle & m_nh;

  CLKernelPtr m_simulate_view_kernel;

  uint64 m_last_sensor_resolution;
  CLBufferPtr m_distances;
  CLBufferPtr m_observed_points;

  CLKernelPtr m_fill_uint_kernel;
  CLKernelPtr m_fill_float_kernel;

  CLKernelPtr m_simulate_multi_ray_kernel;
  uint64 m_last_multi_ray_size;
  CLBufferPtr m_multi_ray_origins;
  CLBufferPtr m_multi_ray_orientations;
  CLBufferPtr m_multi_ray_distances;
  CLBufferPtr m_multi_ray_observed_points;

  CLKernelPtr m_multi_ray_ig_kernel;
  uint64 m_last_multi_ray_ig_size;
  Eigen::Vector3i m_last_environment_ig_size;
  CLBufferPtr m_multi_ray_ig_known_occupied;
  CLBufferPtr m_multi_ray_ig_known_empty;
  CLBufferPtr m_multi_ray_ig_origins;
  CLBufferPtr m_multi_ray_ig_orientations;
  CLBufferPtr m_multi_ray_ig_hits;
  CLBufferPtr m_multi_ray_ig_miss;
  float m_last_upload_time;
  float m_last_simulation_time;

  CLFloatVector m_cl_origins; // prevent these from being deallocated (wastes a lot of time, for some reason)
  CLFloatVector m_cl_orientations;
  CLFloatVector m_cl_hits;
  CLFloatVector m_cl_miss;

  CLKernelPtr m_compute_gain_from_view_kernel;
  uint64 m_last_compute_gain_from_view_size;
  CLBufferPtr m_compute_gain_from_view_nearest_dist;
  CLBufferPtr m_compute_gain_from_view_hits;
  CLBufferPtr m_compute_gain_from_view_miss;

  CLKernelPtr m_fill_environment_from_view_kernel;
  uint64 m_last_fill_environment_from_view_size;
  CLBufferPtr m_fill_environment_from_view_nearest_dist;
  CLBufferPtr m_fill_environment_from_view_observed_points;

  CLKernelPtr m_fill_environment_from_view_cube_kernel;
  uint64 m_last_fill_environment_from_view_cube_size;
  CLBufferPtr m_fill_environment_from_view_cube_nearest_dist;
  CLBufferPtr m_fill_environment_from_view_cube_observed_points;

  CLKernelPtr m_evaluate_sensor_orientation_kernel;
  CLBufferPtr m_evaluate_sensor_orientation_view_cube;
  uint64 m_evaluate_sensor_orientation_view_cube_size;
  CLBufferPtr m_evaluate_sensor_orientation_orientations;
  CLBufferPtr m_evaluate_sensor_orientation_hits;
  CLBufferPtr m_evaluate_sensor_orientation_miss;
  uint64 m_evaluate_sensor_orientation_orientations_size;
  uint64 m_evaluate_sensor_orientation_hits_miss_size;

  Eigen::Vector3i m_last_environment_size;
  CLBufferPtr m_environment;
};

#endif // GENERATE_TEST_DATASET_OPENCL_H
