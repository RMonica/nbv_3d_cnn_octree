#ifndef OCTREE_RAYCAST_OPENCL_H
#define OCTREE_RAYCAST_OPENCL_H

#include <nbv_3d_cnn/opencl_program.h>

#include <ros/ros.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "simulate_nbv_cycle_octree.h"
#include "nbv_3d_cnn/voxelgrid.h"

class OctreeRaycastOpenCL: OpenCLProgram
{
  public:
  typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;
  typedef std::vector<float> FloatVector;
  typedef uint64_t uint64;
  typedef uint32_t uint32;
  typedef std::vector<uint32> Uint32Vector;

  explicit OctreeRaycastOpenCL(ros::NodeHandle & nh, const uint64 num_levels);

  class EnvData {public: virtual ~EnvData() {}};
  typedef std::shared_ptr<EnvData> EnvDataPtr;
  class VoxelgridEnvData: public EnvData
  {
    public:
    const Voxelgrid & occupied_environment;
    const Voxelgrid & empty_environment;

    VoxelgridEnvData(const Voxelgrid & o, const Voxelgrid & e):
      occupied_environment(o), empty_environment(e) {}
  };
  typedef std::shared_ptr<VoxelgridEnvData> VoxelgridEnvDataPtr;
  class SerializedOctreeEnvData: public EnvData
  {
    public:
    const Uint32Vector & serialized_octree;
    SerializedOctreeEnvData(const Uint32Vector & se):
      serialized_octree(se) {}
  };
  typedef std::shared_ptr<SerializedOctreeEnvData> SerializedOctreeEnvDataPtr;

  void SimulateMultiViewWithInformationGain(const EnvDataPtr env_data,
                                            const Eigen::Vector3i & environment_size,
                                            const bool is_3d,
                                            const Vector3fVector &origins,
                                            const QuaternionfVector &orientations,
                                            const Vector3fVector & local_directions,
                                            const bool combine_origins_orientations, // if true, each orientation is
                                                                                     // evaluated for each origin
                                                                                     // else, one orientation must be
                                                                                     // defined for each origin
                                            const float sensor_f,
                                            const float max_range,
                                            const float min_range,
                                            const float a_priori_occupied_prob,
                                            FloatVector & scores);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  float GetLastUploadTime() const {return m_last_upload_time; }
  float GetLastSimulationTime() const {return m_last_simulation_time; }

  private:
  void UploadEnvOctree(const Uint32Vector & serialized_octree,
                 const Eigen::Vector3i &environment_size,
                 const bool is_3d);
  void UploadEnvVoxelgrid(const Voxelgrid & occupied_environment,
                          const Voxelgrid & empty_environment,
                          const bool is_3d);
  void SimulateMultiViewWithInformationGainNoUpEnv(const uint64 subrange_min,
                                                   const uint64 subrange_max,
                                                   const EnvDataPtr env_data,
                                                   const Eigen::Vector3i & environment_size,
                                                   const bool is_3d,
                                                   const Vector3fVector &origins,
                                                   const QuaternionfVector &orientations,
                                                   const Vector3fVector & local_directions,
                                                   const bool combine_origins_orientations,
                                                   const float sensor_f,
                                                   const float max_range,
                                                   const float min_range,
                                                   const float a_priori_occupied_prob,
                                                   FloatVector & scores);

  ros::NodeHandle & m_nh;

  CLBufferPtr m_multi_ray_ig_origins;
  CLBufferPtr m_multi_ray_ig_orientations;
  CLBufferPtr m_multi_ray_ig_hits;
  CLBufferPtr m_multi_ray_ig_miss;
  uint64 m_last_multi_ray_ig_size;
  CLKernelPtr m_multi_ray_ig_kernel;

  CLFloat4Vector m_cl_orientations; // prevent deallocation
  CLFloat4Vector m_cl_origins;
  CLFloat4Vector m_cl_local_directions;
  CLUInt32Vector m_cl_scores;

  CLFloatVector m_cl_known_occupied;
  CLFloatVector m_cl_known_empty;
  CLUInt32Vector m_cl_serialized_octree;

  CLKernelPtr m_multi_view_ig_kernel;
  CLKernelPtr m_multi_view_ig_octree_kernel;
  CLBufferPtr m_multi_view_ig_scores;
  CLBufferPtr m_multi_view_ig_origins;
  CLBufferPtr m_multi_view_ig_orientations;
  CLBufferPtr m_multi_view_ig_local_directions;
  uint64 m_multi_view_ig_last_local_size = 0;
  uint64 m_multi_view_ig_last_origins_size = 0;
  uint64 m_multi_view_ig_last_orientations_size = 0;
  uint64 m_multi_view_ig_last_score_size = 0;

  CLBufferPtr m_multi_ray_serialized_octree;
  uint64 m_last_serialized_octree_size = 0;
  bool m_last_is_3d;
  Eigen::Vector3i m_last_environment_size;

  CLBufferPtr m_multi_ray_occupied_voxelgrid;
  uint64 m_last_occupied_voxelgrid_size = 0;
  CLBufferPtr m_multi_ray_empty_voxelgrid;
  uint64 m_last_empty_voxelgrid_size = 0;

  CLKernelPtr m_fill_uint_kernel;

  float m_last_upload_time;
  float m_last_simulation_time;
};

#endif // OCTREE_RAYCAST_OPENCL_H
