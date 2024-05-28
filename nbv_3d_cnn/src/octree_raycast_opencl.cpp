#include "octree_raycast_opencl.h"

#include "octree_raycast.cl.h"

OctreeRaycastOpenCL::OctreeRaycastOpenCL(ros::NodeHandle & nh, const uint64 num_levels):
  OpenCLProgram(nh, std::string("#define NUM_LEVELS ") + std::to_string(num_levels) + "\n" + OCTREE_RAYCAST_CL), m_nh(nh)
{
  m_last_multi_ray_ig_size = 0;
  m_last_serialized_octree_size = 0;
}

void OctreeRaycastOpenCL::UploadEnvOctree(const Uint32Vector & serialized_octree,
                                    const Eigen::Vector3i &environment_size,
                                    const bool is_3d)
{
  m_last_environment_size = environment_size;
  m_last_is_3d = is_3d;

  if (!m_multi_ray_serialized_octree || serialized_octree.size() > m_last_serialized_octree_size)
  {
    // create size with some margin
    const uint64 new_size = std::max<uint64>(serialized_octree.size(), m_last_serialized_octree_size + m_last_serialized_octree_size / 2);
    m_multi_ray_serialized_octree = CreateBuffer(m_opencl_context, new_size * sizeof(cl_uint),
                                                 "m_multi_ray_serialized_octree");
    m_last_serialized_octree_size = new_size;
    m_cl_serialized_octree.resize(new_size);
  }

  for (uint64 i = 0; i < serialized_octree.size(); i++)
    m_cl_serialized_octree[i] = serialized_octree[i];

  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_ray_serialized_octree, CL_TRUE, 0,
                                             serialized_octree.size() * sizeof(cl_uint),
                                             m_cl_serialized_octree.data());
  m_opencl_command_queue->finish();
}

void OctreeRaycastOpenCL::UploadEnvVoxelgrid(const Voxelgrid & occupied_environment,
                                             const Voxelgrid & empty_environment,
                                             const bool is_3d)
{
  const Eigen::Vector3i environment_size = empty_environment.GetSize();

  if (occupied_environment.GetSize() != environment_size)
  {
    ROS_FATAL("UploadEnvVoxelgrid: occupied and empty environment size mismatch!");
    std::exit(1);
  }

  m_last_environment_size = environment_size;
  m_last_is_3d = is_3d;

  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();

  if (!m_multi_ray_occupied_voxelgrid || environment_size.prod() < m_last_occupied_voxelgrid_size)
  {
    m_multi_ray_occupied_voxelgrid = CreateBuffer(m_opencl_context, environment_size.prod() * sizeof(cl_float),
                                                  "m_multi_ray_occupied_voxelgrid");
    m_last_occupied_voxelgrid_size = environment_size.prod();
  }

  if (!m_multi_ray_empty_voxelgrid || environment_size.prod() < m_last_empty_voxelgrid_size)
  {
    m_multi_ray_empty_voxelgrid = CreateBuffer(m_opencl_context, environment_size.prod() * sizeof(cl_float),
                                                  "m_multi_ray_empty_voxelgrid");
    m_last_empty_voxelgrid_size = environment_size.prod();
  }

  m_cl_known_occupied.resize(environment_size.prod());
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        m_cl_known_occupied[z * width * height + y * width + x] = occupied_environment.at(x, y, z);

  m_cl_known_empty.resize(environment_size.prod());
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        m_cl_known_empty[z * width * height + y * width + x] = empty_environment.at(x, y, z);

  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_ray_occupied_voxelgrid, CL_TRUE, 0,
                                             environment_size.prod() * sizeof(cl_float),
                                             m_cl_known_occupied.data());

  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_ray_empty_voxelgrid, CL_TRUE, 0,
                                             environment_size.prod() * sizeof(cl_float),
                                             m_cl_known_empty.data());
  m_opencl_command_queue->finish();
}

void OctreeRaycastOpenCL::SimulateMultiViewWithInformationGain(const EnvDataPtr env_data,
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
                                                               FloatVector & scores)
{
  ros::Time upload_time = ros::Time::now();
  if (SerializedOctreeEnvDataPtr data = std::dynamic_pointer_cast<SerializedOctreeEnvData>(env_data))
  {
    UploadEnvOctree(data->serialized_octree, environment_size, is_3d);
  }
  else if (VoxelgridEnvDataPtr data = std::dynamic_pointer_cast<VoxelgridEnvData>(env_data))
  {
    UploadEnvVoxelgrid(data->occupied_environment, data->empty_environment, is_3d);
  }
  else
  {
    ROS_FATAL("SimulateMultiViewWithInformationGain: received unknown EnvData type.");
    std::exit(1);
  }
  ros::Duration upload_duration = ros::Time::now() - upload_time;
  m_last_upload_time = upload_duration.toSec();

  ros::Time simulation_time = ros::Time::now();
  const uint64 origins_size = origins.size();
  const uint64 orientations_size = orientations.size();
  const uint64 views_size = origins_size * (combine_origins_orientations ? orientations_size : 1);
  const uint64 local_directions_size = local_directions.size();

  if (!combine_origins_orientations && orientations.size() != origins.size())
  {
    ROS_ERROR("nbv_3d_cnn: SimulateMultiViewWithInformationGain: orientations and origins size mismatch.");
    std::exit(1);
  }

  const uint64 MAX_RAYS_PER_BATCH = 1000 * 1000 * 100;
  const uint64 num_rays_per_origin = (combine_origins_orientations ? orientations_size : 1) * local_directions_size;
  const uint64 origins_per_batch = std::max<uint64>(1, MAX_RAYS_PER_BATCH / num_rays_per_origin);
  const uint64 batches = (origins_size / origins_per_batch) + !!(origins_size % origins_per_batch);

  const uint64 scores_size = views_size;

  scores.resize(scores_size);

  for (uint64 i = 0; i < batches; i++)
  {
    const uint64 subrange_min = i * origins_per_batch;
    const uint64 subrange_max = std::min<uint64>((i + 1) * origins_per_batch, origins.size());
    ROS_INFO("nbv_3d_cnn: SimulateMultiViewWithInformationGain: batch from origin %d to origin %d", int(subrange_min), int(subrange_max));
    SimulateMultiViewWithInformationGainNoUpEnv(subrange_min, subrange_max, env_data, environment_size, is_3d,
                                                origins, orientations, local_directions,
                                                combine_origins_orientations, sensor_f, max_range,
                                                min_range, a_priori_occupied_prob, scores);
  }
  ros::Duration simulation_duration = ros::Time::now() - simulation_time;
  m_last_simulation_time = simulation_duration.toSec();
}

void OctreeRaycastOpenCL::SimulateMultiViewWithInformationGainNoUpEnv(const uint64 subrange_min,
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
                                                                      FloatVector & scores)
{
  const uint64 origins_size = subrange_max - subrange_min;
  const uint64 orientations_size = (combine_origins_orientations ? orientations.size() : origins_size);
  const uint64 views_size = origins_size * (combine_origins_orientations ? orientations_size : 1);
  const uint64 local_directions_size = local_directions.size();
  const uint64 num_rays_per_origin = (combine_origins_orientations ? orientations_size : 1) * local_directions_size;

  const uint64 scores_size = views_size;

  if (!m_multi_view_ig_kernel)
  {
    m_multi_view_ig_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "SimulateMultiViewWithInformationGain"));
  }

  if (!m_multi_view_ig_octree_kernel)
  {
    m_multi_view_ig_octree_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "SimulateMultiViewWithInformationGainOctree"));
  }

  if (!m_multi_view_ig_origins || m_multi_view_ig_last_origins_size < origins_size)
  {
    // create size with some margin
    const uint64 new_size = std::max<uint64>(views_size, m_multi_view_ig_last_origins_size + m_multi_view_ig_last_origins_size / 2);
    m_multi_view_ig_origins = CreateBuffer(m_opencl_context, new_size * sizeof(cl_float4),
                                           "m_multi_view_ig_origins");
    m_multi_view_ig_last_origins_size = new_size;
  }

  if (!m_multi_view_ig_orientations || m_multi_view_ig_last_orientations_size < orientations_size)
  {
    // create size with some margin
    const uint64 new_size = std::max<uint64>(views_size, m_multi_view_ig_last_orientations_size +
                                             m_multi_view_ig_last_orientations_size / 2);
    m_multi_view_ig_orientations = CreateBuffer(m_opencl_context, new_size * sizeof(cl_float4),
                                                "m_multi_view_ig_orientations");
    m_multi_view_ig_last_orientations_size = new_size;
  }

  if (!m_multi_view_ig_local_directions || m_multi_view_ig_last_local_size < local_directions_size)
  {
    const uint64 new_size = std::max<uint64>(local_directions_size, m_multi_view_ig_last_local_size + m_multi_view_ig_last_local_size / 2);
    m_multi_view_ig_local_directions = CreateBuffer(m_opencl_context, new_size * sizeof(cl_float4),
                                                    "m_multi_view_ig_local_directions");
    m_multi_view_ig_last_local_size = new_size;
  }

  if (!m_multi_view_ig_scores || m_multi_view_ig_last_score_size < scores_size)
  {
    const uint64 new_size = std::max<uint64>(scores_size, m_multi_view_ig_last_score_size + m_multi_view_ig_last_score_size / 2);
    m_multi_view_ig_scores = CreateBuffer(m_opencl_context, new_size * sizeof(cl_float),
                                          "m_multi_view_ig_scores");
    m_multi_view_ig_last_score_size = new_size;
  }

  if (m_cl_orientations.size() < orientations_size)
    m_cl_orientations.resize(orientations_size);
  if (m_cl_origins.size() < origins_size)
    m_cl_origins.resize(origins_size);
  for (uint64 i = 0; i < orientations_size; i++)
    m_cl_orientations[i] = EToCL(orientations[i + (combine_origins_orientations ? uint64(0) : subrange_min)]);
  for (uint64 i = 0; i < origins_size; i++)
    m_cl_origins[i] = Float3ToFloat4(EToCL(origins[i + subrange_min]));

  if (m_cl_local_directions.size() < local_directions_size)
    m_cl_local_directions.resize(local_directions_size);
  for (uint64 i = 0; i < local_directions_size; i++)
    m_cl_local_directions[i] = Float3ToFloat4(EToCL(local_directions[i]));

  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_view_ig_origins, CL_TRUE, 0,
                                             origins_size * sizeof(cl_float4),
                                             m_cl_origins.data());
  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_view_ig_orientations, CL_TRUE, 0,
                                             orientations_size * sizeof(cl_float4),
                                             m_cl_orientations.data());
  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_view_ig_local_directions, CL_TRUE, 0,
                                             local_directions_size * sizeof(cl_float4),
                                             m_cl_local_directions.data());

  if (!m_fill_uint_kernel)
  {
    m_fill_uint_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "FillUint"));
  }

  {
  //  void kernel FillUint(
  //                       const uint c,
  //                       global uint * to_be_filled
  //                       )

    uint64 c = 0;
    m_fill_uint_kernel->setArg(c++, cl_uint(0));
    m_fill_uint_kernel->setArg(c++, *m_multi_view_ig_scores);

    int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_fill_uint_kernel, cl::NullRange,
                                      cl::NDRange(scores_size), cl::NullRange);
  }

  if (std::dynamic_pointer_cast<SerializedOctreeEnvData>(env_data))
  {
    /*global uint * occupied_octree,
                                                   const uint width, const uint height, const uint depth,
                                                   bool is_3d,
                                                   const float sensor_focal_length,
                                                   global const float3 * origins,
                                                   global const float4 * orientations,
                                                   global const float3 * local_orientations,
                                                   uint num_orientations,
                                                   const uint num_local_orientations,
                                                   const float max_range,
                                                   const float min_range,
                                                   const float a_priori_occupied_prob,
                                                   global float * pixel_score */

    {
      uint64 c = 0;
      m_multi_view_ig_octree_kernel->setArg(c++, *m_multi_ray_serialized_octree);
      m_multi_view_ig_octree_kernel->setArg(c++, cl_uint(environment_size.x()));
      m_multi_view_ig_octree_kernel->setArg(c++, cl_uint(environment_size.y()));
      m_multi_view_ig_octree_kernel->setArg(c++, cl_uint(environment_size.z()));
      m_multi_view_ig_octree_kernel->setArg(c++, cl_uchar(is_3d));
      m_multi_view_ig_octree_kernel->setArg(c++, cl_float(sensor_f));
      m_multi_view_ig_octree_kernel->setArg(c++, *m_multi_view_ig_origins);
      m_multi_view_ig_octree_kernel->setArg(c++, *m_multi_view_ig_orientations);
      m_multi_view_ig_octree_kernel->setArg(c++, *m_multi_view_ig_local_directions);
      m_multi_view_ig_octree_kernel->setArg(c++, cl_uint(combine_origins_orientations ? orientations_size : 0));
      m_multi_view_ig_octree_kernel->setArg(c++, cl_uint(local_directions_size));
      m_multi_view_ig_octree_kernel->setArg(c++, cl_float(max_range));
      m_multi_view_ig_octree_kernel->setArg(c++, cl_float(min_range));
      m_multi_view_ig_octree_kernel->setArg(c++, cl_float(a_priori_occupied_prob));
      m_multi_view_ig_octree_kernel->setArg(c++, *m_multi_view_ig_scores);
    }

    int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_multi_view_ig_octree_kernel, cl::NullRange,
                                      cl::NDRange(scores_size * num_rays_per_origin), cl::NullRange);
  }
  else if (std::dynamic_pointer_cast<VoxelgridEnvData>(env_data))
  {
    {
      /*global const float * occupied_environment,
                                                 global const float * empty_environment,
                                                 const uint width,
                                                 const uint height,
                                                 const uint depth,
                                                 const uchar is_3d,
                                                 const float sensor_focal_length,
                                                 global const float4 * origins,
                                                 global const float4 * orientations,
                                                 global const float4 * local_orientations,
                                                 const uint num_orientations,
                                                 const uint num_local_orientations,
                                                 const float max_range,
                                                 const float min_range,
                                                 const float a_priori_occupied_prob,
                                                 global float * pixel_score*/
      uint64 c = 0;
      m_multi_view_ig_kernel->setArg(c++, *m_multi_ray_occupied_voxelgrid);
      m_multi_view_ig_kernel->setArg(c++, *m_multi_ray_empty_voxelgrid);
      m_multi_view_ig_kernel->setArg(c++, cl_uint(environment_size.x()));
      m_multi_view_ig_kernel->setArg(c++, cl_uint(environment_size.y()));
      m_multi_view_ig_kernel->setArg(c++, cl_uint(environment_size.z()));
      m_multi_view_ig_kernel->setArg(c++, cl_uchar(is_3d));
      m_multi_view_ig_kernel->setArg(c++, cl_float(sensor_f));
      m_multi_view_ig_kernel->setArg(c++, *m_multi_view_ig_origins);
      m_multi_view_ig_kernel->setArg(c++, *m_multi_view_ig_orientations);
      m_multi_view_ig_kernel->setArg(c++, *m_multi_view_ig_local_directions);
      m_multi_view_ig_kernel->setArg(c++, cl_uint(combine_origins_orientations ? orientations_size : 0));
      m_multi_view_ig_kernel->setArg(c++, cl_uint(local_directions_size));
      m_multi_view_ig_kernel->setArg(c++, cl_float(max_range));
      m_multi_view_ig_kernel->setArg(c++, cl_float(min_range));
      m_multi_view_ig_kernel->setArg(c++, cl_float(a_priori_occupied_prob));
      m_multi_view_ig_kernel->setArg(c++, *m_multi_view_ig_scores);
    }

    int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_multi_view_ig_kernel, cl::NullRange,
                                      cl::NDRange(scores_size * num_rays_per_origin), cl::NullRange);
  }

  if (m_cl_scores.size() < scores_size)
    m_cl_scores.resize(scores_size);
  m_opencl_command_queue->enqueueReadBuffer(*m_multi_view_ig_scores, CL_TRUE, 0,
                                            scores_size * sizeof(cl_uint),
                                            m_cl_scores.data());

  for (uint64 i = 0; i < scores_size; i++)
    scores[i + subrange_min] = m_cl_scores[i] / 1000.0f;
}
