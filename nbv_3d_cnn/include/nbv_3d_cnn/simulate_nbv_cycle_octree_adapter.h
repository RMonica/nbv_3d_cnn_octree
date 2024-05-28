#ifndef SIMULATE_NBV_CYCLE_OCTREE_ADAPTER_H
#define SIMULATE_NBV_CYCLE_OCTREE_ADAPTER_H

// ROS
#include <ros/ros.h>

// custom
#include <nbv_3d_cnn/generate_test_dataset_opencl.h>
#include <nbv_3d_cnn/voxelgrid.h>
#include <nbv_3d_cnn/origin_visibility.h>

// STL
#include <string>
#include <stdint.h>
#include <vector>
#include <memory>
#include <map>
#include <random>

#include "simulate_nbv_cycle_adapter.h"

class InformationGainOctreeNBV;
class OctreeRaycastOpenCL;
class OctreePredict;

class AutocompleteOctreeIGainNBVAdapter: public INBVAdapter
{
  public:
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > Vector2iVector;

  typedef std::shared_ptr<OctreeRaycastOpenCL> OctreeRaycastOpenCLPtr;
  typedef uint64_t uint64;

  explicit AutocompleteOctreeIGainNBVAdapter(ros::NodeHandle & nh,
                                             GenerateTestDatasetOpenCL &opencl,
                                             GenerateSingleImage &generate_single_image,
                                             std::default_random_engine & random_generator,
                                             const float max_range,
                                             const float min_range,
                                             const Eigen::Vector2i & sensor_resolution,
                                             const float sensor_focal_length,
                                             const bool is_simulated_sensor,
                                             const bool is_3d,
                                             const bool use_octree_for_prediction,
                                             const bool use_octree_for_nbv,
                                             const uint64_t accuracy_skip,
                                             const uint64_t sample_fixed_number_of_views);

  bool GetNextBestView(const Voxelgrid & environment,
                       const Voxelgrid & empty,
                       const Voxelgrid & occupied,
                       const Voxelgrid & frontier,
                       const Vector3fVector & skip_origins,
                       const QuaternionfVector & skip_orientations,
                       Eigen::Vector3f & origin,
                       Eigen::Quaternionf & orientation,
                       ViewWithScoreVector * const all_views_with_scores = NULL) override;

  bool GetNextBestViewFromList(const Voxelgrid & environment,
                               const Voxelgrid & empty,
                               const Voxelgrid & occupied,
                               const Voxelgrid & frontier,
                               const Vector3fVector & origins,
                               const QuaternionfVector & orientations,
                               const bool combine_origins_orientations,
                               Eigen::Vector3f & origin,
                               Eigen::Quaternionf & orientation,
                               ViewWithScoreVector * const all_views_with_scores = NULL);

  bool Predict3d(const Voxelgrid &empty, const Voxelgrid &occupied, Voxelgrid &autocompleted);
  bool Predict(const Voxelgrid &empty, const Voxelgrid &occupied, Voxelgrid &autocompleted);

  virtual bool IsRandom() const override {return m_sample_fixed_number_of_views; }

  Voxelgrid::Ptr GetLastExpectedObservation() const;
  Eigen::Quaternionf GetLastOrientation() const;
  Voxelgrid::Ptr GetLastNotSmoothedScores() const;

  cv::Mat GetDebugImage(const Voxelgrid &environment) const;
  void SaveDebugGrids(const std::string & prefix, const Voxelgrid &environment_image) const override;

  Voxelgrid GetLastAutocompletedImage() const;

  StringStringMap GetDebugInfo() const override;

  Voxelgrid GetScores() const override;
  Voxelgrid4 GetColorScores() const override;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  private:
  ros::NodeHandle & m_nh;

  bool m_is_3d;

  std::shared_ptr<OctreePredict> m_octree_predict;

  Voxelgrid m_last_autocompleted_image;

  bool m_use_octree_for_prediction;
  bool m_use_octree_for_nbv;

  uint64 m_sample_fixed_number_of_views;

  nbv_3d_cnn_msgs::FloatsConstPtr m_raw_data;
  ros::CallbackQueue m_raw_data_callback_queue;
  ros::NodeHandle m_private_nh;

  GenerateTestDatasetOpenCL & m_opencl;
  GenerateSingleImage & m_generate_single_image;

  std::shared_ptr<InformationGainOctreeNBV> m_information_gain_octree;

  OctreeRaycastOpenCLPtr m_octree_opencl;
};

#endif // SIMULATE_NBV_CYCLE_OCTREE_ADAPTER_H
