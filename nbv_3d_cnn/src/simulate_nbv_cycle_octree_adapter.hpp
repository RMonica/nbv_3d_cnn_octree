#ifndef SIMULATE_NBV_CYCLE_OCTREE_ADAPTER_HPP
#define SIMULATE_NBV_CYCLE_OCTREE_ADAPTER_HPP

#include <nbv_3d_cnn/simulate_nbv_cycle_octree_adapter.h>

// ROS
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

#include <nbv_3d_cnn_octree_msgs/PredictOctreeAction.h>
#include <nbv_3d_cnn_octree_msgs/PredictImageAction.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/server/simple_action_server.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// custom
#include <nbv_3d_cnn/generate_test_dataset_opencl.h>
#include <nbv_3d_cnn/voxelgrid.h>
#include <nbv_3d_cnn/origin_visibility.h>
#include <nbv_3d_cnn_octree_common/octree_load_save.h>

// STL
#include <string>
#include <stdint.h>
#include <vector>
#include <memory>
#include <map>
#include <random>

#include <nbv_3d_cnn/simulate_nbv_cycle_adapter.h>

class OctreeRaycastOpenCL;

class OctreePredict
{
  public:
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_octree_msgs::PredictOctreeAction> PredictOctreeActionClient;
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_octree_msgs::PredictImageAction> PredictImageActionClient;
  typedef std::shared_ptr<PredictOctreeActionClient> PredictOctreeActionClientPtr;
  typedef std::shared_ptr<PredictImageActionClient> PredictImageActionClientPtr;

  typedef uint8_t uint8;
  typedef uint64_t uint64;
  typedef int64_t int64;
  typedef uint32_t uint32;
  typedef std::vector<uint32> Uint32Vector;

  typedef std::map<std::string, std::string> StringStringMap;
  typedef std::pair<std::string, std::string> StringStringPair;

  OctreePredict(ros::NodeHandle & nh, const bool use_octree);

  template <int DIMS>
  cv::Mat PredictOctree(const cv::Mat & unpad_empty_img, const cv::Mat & unpad_occupied_img,
                        const cv::Mat & uninteresting_output_mask,
                        float & prediction_time, uint64 & total_output_values);

  cv::Mat PredictImage(const cv::Mat & unpad_empty_img, const cv::Mat & unpad_occupied_img,
                       const cv::Mat & uninteresting_output_mask, const bool is_3d,
                       float & prediction_time, uint64 & total_output_values);

  bool Predict3d(const Voxelgrid & empty, const Voxelgrid & occupied, Voxelgrid & autocompleted);

  bool Predict2d(const Voxelgrid & empty, const Voxelgrid & occupied, Voxelgrid & autocompleted);

  StringStringMap GetDebugInfo() {return m_last_debug_info; }

  uint64 GetMaxLayers() const {return m_max_layers; }

  private:
  ros::NodeHandle & m_nh;
  PredictOctreeActionClientPtr m_predict_octree_ac;
  PredictImageActionClientPtr m_predict_image_ac;

  uint64 m_max_layers;
  bool m_use_octree;

  StringStringMap m_last_debug_info;
};

class InformationGainOctreeNBV
{
  public:
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > Vector2iVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iVector;
  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;
  typedef std::vector<bool> BoolVector;

  typedef uint64_t uint64;
  typedef std::vector<uint64> Uint64Vector;
  typedef uint32_t uint32;
  typedef std::vector<uint32> Uint32Vector;
  typedef std::vector<float> FloatVector;
  typedef std::map<std::string, std::string> StringStringMap;
  typedef std::pair<std::string, std::string> StringStringPair;

  explicit InformationGainOctreeNBV(ros::NodeHandle & nh,
                                    GenerateTestDatasetOpenCL & opencl,
                                    OctreeRaycastOpenCL & octree_opencl,
                                    GenerateSingleImage & generate_single_image,
                                    std::default_random_engine & random_generator,
                                    const float max_range,
                                    const float min_range,
                                    const float a_priori_occupied_prob,
                                    const Eigen::Vector2i & sensor_resolution,
                                    const float sensor_focal_length,
                                    const bool is_simulated_sensor,
                                    const bool is_omniscient,
                                    const bool is_3d,
                                    const bool use_octree,
                                    const uint64_t accuracy_skip,
                                    const uint64_t sample_fixed_number_of_views);

  bool GetNextBestView(const Voxelgrid & environment,
                       const Voxelgrid & empty,
                       const Voxelgrid & autocompleted,
                       const Voxelgrid & occupied,
                       const uint64 max_layers,
                       const Vector3fVector & skip_origins,
                       const QuaternionfVector &skip_orientations,
                       Eigen::Vector3f & origin,
                       Eigen::Quaternionf & orientation,
                       ViewWithScoreVector * const all_views_with_score);

  bool GetNextBestViewFromList(const Voxelgrid & environment,
                               const Voxelgrid & empty,
                               const Voxelgrid & autocompleted,
                               const Voxelgrid & occupied,
                               const uint64 max_layers,
                               const Vector3fVector & origins,
                               const QuaternionfVector &orientations,
                               const bool combine_origins_orientations,
                               Eigen::Vector3f & origin,
                               Eigen::Quaternionf & orientation,
                               ViewWithScoreVector * const all_views_with_score);

  const Voxelgrid::Ptr GetLastExpectedObservation() const
    {return Voxelgrid::Ptr(new Voxelgrid(m_last_expected_observation)); }
  const Eigen::Quaternionf & GetLastOrientation() const {return m_last_orientation; }
  const Voxelgrid::Ptr GetLastNotSmoothedScores() const
    {return Voxelgrid::Ptr(new Voxelgrid(m_last_not_smoothed_scores)); }

  cv::Mat GetDebugImage(const Voxelgrid &environment) const;

  const StringStringMap & GetDebugInfo() const {return m_last_debug_info; }

  // cv::Vec2f
  const cv::Mat & GetLastPredictedImage() const {return m_last_predicted_image; }
  const cv::Mat & GetLastInitialMask() const {return m_last_initial_mask; }
  const uint64 GetLastMaxLayers() const {return m_last_max_layers; }

  Voxelgrid GetScores() const {return m_last_scores; }
  Voxelgrid4 GetColorScores() const {return Voxelgrid4(); }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  private:
  void ForEachEmpty(const Voxelgrid & empty, const uint64 skip_accuracy,
                    const std::function<void(const uint64, const Eigen::Vector3i &)> &f) const;

  template <int DIMS>
  Uint32Vector VoxelgridToSerializedOctree(const Voxelgrid & empty,
                                           const Voxelgrid & occupied,
                                           const uint64 max_layers);

  ros::NodeHandle & m_nh;

  bool m_is_3d;
  bool m_is_omniscient;
  bool m_is_simulated_sensor;
  bool m_use_octree;
  uint64 m_accuracy_skip;
  uint64 m_sample_fixed_number_of_views;

  Voxelgrid m_last_expected_observation;
  Eigen::Quaternionf m_last_orientation;
  Eigen::Vector3f m_last_origin;
  Voxelgrid m_last_not_smoothed_scores;
  Voxelgrid m_last_scores;

  cv::Mat m_last_predicted_image; // vec2f
  cv::Mat m_last_initial_mask;
  uint64 m_last_max_layers;

  StringStringMap m_last_debug_info;

  float m_max_range;
  float m_min_range;
  float m_a_priori_occupied_prob;
  Eigen::Vector2i m_sensor_resolution;
  float m_sensor_focal_length;

  GenerateTestDatasetOpenCL & m_opencl;
  OctreeRaycastOpenCL & m_octree_opencl;
  GenerateSingleImage & m_generate_single_image;

  std::default_random_engine & m_random_generator;
};

#endif // SIMULATE_NBV_CYCLE_OCTREE_ADAPTER_HPP
