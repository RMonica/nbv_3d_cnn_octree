#include "voxelgrid_predict.h"

#include <ros/ros.h>

#include <nbv_3d_cnn_octree_msgs/PredictOctreeAction.h>
#include <nbv_3d_cnn_octree_msgs/PredictImageAction.h>
#include <nbv_3d_cnn_msgs/Predict3dAction.h>
#include <nbv_3d_cnn_msgs/PredictAction.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/server/simple_action_server.h>
#include <cv_bridge/cv_bridge.h>

#include <memory>
#include <string>

#include <opencv2/imgproc.hpp>

#include <nbv_3d_cnn_octree_common/image_to_octree.h>
#include <nbv_3d_cnn_octree_common/octree_load_save.h>

template <typename T>
const float & SUBSCRIPT(const T & v, const size_t i) {return v[i]; }
template <>
const float & SUBSCRIPT<float>(const float & v, const size_t i) {return v; }
template <typename T>
float & SUBSCRIPT(T & v, const size_t i) {return v[i]; }
template <>
float & SUBSCRIPT<float>(float & v, const size_t i) {return v; }

class VoxelGridPredict
{
  public:
  typedef actionlib::SimpleActionServer<nbv_3d_cnn_msgs::Predict3dAction> Predict3dActionServer;
  typedef actionlib::SimpleActionServer<nbv_3d_cnn_msgs::PredictAction> PredictActionServer;
  typedef std::shared_ptr<Predict3dActionServer> Predict3dActionServerPtr;
  typedef std::shared_ptr<PredictActionServer> PredictActionServerPtr;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn_octree_msgs::PredictOctreeAction> PredictOctreeActionClient;
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_octree_msgs::PredictImageAction> PredictImageActionClient;
  typedef std::shared_ptr<PredictOctreeActionClient> PredictOctreeActionClientPtr;
  typedef std::shared_ptr<PredictImageActionClient> PredictImageActionClientPtr;

  typedef uint8_t uint8;
  typedef uint64_t uint64;
  typedef int64_t int64;

  VoxelGridPredict(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    int param_int;

    m_nh.param<std::string>("predict_3d_action_name", param_string, "predict3d");
    m_predict_3d_as.reset(new Predict3dActionServer(m_nh, param_string, boost::bind(&VoxelGridPredict::onPredict3d, this, _1), false));

    m_nh.param<std::string>("predict_action_name", param_string, "predict");
    m_predict_as.reset(new PredictActionServer(m_nh, param_string, boost::bind(&VoxelGridPredict::onPredict2d, this, _1), false));

    m_nh.param<bool>("use_octree", m_use_octree, false);
    if (m_use_octree)
    {
      m_nh.param<std::string>("predict_octree_action_name", param_string, "predict_octree");
      m_predict_octree_ac.reset(new PredictOctreeActionClient(param_string, true));
    }
    else
    {
      m_nh.param<std::string>("predict_image_action_name", param_string, "predict_image");
      m_predict_image_ac.reset(new PredictImageActionClient(param_string, true));
    }

    m_nh.param<int>("max_layers", param_int, 6); // max octree depth
    m_max_layers = param_int;

    m_predict_3d_as->start();
    m_predict_as->start();
  }

  template<typename T>
  cv::Mat OccupiedFromEmptyAndFrontier(const cv::Mat & empty, const cv::Mat & frontier, bool is_3d)
  {
    cv::Mat exp_empty;
    if (is_3d)
      exp_empty = DilateCross3D<T>(empty);
    else
      cv::dilate(empty, exp_empty, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));

    cv::Mat occupied = exp_empty - empty - frontier;
    return occupied;
  }

  template <typename T>
  cv::Mat DilateCross3D(const cv::Mat & img)
  {
    cv::Mat result = img.clone();

    const int * const size = img.size;
    for (int64 z = 0; z < size[0]; z++)
      for (int64 y = 0; y < size[1]; y++)
        for (int64 x = 0; x < size[2]; x++)
        {
          for (int64 dz = -1; dz <= 1; dz++)
            for (int64 dy = -1; dy <= 1; dy++)
              for (int64 dx = -1; dx <= 1; dx++)
              {
                if ((dx*dx + dy*dy + dz*dz) > 1)
                  continue;
                const int64 nx = x + dx;
                const int64 ny = y + dy;
                const int64 nz = z + dz;
                if (nx < 0 || ny < 0 || nz < 0)
                  continue;
                if (nx >= size[2] || ny >= size[1] || nz >= size[0])
                  continue;

                const T v = img.at<T>(nz, ny, nx);
                result.at<T>(z, y, x) = std::max(v, result.at<T>(z, y, x));
              }
        }

    return result;
  }

  template <typename T, int NUM_CHANNELS>
  nbv_3d_cnn_octree_msgs::OctreeLevel SparseImageToOctreeLevel(const OctreeLoadSave::SparseImage<T> & sparse_img)
  {
    nbv_3d_cnn_octree_msgs::OctreeLevel level;
    level.size.resize(2);
    level.size[0] = sparse_img.height;
    level.size[1] = sparse_img.width;

    level.indices.resize(sparse_img.indices.size() * 2);
    for (uint64 i = 0; i < sparse_img.indices.size(); i++)
    {
      level.indices[i * 2 + 0] = sparse_img.indices[i].y;
      level.indices[i * 2 + 1] = sparse_img.indices[i].x;
    }
    level.values.resize(sparse_img.values.size() * NUM_CHANNELS);
    for (uint64 i = 0; i < sparse_img.values.size(); i++)
    {
      for (int h = 0; h < NUM_CHANNELS; h++)
        level.values[i * NUM_CHANNELS + h] = SUBSCRIPT<T>(sparse_img.values[i], h);
    }

    return level;
  }

  template <typename T, int NUM_CHANNELS>
  nbv_3d_cnn_octree_msgs::OctreeLevel SparseImageToOctreeLevel3D(const OctreeLoadSave::SparseImage3D<T> & sparse_img)
  {
    nbv_3d_cnn_octree_msgs::OctreeLevel level;
    level.size.resize(3);
    level.size[2] = sparse_img.width;
    level.size[1] = sparse_img.height;
    level.size[0] = sparse_img.depth;

    level.indices.resize(sparse_img.indices.size() * 3);
    for (uint64 i = 0; i < sparse_img.indices.size(); i++)
    {
      level.indices[i * 3 + 0] = sparse_img.indices[i].z;
      level.indices[i * 3 + 1] = sparse_img.indices[i].y;
      level.indices[i * 3 + 2] = sparse_img.indices[i].x;
    }
    level.values.resize(sparse_img.values.size() * NUM_CHANNELS);
    for (uint64 i = 0; i < sparse_img.values.size(); i++)
    {
      for (int h = 0; h < NUM_CHANNELS; h++)
        level.values[i * NUM_CHANNELS + h] = SUBSCRIPT<T>(sparse_img.values[i], h);
    }

    return level;
  }

  template <typename T, int NUM_CHANNELS>
  OctreeLoadSave::SparseImage<T> OctreeLevelToSparseImage2D(const nbv_3d_cnn_octree_msgs::OctreeLevel & level)
  {
    OctreeLoadSave::SparseImage<T> sparse_img;

    const uint64 height = level.size[0];
    const uint64 width = level.size[1];

    sparse_img.width = width;
    sparse_img.height = height;

    const uint64 size = level.indices.size() / 2;
    if (level.values.size() != size * NUM_CHANNELS)
    {
      ROS_ERROR("voxelgrid_predict: OctreeLevelToSparseImage: received octree level with values size %d, but %d expected from indices "
                "(num channels is %d).",
                int(level.values.size()), int(size * NUM_CHANNELS), int(NUM_CHANNELS));
      return sparse_img;
    }

    sparse_img.indices.resize(size);
    sparse_img.values.resize(size);

    for (uint64 i = 0; i < size; i++)
    {
      sparse_img.indices[i].y = level.indices[i * 2 + 0];
      sparse_img.indices[i].x = level.indices[i * 2 + 1];
    }
    for (uint64 i = 0; i < size; i++)
    {
      for (int h = 0; h < NUM_CHANNELS; h++)
        SUBSCRIPT<T>(sparse_img.values[i], h) = level.values[i * NUM_CHANNELS + h];
    }

    return sparse_img;
  }

  template <typename T, int NUM_CHANNELS>
  OctreeLoadSave::SparseImage3D<T> OctreeLevelToSparseImage3D(const nbv_3d_cnn_octree_msgs::OctreeLevel & level)
  {
    OctreeLoadSave::SparseImage3D<T> sparse_img;

    const uint64 depth = level.size[0];
    const uint64 height = level.size[1];
    const uint64 width = level.size[2];

    sparse_img.width = width;
    sparse_img.height = height;
    sparse_img.depth = depth;

    const uint64 size = level.indices.size() / 3;
    if (level.values.size() != size * NUM_CHANNELS)
    {
      ROS_ERROR("voxelgrid_predict: OctreeLevelToSparseImage3D: received octree level with values size %d, but %d expected from indices "
                "(num channels is %d).",
                int(level.values.size()), int(size * NUM_CHANNELS), int(NUM_CHANNELS));
      return sparse_img;
    }

    sparse_img.indices.resize(size);
    sparse_img.values.resize(size);

    for (uint64 i = 0; i < size; i++)
    {
      sparse_img.indices[i].z = level.indices[i * 3 + 0];
      sparse_img.indices[i].y = level.indices[i * 3 + 1];
      sparse_img.indices[i].x = level.indices[i * 3 + 2];
    }
    for (uint64 i = 0; i < size; i++)
    {
      for (int h = 0; h < NUM_CHANNELS; h++)
        SUBSCRIPT<T>(sparse_img.values[i], h) = level.values[i * NUM_CHANNELS + h];
    }

    return sparse_img;
  }

  template <int DIMS>
  cv::Mat PredictOctree(const cv::Mat & unpad_empty_img, const cv::Mat & unpad_frontier_img, const cv::Mat & unpad_occupied_img,
                        const cv::Mat & uninteresting_output_mask,
                        float & prediction_time, uint64 & total_output_values)
  {
    const bool is_3d = (DIMS == 3);
    const uint64 unpad_depth = is_3d ? unpad_empty_img.size[0] : 1;
    const uint64 unpad_height = is_3d ? unpad_empty_img.size[1] : unpad_empty_img.rows;
    const uint64 unpad_width = is_3d ? unpad_empty_img.size[2] : unpad_empty_img.cols;

    cv::Mat interesting_input_mask = unpad_empty_img.clone();
    interesting_input_mask = 1.0f; // interesting region of input
    interesting_input_mask = image_to_octree::PadToGreaterPower2<float>(interesting_input_mask);

    cv::Mat frontier_img = image_to_octree::PadToGreaterPower2<float>(unpad_frontier_img);
    cv::Mat empty_img = image_to_octree::PadToGreaterPower2<float>(unpad_empty_img);
    cv::Mat occupied_img = image_to_octree::PadToGreaterPower2<float>(unpad_occupied_img);

    std::vector<cv::Mat> images;
    images.push_back(empty_img);
    images.push_back(frontier_img);
    cv::Mat input_img;
    cv::merge(images, input_img);

    cv::Mat interesting_input_mask_byte;
    interesting_input_mask.convertTo(interesting_input_mask_byte, CV_8UC1, 255);
    cv::Mat uninteresting_output_mask_byte;
    uninteresting_output_mask.convertTo(uninteresting_output_mask_byte, CV_8UC1, 255);

    image_to_octree::OctreeLevels octree_levels =
        image_to_octree::ImageToOctreeLevelsD<cv::Vec2f, DIMS>(input_img, interesting_input_mask_byte, m_max_layers);

    image_to_octree::OctreeLevels uninteresting_levels =
        image_to_octree::ImageToOctreeLevelsD<float, DIMS>(uninteresting_output_mask, uninteresting_output_mask_byte, m_max_layers, true);

    nbv_3d_cnn_octree_msgs::PredictOctreeGoal octree_goal;

    {
      nbv_3d_cnn_octree_msgs::Octree input_octree;
      input_octree.is_3d = is_3d;
      input_octree.num_channels = 2;
      for (uint64 l = 0; l < octree_levels.imgs.size(); l++)
      {
        cv::Mat img = octree_levels.imgs[l];
        cv::Mat mask = octree_levels.img_masks[l];
        nbv_3d_cnn_octree_msgs::OctreeLevel level;
        if (!is_3d)
        {
          OctreeLoadSave::SparseImage<cv::Vec2f> sparse_img = OctreeLoadSave::ImageToSparseImage<cv::Vec2f>(img, mask);
          level = SparseImageToOctreeLevel<cv::Vec2f, 2>(sparse_img);
        }
        else
        {
          OctreeLoadSave::SparseImage3D<cv::Vec2f> sparse_img = OctreeLoadSave::ImageToSparseImage3D<cv::Vec2f>(img, mask);
          level = SparseImageToOctreeLevel3D<cv::Vec2f, 2>(sparse_img);
        }

        input_octree.levels.push_back(level);
      }
      octree_goal.empty_and_frontier = input_octree;
    }

    {
      nbv_3d_cnn_octree_msgs::Octree uninteresting_octree;
      uninteresting_octree.is_3d = is_3d;
      uninteresting_octree.num_channels = 1;
      for (uint64 l = 0; l < octree_levels.imgs.size(); l++)
      {
        cv::Mat img = uninteresting_levels.imgs[l];
        cv::Mat mask = uninteresting_levels.img_masks[l];
        nbv_3d_cnn_octree_msgs::OctreeLevel level;
        if (!is_3d)
        {
          OctreeLoadSave::SparseImage<float> sparse_img = OctreeLoadSave::ImageToSparseImage<float>(img, mask);
          level = SparseImageToOctreeLevel<float, 1>(sparse_img);
        }
        else
        {
          OctreeLoadSave::SparseImage3D<float> sparse_img = OctreeLoadSave::ImageToSparseImage3D<float>(img, mask);
          level = SparseImageToOctreeLevel3D<float, 1>(sparse_img);
        }

        uninteresting_octree.levels.push_back(level);
      }

      octree_goal.uninteresting_octree = uninteresting_octree;
    }

    ROS_INFO("voxelgrid_predict: waiting for server.");
    m_predict_octree_ac->waitForServer();

    ROS_INFO("voxelgrid_predict: sending goal.");
    m_predict_octree_ac->sendGoal(octree_goal);

    ROS_INFO("voxelgrid_predict: waiting for result.");
    m_predict_octree_ac->waitForResult();

    if (m_predict_octree_ac->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      ROS_INFO("voxelgrid_predict: received result with failure state %s.", m_predict_octree_ac->getState().toString().c_str());
      throw std::string("PredictOctree: received result with failure state");
    }

    ROS_INFO("voxelgrid_predict: got result.");
    nbv_3d_cnn_octree_msgs::PredictOctreeResult octree_result = *(m_predict_octree_ac->getResult());

    cv::Mat result_img;
    const nbv_3d_cnn_octree_msgs::Octree & octree_scores = octree_result.octree_scores;
    const uint64 num_levels = octree_scores.levels.size();
    for (uint64 l = 0; l < num_levels; l++)
    {
      cv::Mat img, useless_mask;
      if (!is_3d)
      {
        OctreeLoadSave::SparseImage<float> sparse_image = OctreeLevelToSparseImage2D<float, 1>(octree_scores.levels[l]);
        img = OctreeLoadSave::SparseImageToImage(sparse_image, useless_mask);
      }
      else
      {
        OctreeLoadSave::SparseImage3D<float> sparse_image = OctreeLevelToSparseImage3D<float, 1>(octree_scores.levels[l]);
        img = OctreeLoadSave::SparseImageToImage3D(sparse_image, useless_mask);
      }
      if (!result_img.data) // first iter
        result_img = img;
      else
        result_img = result_img + img;
      if (l + 1 < num_levels)
        result_img = image_to_octree::UpsampleImage2D3D<float>(result_img, is_3d);
    }

    // remove padding
    if (!is_3d)
      result_img = image_to_octree::CropImage<float, 2>(result_img, {0, 0}, {int(unpad_height), int(unpad_width)});
    else
      result_img = image_to_octree::CropImage<float, 3>(result_img, {0, 0, 0}, {int(unpad_depth), int(unpad_height), int(unpad_width)});

    prediction_time = octree_result.prediction_time;
    total_output_values = octree_result.total_output_values;

    return result_img;
  }

  cv::Mat PredictImage(const cv::Mat & unpad_empty_img, const cv::Mat & unpad_frontier_img,
                       const cv::Mat & uninteresting_output_mask, const bool is_3d, float & prediction_time, uint64 & total_output_values)
  {
    nbv_3d_cnn_octree_msgs::PredictImageGoal image_goal;

    uint64 unpad_height, unpad_width, unpad_depth;
    if (!is_3d)
    {
      unpad_height = unpad_empty_img.rows;
      unpad_width = unpad_empty_img.cols;
      unpad_depth = 1;

      image_goal.sizes.resize(2);
      image_goal.sizes[0] = unpad_height;
      image_goal.sizes[1] = unpad_width;
    }
    if (is_3d)
    {
      unpad_depth = unpad_empty_img.size[0];
      unpad_height = unpad_empty_img.size[1];
      unpad_width = unpad_empty_img.size[2];

      image_goal.sizes.resize(3);
      image_goal.sizes[0] = unpad_depth;
      image_goal.sizes[1] = unpad_height;
      image_goal.sizes[2] = unpad_width;
    }
    const uint64 unpad_size = unpad_depth * unpad_height * unpad_width;

    image_goal.empty.resize(unpad_size);
    image_goal.frontier.resize(unpad_size);
    image_goal.uninteresting.resize(unpad_size);

    if (!is_3d)
    {
      for (uint64 y = 0; y < unpad_height; y++)
        for (uint64 x = 0; x < unpad_width; x++)
        {
          const uint64 idx = x + y * unpad_width;
          image_goal.empty[idx] = unpad_empty_img.at<float>(y, x);
          image_goal.frontier[idx] = unpad_frontier_img.at<float>(y, x);
          image_goal.uninteresting[idx] = uninteresting_output_mask.at<float>(y, x);
        }
    }
    if (is_3d)
    {
      for (uint64 z = 0; z < unpad_depth; z++)
        for (uint64 y = 0; y < unpad_height; y++)
          for (uint64 x = 0; x < unpad_width; x++)
          {
            const uint64 idx = x + y * unpad_width + z * unpad_width * unpad_height;
            image_goal.empty[idx] = unpad_empty_img.at<float>(z, y, x);
            image_goal.frontier[idx] = unpad_frontier_img.at<float>(z, y, x);
            image_goal.uninteresting[idx] = uninteresting_output_mask.at<float>(z, y, x);
          }
    }

    ROS_INFO("voxelgrid_predict: waiting for image server.");
    m_predict_image_ac->waitForServer();

    ROS_INFO("voxelgrid_predict: sending goal.");
    m_predict_image_ac->sendGoal(image_goal);

    ROS_INFO("voxelgrid_predict: waiting for result.");
    m_predict_image_ac->waitForResult();

    if (m_predict_image_ac->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      ROS_INFO("voxelgrid_predict: received result with failure state %s.", m_predict_image_ac->getState().toString().c_str());
      throw std::string("PredictImage: received result with failure state");
    }

    ROS_INFO("voxelgrid_predict: got result.");
    nbv_3d_cnn_octree_msgs::PredictImageResult image_result = *(m_predict_image_ac->getResult());

    if (image_result.image_scores.size() != unpad_size)
    {
      ROS_ERROR("voxelgrid_predict: expected image_scores size is %d, received %d",
                int(image_result.image_scores.size()), int(unpad_size));
      throw std::string("PredictImage: received incorrect result.");
    }

    cv::Mat result_img;
    if (!is_3d)
    {
      result_img = cv::Mat(unpad_height, unpad_width, CV_32FC1);

      for (uint64 y = 0; y < unpad_height; y++)
        for (uint64 x = 0; x < unpad_width; x++)
          result_img.at<float>(y, x) = image_result.image_scores[x + y * unpad_width];
    }
    if (is_3d)
    {
      const int sizes[3] = {int(unpad_depth), int(unpad_height), int(unpad_width)};
      result_img = cv::Mat(3, sizes, CV_32FC1);

      for (uint64 z = 0; z < unpad_depth; z++)
        for (uint64 y = 0; y < unpad_height; y++)
          for (uint64 x = 0; x < unpad_width; x++)
            result_img.at<float>(z, y, x) = image_result.image_scores[x + y * unpad_width + z * unpad_width * unpad_height];
    }

    prediction_time = image_result.prediction_time;
    total_output_values = image_result.total_output_values;

    return result_img;
  }

  cv::Mat FloatMultiArrayToMat3D(const std_msgs::Float32MultiArray & msg)
  {
    uint64 depth = 0, height = 0, width = 0;
    for (const std_msgs::MultiArrayDimension & dim : msg.layout.dim)
    {
      if (dim.label == "x")
        width = dim.size;
      else if (dim.label == "y")
        height = dim.size;
      else if (dim.label == "z")
        depth = dim.size;
      else
      {
        ROS_ERROR("voxelgrid_predict: FloatMultiArrayToMat3D: unknown dimension '%s' in message.", dim.label.c_str());
        throw std::string("FloatMultiArrayToMat3D: unexpected dimension in message.");
      }
    }

    if (msg.data.size() != depth * height * width)
    {
      ROS_ERROR("voxelgrid_predict: FloatMultiArrayToMat3D: expected %d elements in message, got %d.",
                int(depth * height * width), int(msg.data.size()));
      throw std::string("FloatMultiArrayToMat3D: incorrect no. of elements in message.");
    }

    const int sizes[3] = {int(depth), int(height), int(width)};

    cv::Mat result(3, sizes, CV_32FC1);
    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
          result.at<float>(z, y, x) = msg.data[x + y * width + z * height * width];

    return result;
  }

  std_msgs::Float32MultiArray Mat3DToFloatMultiArray(const cv::Mat & mat)
  {
    const uint64 depth = mat.size[0], height = mat.size[1], width = mat.size[2];
    std_msgs::Float32MultiArray result;

    result.data.resize(depth * height * width);

    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
          result.data[x + y * width + z * height * width] = mat.at<float>(z, y, x);

    std_msgs::MultiArrayLayout & layout = result.layout;
    std_msgs::MultiArrayDimension dim;
    dim.label = "z";
    dim.size = depth;
    dim.stride = height * width;
    layout.dim.push_back(dim);
    dim.label = "y";
    dim.size = height;
    dim.stride = width;
    layout.dim.push_back(dim);
    dim.label = "x";
    dim.size = width;
    dim.stride = 1;
    layout.dim.push_back(dim);

    return result;
  }

  void onPredict3d(const nbv_3d_cnn_msgs::Predict3dGoalConstPtr goal_ptr)
  {
    const nbv_3d_cnn_msgs::Predict3dGoal goal = *goal_ptr;
    ROS_INFO("voxelgrid_predict: onPredict3d start.");

    cv::Mat unpad_empty_img;
    cv::Mat unpad_frontier_img;
    try
    {
      unpad_empty_img = FloatMultiArrayToMat3D(goal.empty);
      unpad_frontier_img = FloatMultiArrayToMat3D(goal.frontier);
    }
    catch (const std::string & e)
    {
      ROS_ERROR("voxelgrid_predict: onPredict3d: message parsing failed: %s.", e.c_str());
      m_predict_3d_as->setAborted();
      return;
    }

    cv::Mat unpad_occupied_img = OccupiedFromEmptyAndFrontier<float>(unpad_empty_img, unpad_frontier_img, true);

    cv::Mat interesting_output_mask = 1.0 - unpad_empty_img - unpad_occupied_img;
    interesting_output_mask = image_to_octree::PadToGreaterPower2<float>(interesting_output_mask);
    cv::Mat uninteresting_output_mask = 1.0 - interesting_output_mask;

    cv::Mat result_img;
    float prediction_time;
    uint64 total_output_values;
    try
    {
      if (!m_use_octree)
      {
        result_img = PredictImage(unpad_empty_img, unpad_frontier_img, uninteresting_output_mask, true,
                                  prediction_time, total_output_values);
      }
      else
      {
        result_img = PredictOctree<3>(unpad_empty_img, unpad_frontier_img, unpad_occupied_img,
                                      uninteresting_output_mask, prediction_time, total_output_values);
      }
    }
    catch (const std::string & e)
    {
      ROS_ERROR("voxelgrid_predict: onPredict: %s.", e.c_str());
      m_predict_3d_as->setAborted();
      return;
    }

    nbv_3d_cnn_msgs::Predict3dResult result;

    result.scores = Mat3DToFloatMultiArray(result_img);

    { // copy debug stats
      nbv_3d_cnn_msgs::DebugInfo info;
      info.name = "prediction_time";
      info.value = std::to_string(prediction_time);
      result.debug_info.push_back(info);

      info.name = "total_output_values";
      info.value = std::to_string(total_output_values);
      result.debug_info.push_back(info);
    }

    m_predict_3d_as->setSucceeded(result);
  }

  void onPredict2d(const nbv_3d_cnn_msgs::PredictGoalConstPtr goal_ptr)
  {
    const nbv_3d_cnn_msgs::PredictGoal goal = *goal_ptr;
    ROS_INFO("voxelgrid_predict: onPredict2d start.");

    cv_bridge::CvImagePtr empty_image_ptr = cv_bridge::toCvCopy(goal.empty, "8UC1");
    cv::Mat unpad_empty_img = empty_image_ptr->image;
    cv_bridge::CvImagePtr frontier_image_ptr = cv_bridge::toCvCopy(goal.frontier, "8UC1");
    cv::Mat unpad_frontier_img = frontier_image_ptr->image;
    unpad_empty_img.convertTo(unpad_empty_img, CV_32FC1);
    cv::min(unpad_empty_img, 1.0f, unpad_empty_img);
    unpad_frontier_img.convertTo(unpad_frontier_img, CV_32FC1);
    cv::min(unpad_frontier_img, 1.0f, unpad_frontier_img);

    cv::Mat unpad_occupied_img = OccupiedFromEmptyAndFrontier<float>(unpad_empty_img, unpad_frontier_img, false);

    cv::Mat interesting_output_mask = 1.0f - unpad_empty_img - unpad_occupied_img;
    interesting_output_mask = image_to_octree::PadToGreaterPower2<float>(interesting_output_mask);
    cv::Mat uninteresting_output_mask = 1.0f - interesting_output_mask;

    cv::Mat result_img;
    float prediction_time;
    uint64 total_output_values;
    try
    {
      if (!m_use_octree)
      {
        result_img = PredictImage(unpad_empty_img, unpad_frontier_img, uninteresting_output_mask, false,
                                  prediction_time, total_output_values);
      }
      else
      {
        result_img = PredictOctree<2>(unpad_empty_img, unpad_frontier_img, unpad_occupied_img,
                                      uninteresting_output_mask, prediction_time, total_output_values);
      }
    }
    catch (const std::string & e)
    {
      ROS_ERROR("voxelgrid_predict: onPredict: %s.", e.c_str());
      m_predict_as->setAborted();
      return;
    }

    nbv_3d_cnn_msgs::PredictResult result;

    { // copy debug stats
      nbv_3d_cnn_msgs::DebugInfo info;
      info.name = "prediction_time";
      info.value = std::to_string(prediction_time);
      result.debug_info.push_back(info);

      info.name = "total_output_values";
      info.value = std::to_string(total_output_values);
      result.debug_info.push_back(info);
    }

    cv_bridge::CvImage out_cv_image;
    out_cv_image.image = result_img;
    out_cv_image.encoding = "32FC1";
    out_cv_image.header = empty_image_ptr->header;
    out_cv_image.toImageMsg(result.scores);

    m_predict_as->setSucceeded(result);
  }

  private:
  ros::NodeHandle & m_nh;

  Predict3dActionServerPtr m_predict_3d_as;
  PredictActionServerPtr m_predict_as;
  PredictOctreeActionClientPtr m_predict_octree_ac;
  PredictImageActionClientPtr m_predict_image_ac;

  uint64 m_max_layers;
  bool m_use_octree;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "voxelgrid_predict");

  ros::NodeHandle nh("~");
  VoxelGridPredict vgp(nh);

  ros::spin();

  return 0;
}
