#include "simulate_nbv_cycle_octree_adapter.hpp"

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>

#include <memory>
#include <string>

#include <opencv2/imgproc.hpp>

#include <nbv_3d_cnn_octree_common/image_to_octree.h>
#include <nbv_3d_cnn_octree_common/octree_load_save.h>
#include <nbv_3d_cnn/origin_visibility.h>
#include "simulate_nbv_cycle_octree.h"

#include "octree_raycast_opencl.h"

template <typename T>
const float & SUBSCRIPT(const T & v, const size_t i) {return v[i]; }
template <>
const float & SUBSCRIPT<float>(const float & v, const size_t i) {return v; }
template <typename T>
float & SUBSCRIPT(T & v, const size_t i) {return v[i]; }
template <>
float & SUBSCRIPT<float>(float & v, const size_t i) {return v; }

typedef uint64_t uint64;
typedef uint8_t uint8;

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

template <typename T, int NUM_CHANNELS, bool IS_3D, std::enable_if_t<!IS_3D, bool> = true>
OctreeLoadSave::SparseImage<T> OctreeLevelToSparseImage(const nbv_3d_cnn_octree_msgs::OctreeLevel & level)
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

template <typename T, int NUM_CHANNELS, bool IS_3D, std::enable_if_t<IS_3D, bool> = true>
OctreeLoadSave::SparseImage3D<T> OctreeLevelToSparseImage(const nbv_3d_cnn_octree_msgs::OctreeLevel & level)
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

template <typename T, int NUM_CHANNELS, bool IS_3D, std::enable_if_t<IS_3D, bool> = true >
std::vector<OctreeLoadSave::SparseImage3D<T> > OctreeToSparseOctreeLevels(const nbv_3d_cnn_octree_msgs::Octree & octree_scores)
{
  const uint64 num_levels = octree_scores.levels.size();
  std::vector<OctreeLoadSave::SparseImage3D<T> > result;
  for (uint64 l = 0; l < num_levels; l++)
  {
    OctreeLoadSave::SparseImage3D<T> sparse_image = OctreeLevelToSparseImage<T, NUM_CHANNELS, IS_3D>(octree_scores.levels[l]);
    result.push_back(sparse_image);
  }

  return result;
}

template <typename T, int NUM_CHANNELS, bool IS_3D, std::enable_if_t<!IS_3D, bool> = true >
std::vector<OctreeLoadSave::SparseImage<T> > OctreeToSparseOctreeLevels(const nbv_3d_cnn_octree_msgs::Octree & octree_scores)
{
  const uint64 num_levels = octree_scores.levels.size();
  std::vector<OctreeLoadSave::SparseImage<T> > result;
  for (uint64 l = 0; l < num_levels; l++)
  {
    OctreeLoadSave::SparseImage<T> sparse_image = OctreeLevelToSparseImage<T, NUM_CHANNELS, IS_3D>(octree_scores.levels[l]);
    result.push_back(sparse_image);
  }

  return result;
}

cv::Mat VoxelGridToMat3D(const Voxelgrid & vg)
{
  const uint64 depth = vg.GetDepth();
  const uint64 height = vg.GetHeight();
  const uint64 width = vg.GetWidth();

  const int sizes[3] = {int(depth), int(height), int(width)};

  cv::Mat result(3, sizes, CV_32FC1);
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        result.at<float>(z, y, x) = vg.at(x, y, z);

  return result;
}

cv::Mat VoxelGridToMat2D(const Voxelgrid & vg)
{
  const uint64 height = vg.GetHeight();
  const uint64 width = vg.GetWidth();

  cv::Mat result(height, width, CV_32FC1);
  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
      result.at<float>(y, x) = vg.at(x, y, 0);

  return result;
}

Voxelgrid Mat3DToVoxelgrid(const cv::Mat & mat)
{
  const uint64 depth = mat.size[0], height = mat.size[1], width = mat.size[2];
  Voxelgrid result(width, height, depth);

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        result.at(x, y, z) = mat.at<float>(z, y, x);

  return result;
}

Voxelgrid Mat2DToVoxelgrid(const cv::Mat & mat)
{
  const uint64 height = mat.rows;
  const uint64 width = mat.cols;
  Voxelgrid result(width, height, 1);

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
      result.at(x, y, 0) = mat.at<float>(y, x);

  return result;
}

OctreePredict::OctreePredict(ros::NodeHandle & nh, const bool use_octree): m_nh(nh)
{
  std::string param_string;
  int param_int;

  m_use_octree = use_octree;

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

  m_nh.param<int>("octree_max_layers", param_int, 6); // max octree depth
  m_max_layers = param_int;
}

template <int DIMS>
cv::Mat OctreePredict::PredictOctree(const cv::Mat & unpad_empty_img, const cv::Mat & unpad_occupied_img,
                      const cv::Mat & uninteresting_output_mask,
                      float & prediction_time, uint64 & total_output_values)
{
  ros::Time prepare_time = ros::Time::now();
  const bool is_3d = (DIMS == 3);
  const uint64 unpad_depth = is_3d ? unpad_empty_img.size[0] : 1;
  const uint64 unpad_height = is_3d ? unpad_empty_img.size[1] : unpad_empty_img.rows;
  const uint64 unpad_width = is_3d ? unpad_empty_img.size[2] : unpad_empty_img.cols;

  cv::Mat interesting_input_mask = unpad_empty_img.clone();
  interesting_input_mask = 1.0f; // interesting region of input
  interesting_input_mask = image_to_octree::PadToGreaterPower2<float>(interesting_input_mask);

  cv::Mat empty_img = image_to_octree::PadToGreaterPower2<float>(unpad_empty_img);
  cv::Mat occupied_img = image_to_octree::PadToGreaterPower2<float>(unpad_occupied_img);

  std::vector<cv::Mat> images;
  images.push_back(empty_img);
  images.push_back(occupied_img);
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

  ros::Duration prepare_duration = ros::Time::now() - prepare_time;
  m_last_debug_info.insert(StringStringPair("time_predict_octree_prepare", std::to_string(prepare_duration.toSec())));

  ros::Time call_time = ros::Time::now();

  ROS_INFO("octree_predict: waiting for server.");
  m_predict_octree_ac->waitForServer();

  ROS_INFO("octree_predict: sending goal.");
  m_predict_octree_ac->sendGoal(octree_goal);

  ROS_INFO("octree_predict: waiting for result.");
  m_predict_octree_ac->waitForResult();

  if (m_predict_octree_ac->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_INFO("octree_predict: received result with failure state %s.", m_predict_octree_ac->getState().toString().c_str());
    throw std::string("PredictOctree: received result with failure state");
  }

  ROS_INFO("octree_predict: extracting result");
  nbv_3d_cnn_octree_msgs::PredictOctreeResult octree_result = *(m_predict_octree_ac->getResult());

  ros::Duration call_duration = ros::Time::now() - call_time;
  m_last_debug_info.insert(StringStringPair("time_predict_octree_call", std::to_string(call_duration.toSec())));

  ros::Time postprocess_time = ros::Time::now();

  cv::Mat result_img;
  cv::Mat result_mask;
  OctreeLoadSave::OctreeLevels octree_levels_out;
  const nbv_3d_cnn_octree_msgs::Octree & octree_scores_msg = octree_result.octree_scores;

  const uint64 num_levels = octree_scores_msg.levels.size();
  for (uint64 l = 0; l < num_levels; l++)
  {
    cv::Mat img, mask;
    if (!is_3d)
    {
      OctreeLoadSave::SparseImage<float> sparse_image = OctreeLevelToSparseImage<float, 1, false>(octree_scores_msg.levels[l]);
      img = OctreeLoadSave::SparseImageToImage(sparse_image, mask);
    }
    else
    {
      OctreeLoadSave::SparseImage3D<float> sparse_image = OctreeLevelToSparseImage<float, 1, true>(octree_scores_msg.levels[l]);
      img = OctreeLoadSave::SparseImageToImage3D(sparse_image, mask);
    }

    octree_levels_out.imgs.push_back(img);
    octree_levels_out.img_masks.push_back(mask);

    if (!result_img.data) // first iter
    {
      result_img = img;
      result_mask = mask;
    }
    else
    {
      result_img = result_img + img;
      result_mask = result_mask + mask;
    }
    if (l + 1 < num_levels)
    {
      result_img = image_to_octree::UpsampleImage2D3D<float>(result_img, is_3d);
      result_mask = image_to_octree::UpsampleImage2D3D<uint8>(result_mask, is_3d);
    }
  }

  ros::Time octree_recover_time = ros::Time::now();
  auto sparse_octree_levels = OctreeToSparseOctreeLevels<float, 1, DIMS==3>(octree_scores_msg);
  const simulate_nbv_cycle_octree::Octree<float, DIMS> oct = simulate_nbv_cycle_octree::SparseOctreeLevelsToOctree
    <float, DIMS>(sparse_octree_levels);
  ros::Duration octree_recover_duration = ros::Time::now() - octree_recover_time;
  m_last_debug_info.insert(StringStringPair("time_predict_octree_recover", std::to_string(octree_recover_duration.toSec())));
  m_last_debug_info.insert(StringStringPair("total_octree_cells", std::to_string(oct.GetTotalCells())));

  // check self-consistency
  if (false)
  {
    uint64 counter = 0;
    image_to_octree::IntArray<DIMS> sizes = oct.dense_size;
    image_to_octree::ForeachSize<DIMS>(sizes, 1, [&](const image_to_octree::IntArray<DIMS> & i) -> bool
    {
      const float octree_at = simulate_nbv_cycle_octree::OctreeAt<float, DIMS>(oct, i);
      const float img_at = is_3d ? result_img.at<float>(i[0], i[1], i[2]) : result_img.at<float>(i[0], i[1]);
      if (std::isnan(octree_at) && img_at == 0.0f)
        return true;
      if (octree_at != img_at)
      {
        std::cout << "at ";
        for (uint64 h = 0; h < DIMS; h++)
          std::cout << i[h] << " ";
        std::cout << "img_at " << img_at << " octree_at " << octree_at << std::endl;
        counter++;
      }

      return true;
    });
    if (counter > 0)
      std::exit(1);
  }

  // remove padding
  if (!is_3d)
    result_img = image_to_octree::CropImage<float, 2>(result_img, {0, 0}, {int(unpad_height), int(unpad_width)});
  else
    result_img = image_to_octree::CropImage<float, 3>(result_img, {0, 0, 0}, {int(unpad_depth), int(unpad_height), int(unpad_width)});

  prediction_time = octree_result.prediction_time;
  total_output_values = octree_result.total_output_values;

  ros::Duration postprocess_duration = ros::Time::now() - postprocess_time;
  m_last_debug_info.insert(StringStringPair("time_predict_octree_postprocess", std::to_string(postprocess_duration.toSec())));

  m_last_debug_info.insert(StringStringPair("x_memory_allocated", std::to_string(octree_result.memory_allocated)));

  return result_img;
}

cv::Mat OctreePredict::PredictImage(const cv::Mat & unpad_empty_img, const cv::Mat & unpad_occupied_img,
                     const cv::Mat & uninteresting_output_mask, const bool is_3d, float & prediction_time, uint64 & total_output_values)
{
  nbv_3d_cnn_octree_msgs::PredictImageGoal image_goal;
  ros::Time prepare_time = ros::Time::now();

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
        image_goal.frontier[idx] = unpad_occupied_img.at<float>(y, x);
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
          image_goal.frontier[idx] = unpad_occupied_img.at<float>(z, y, x);
          image_goal.uninteresting[idx] = uninteresting_output_mask.at<float>(z, y, x);
        }
  }

  ros::Duration prepare_duration = ros::Time::now() - prepare_time;
  m_last_debug_info.insert(StringStringPair("time_predict_image_prepare", std::to_string(prepare_duration.toSec())));

  ros::Time call_time = ros::Time::now();

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

  ros::Duration call_duration = ros::Time::now() - call_time;
  m_last_debug_info.insert(StringStringPair("time_predict_image_call", std::to_string(call_duration.toSec())));

  ros::Time postprocess_time = ros::Time::now();

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

  ros::Duration postprocess_duration = ros::Time::now() - postprocess_time;
  m_last_debug_info.insert(StringStringPair("time_predict_image_postprocess", std::to_string(postprocess_duration.toSec())));

  m_last_debug_info.insert(StringStringPair("x_memory_allocated", std::to_string(image_result.memory_allocated)));

  return result_img;
}

bool OctreePredict::Predict3d(const Voxelgrid & empty, const Voxelgrid & occupied, Voxelgrid & autocompleted)
{
  ROS_INFO("voxelgrid_predict: Predict3d start.");

  m_last_debug_info.clear();

  const cv::Mat unpad_empty_img = VoxelGridToMat3D(empty);
  const cv::Mat unpad_occupied_img = VoxelGridToMat3D(occupied);

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
      result_img = PredictImage(unpad_empty_img, unpad_occupied_img, uninteresting_output_mask, true,
                                prediction_time, total_output_values);
    }
    else
    {
      result_img = PredictOctree<3>(unpad_empty_img, unpad_occupied_img,
                                    uninteresting_output_mask, prediction_time, total_output_values);
    }
  }
  catch (const std::string & e)
  {
    ROS_ERROR("OctreePredict: onPredict: %s.", e.c_str());
    return false;
  }

  autocompleted = Mat3DToVoxelgrid(result_img);

  m_last_debug_info.insert(StringStringPair("prediction_time", std::to_string(prediction_time)));
  m_last_debug_info.insert(StringStringPair("total_output_values", std::to_string(total_output_values)));

  return true;
}

bool OctreePredict::Predict2d(const Voxelgrid & empty, const Voxelgrid & occupied, Voxelgrid & autocompleted)
{
  ROS_INFO("voxelgrid_predict: onPredict2d start.");

  m_last_debug_info.clear();

  cv::Mat unpad_empty_img = VoxelGridToMat2D(empty);
  cv::Mat unpad_occupied_img = VoxelGridToMat2D(occupied);

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
      result_img = PredictImage(unpad_empty_img, unpad_occupied_img, uninteresting_output_mask, false,
                                prediction_time, total_output_values);
    }
    else
    {
      result_img = PredictOctree<2>(unpad_empty_img, unpad_occupied_img,
                                    uninteresting_output_mask, prediction_time, total_output_values);
    }
  }
  catch (const std::string & e)
  {
    ROS_ERROR("OctreePredict: onPredict: %s.", e.c_str());
    return false;
  }

  autocompleted = Mat2DToVoxelgrid(result_img);

  m_last_debug_info.insert(StringStringPair("prediction_time", std::to_string(prediction_time)));
  m_last_debug_info.insert(StringStringPair("total_output_values", std::to_string(total_output_values)));

  return true;
}

// -----------------------------------------

InformationGainOctreeNBV::InformationGainOctreeNBV(ros::NodeHandle & nh,
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
                                                   const uint64_t sample_fixed_number_of_views):
  m_nh(nh), m_opencl(opencl), m_octree_opencl(octree_opencl), m_generate_single_image(generate_single_image),
  m_random_generator(random_generator)
{
  m_max_range = max_range;
  m_min_range = min_range;
  m_a_priori_occupied_prob = a_priori_occupied_prob;
  m_sensor_focal_length = sensor_focal_length;
  m_sensor_resolution = sensor_resolution;

  m_is_simulated_sensor = is_simulated_sensor;

  m_is_omniscient = is_omniscient;

  m_is_3d = is_3d;
  m_use_octree = use_octree;
  m_accuracy_skip = accuracy_skip;
  m_sample_fixed_number_of_views = sample_fixed_number_of_views;
}

void InformationGainOctreeNBV::ForEachEmpty(const Voxelgrid & empty, const uint64 skip_accuracy,
                                                   const std::function<void(const uint64 index, const Eigen::Vector3i &)> &f) const
{
  const uint64 width = empty.GetWidth();
  const uint64 height = empty.GetHeight();
  const uint64 depth = empty.GetDepth();

  uint64 i = 0;
  for (uint64 z = 0; z < depth; z += skip_accuracy)
    for (uint64 y = 0; y < height; y += skip_accuracy)
      for (uint64 x = 0; x < width; x += skip_accuracy)
      {
        const Eigen::Vector3i xyz(x, y, z);
        if (!empty.at(xyz))
          continue;

        f(i, xyz);
        i++;
      }
}

template <int DIMS>
InformationGainOctreeNBV::Uint32Vector InformationGainOctreeNBV::VoxelgridToSerializedOctree(const Voxelgrid & empty,
                                                                                             const Voxelgrid & occupied,
                                                                                             const uint64 max_layers)
{
  const uint64 width = empty.GetWidth();
  const uint64 height = empty.GetHeight();
  const uint64 depth = empty.GetDepth();
  int sizes[3] = {int(depth), int(height), int(width)};
  cv::Mat image;
  if (DIMS == 2)
  {
    image = cv::Mat(height, width, CV_32FC2);
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        image.at<cv::Vec2f>(y, x)[0] = occupied.at(x, y, 0);
        image.at<cv::Vec2f>(y, x)[1] = empty.at(x, y, 0);
      }
  }
  else
  {
    image = cv::Mat(3, sizes, CV_32FC2);
    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          image.at<cv::Vec2f>(z, y, x)[0] = occupied.at(x, y, z);
          image.at<cv::Vec2f>(z, y, x)[1] = empty.at(x, y, z);
        }
  }
  image = image_to_octree::PadToGreaterPower2<cv::Vec2f>(image, cv::Vec2f(0.0f, 1.0f));

  cv::Mat initial_mask;
  if (DIMS == 2)
    initial_mask = cv::Mat(height, width, CV_8UC1);
  else
    initial_mask = cv::Mat(3, sizes, CV_8UC1);
  initial_mask = 1;
  initial_mask = image_to_octree::PadToGreaterPower2<uint8>(initial_mask);

  const OctreeLoadSave::OctreeLevels octree_levels = image_to_octree::ImageToOctreeLevelsD<cv::Vec2f, DIMS>(image, initial_mask,
                                                                                                            max_layers, false);
  m_last_predicted_image = image;
  m_last_initial_mask = initial_mask;
  m_last_max_layers = max_layers;
  const simulate_nbv_cycle_octree::Octree<cv::Vec2f, DIMS> oct =
    simulate_nbv_cycle_octree::OctreeLevelsToOctree<cv::Vec2f, DIMS>(octree_levels);

  const Uint32Vector serialized_oct = simulate_nbv_cycle_octree::SerializeOctreeToUint32(oct);

  uint64 counter = 0;
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        const cv::Vec2f octree_at = simulate_nbv_cycle_octree::SerializedOctreeAt<cv::Vec2f, DIMS>(serialized_oct, z, y, x);
        const float empty_at = empty.at(x, y, z);
        const float occupied_at = occupied.at(x, y, z);
        if (empty_at != octree_at[1] || occupied_at != octree_at[0])
        {
          std::cout << "at " << x << " " << y << " " << z << ": ";
          std::cout << "empty_at " << empty_at << " occupied_at " << occupied_at << " octree_at " << octree_at << std::endl;
        }
        counter++;
      }

  return serialized_oct;
}

bool InformationGainOctreeNBV::GetNextBestView(const Voxelgrid & environment,
                                               const Voxelgrid & empty,
                                               const Voxelgrid & autocompleted,
                                               const Voxelgrid & occupied,
                                               const uint64 max_layers,
                                               const Vector3fVector & skip_origins,
                                               const QuaternionfVector & skip_orientations,
                                               Eigen::Vector3f & origin,
                                               Eigen::Quaternionf &orientation,
                                               ViewWithScoreVector * const all_views_with_score)
{
  m_last_debug_info.clear();

  Vector3fVector sample_origins;
  ROS_INFO("nbv_3d_cnn: InformationGainNBVAdapter: computing origins");
  ForEachEmpty(empty, m_accuracy_skip, [&](const uint64 index, const Eigen::Vector3i & xyz)
  {
    sample_origins.push_back(xyz.cast<float>());
  });
  ROS_INFO("nbv_3d_cnn: InformationGainOctreeNBVAdapter: origins: %u", unsigned(sample_origins.size()));

  QuaternionfVector sample_orientations;
  if (m_is_3d)
  {
    sample_orientations = OriginVisibility::GenerateStandardOrientationSet(8);
  }
  else
  {
    const uint64 NUM_ORIENT = 8;
    for (uint64 i = 0; i < NUM_ORIENT; i++)
    {
      const float angle = 2.0f * M_PI * float(i) / float(NUM_ORIENT);
      Eigen::Quaternionf so = Eigen::Quaternionf(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()) *
                                                 Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitX())
                                                 );
      sample_orientations.push_back(so);
    }
  }
  ROS_INFO("nbv_3d_cnn: InformationGainOctreeNBVAdapter: orientations: %u", unsigned(sample_orientations.size()));

  ROS_INFO("nbv_3d_cnn: InformationGainOctreeNBVAdapter: poses: %u", unsigned(sample_origins.size() * sample_orientations.size()));

  QuaternionfVector sampled_fixed_orientations;
  Vector3fVector sampled_fixed_origins;
  if (m_sample_fixed_number_of_views)
  {
    const uint64 total_selectable = sample_orientations.size() * sample_origins.size();
    if (m_sample_fixed_number_of_views >= total_selectable)
    {
      sampled_fixed_orientations.reserve(total_selectable);
      sampled_fixed_origins.reserve(total_selectable);
      for (uint64 i = 0; i < sample_origins.size(); i++)
        for (uint64 h = 0; h < sample_orientations.size(); h++)
        {
          sampled_fixed_origins.push_back(sample_origins[i]);
          sampled_fixed_orientations.push_back(sample_orientations[h]);
        }
    }
    else
    {
      sampled_fixed_orientations.reserve(m_sample_fixed_number_of_views);
      sampled_fixed_origins.reserve(m_sample_fixed_number_of_views);

      for (uint64 vi = 0; vi < m_sample_fixed_number_of_views; vi++)
      {
        std::uniform_int_distribution<uint64> distribution(0, total_selectable - 1);
        const uint64 selected = distribution(m_random_generator);

        sampled_fixed_origins.push_back(sample_origins[selected / sample_orientations.size()]);
        sampled_fixed_orientations.push_back(sample_orientations[selected % sample_orientations.size()]);
      }
    }

    ROS_INFO("nbv_3d_cnn: InformationGainNBVAdapter: sampled %u viewpoints of %u total viewpoints.",
             unsigned(m_sample_fixed_number_of_views), unsigned(total_selectable));
  }

  const Vector3fVector & origins = m_sample_fixed_number_of_views ? sampled_fixed_origins : sample_origins;
  const QuaternionfVector & orientations = m_sample_fixed_number_of_views ? sampled_fixed_orientations : sample_orientations;
  const bool combine_origins_orientations = !m_sample_fixed_number_of_views;

  return GetNextBestViewFromList(environment, empty, autocompleted, occupied, max_layers,
                                 origins, orientations, combine_origins_orientations,
                                 origin, orientation, all_views_with_score);
}

bool InformationGainOctreeNBV::GetNextBestViewFromList(const Voxelgrid & environment,
                                                       const Voxelgrid & empty,
                                                       const Voxelgrid & autocompleted,
                                                       const Voxelgrid & occupied,
                                                       const uint64 max_layers,
                                                       const Vector3fVector & origins,
                                                       const QuaternionfVector & orientations,
                                                       const bool combine_origins_orientations,
                                                       Eigen::Vector3f & origin,
                                                       Eigen::Quaternionf &orientation,
                                                       ViewWithScoreVector * const all_views_with_score)
{
  const uint64 width = environment.GetWidth();
  const uint64 height = environment.GetHeight();
  const uint64 depth = environment.GetDepth();

  Voxelgrid known_occupied = occupied;
  if (m_is_omniscient)
  {
    Voxelgrid environment_shrink = *environment.ErodeCross(Eigen::Vector3i::Ones());
    known_occupied = *known_occupied.Or(environment_shrink);
  }

  Vector3fVector local_directions(m_sensor_resolution.x() * m_sensor_resolution.y());
  for (uint64 y = 0; y < m_sensor_resolution.y(); y++)
    for (uint64 x = 0; x < m_sensor_resolution.x(); x++)
    {
      Eigen::Vector3f dir(float(x) - (float(m_sensor_resolution.x())/2.0f) + 0.5f,
                          float(y) - (float(m_sensor_resolution.y())/2.0f) + 0.5f,
                          m_sensor_focal_length);
      local_directions[x + y * m_sensor_resolution.x()] = dir.normalized();
    }

  ROS_INFO("nbv_3d_cnn: InformationGainOctreeNBVAdapter: octree conversion");

  const Voxelgrid empty_or_not_occupied = *empty.AddedTo(*occupied.MultipliedBy(-1.0f));

  OctreeRaycastOpenCL::EnvDataPtr environment_data;
  Uint32Vector serialized_octree;
  if (m_use_octree)
  {
    if (m_is_3d)
      serialized_octree = VoxelgridToSerializedOctree<3>(empty_or_not_occupied, autocompleted, max_layers);
    else
      serialized_octree = VoxelgridToSerializedOctree<2>(empty_or_not_occupied, autocompleted, max_layers);
    ROS_INFO("nbv_3d_cnn: InformationGainOctreeNBVAdapter: serialized octree size: %d", int(serialized_octree.size()));
    m_last_debug_info.insert(StringStringPair("serialized_octree_size", std::to_string(serialized_octree.size())));

    environment_data.reset(new OctreeRaycastOpenCL::SerializedOctreeEnvData(serialized_octree));
  }
  else
  {
    environment_data.reset(new OctreeRaycastOpenCL::VoxelgridEnvData(autocompleted, empty_or_not_occupied));
  }

  ROS_INFO("nbv_3d_cnn: InformationGainOctreeNBVAdapter: evaluatemultiview");


  const float sensor_focal_length_for_raycast = m_is_simulated_sensor ? 1.0 // set to 1.0 to better match the simulated sensor model
                                                : m_sensor_focal_length;


  const uint64 num_views = combine_origins_orientations ? (origins.size() * orientations.size()) : origins.size();

  FloatVector ovv;
  ros::Time time1 = ros::Time::now();
  m_octree_opencl.SimulateMultiViewWithInformationGain(environment_data, Eigen::Vector3i(width, height, depth), m_is_3d,
                                                       origins, orientations, local_directions,
                                                       combine_origins_orientations,
                                                       sensor_focal_length_for_raycast,
                                                       m_max_range, m_min_range, m_a_priori_occupied_prob, ovv);
  ros::Duration dur1 = ros::Time::now() - time1;
  m_last_debug_info.insert(StringStringPair("opencl_raycast_num_views", std::to_string(num_views)));
  m_last_debug_info.insert(StringStringPair("opencl_raycast_time", std::to_string(dur1.toSec())));
  m_last_debug_info.insert(StringStringPair("opencl_raycast_time_per_view_mus", std::to_string(dur1.toSec() * 1000000.0f / num_views)));
  m_last_debug_info.insert(StringStringPair("opencl_upload_time", std::to_string(m_octree_opencl.GetLastUploadTime())));
  m_last_debug_info.insert(StringStringPair("opencl_simulation_time", std::to_string(m_octree_opencl.GetLastSimulationTime())));

  FloatVector & scores = ovv;

//  FloatVector scores(num_views, 0.0f);
//  for (uint64 i = 0; i < num_views; i++)
//  {
//    for (uint64 h = 0; h < local_directions.size(); h++)
//      scores[i] += ovv[h + i * local_directions.size()];
//  }

  ROS_INFO("nbv_3d_cnn: InformationGainOctreeNBVAdapter: computing scores and visibility matrix");

  Eigen::Vector3f max_origin = -Eigen::Vector3f::Ones();
  Eigen::Quaternionf max_orientation = Eigen::Quaternionf::Identity();
  float max_score;

  for (uint64 i = 0; i < num_views; i++)
  {
    if (i == 0 || scores[i] > max_score)
    {
      max_score = scores[i];

      const uint64 origin_i = combine_origins_orientations ? (i / orientations.size()) : i;
      const uint64 orientation_i = combine_origins_orientations ? (i % orientations.size()) : i;
      max_origin = origins[origin_i];
      max_orientation = orientations[orientation_i];
    }
  }

  ROS_INFO("nbv_3d_cnn: Generating debug image");

  origin = max_origin;
  orientation = max_orientation;

  Voxelgrid expected_observation = *environment.FilledWith(0.0f);

  if (!m_is_3d)
  {

    Eigen::Quaternionf axisq;
    {
      Eigen::Vector3f axis;
      OriginVisibility temp_ov(Eigen::Vector3f(0, 0, 0), 16, m_is_3d);
      axis = temp_ov.GetAxisFromFrame(temp_ov.BearingToFrame(max_orientation * Eigen::Vector3f::UnitZ())).col(2);
      axisq = GenerateSingleImage::Bearing2DToQuat(axis);
    }

    FloatVector dists;
    Vector3fVector ray_bearings;
    const Eigen::Vector2i res(16, m_is_3d ? 16 : 1);
    const uint64 focal = 8;
    Vector3iVector nearest = m_generate_single_image.SimulateView(environment, origin, axisq, focal, res,
                                          m_max_range, m_min_range, dists, ray_bearings);
    m_opencl.FillEnvironmentFromView(expected_observation, origin, axisq, focal, res, dists,
                                     nearest, expected_observation);
  }

  m_last_expected_observation = expected_observation;
  m_last_orientation = max_orientation;
  m_last_origin = max_origin;

  ROS_INFO("nbv_3d_cnn: InformationGainOctreeNBVAdapter: best score %f", float(max_score));
  ROS_INFO_STREAM("nbv_3d_cnn: InformationGainOctreeNBVAdapter: best origin " << origin.transpose() <<
                  " best orientation: " << orientation.vec().transpose() << " " << orientation.w());

  if (all_views_with_score)
  {
    all_views_with_score->clear();
    for (uint64 i = 0; i < num_views; i++)
    {
      ViewWithScore vws;
      const uint64 origin_i = combine_origins_orientations ? (i / orientations.size()) : i;
      const uint64 orientation_i = combine_origins_orientations ? (i % orientations.size()) : i;
      vws.origin = origins[origin_i];
      vws.orientation = orientations[orientation_i];
      vws.score = scores[i];
      all_views_with_score->push_back(vws);
    }
  }

  return max_score > 0.0f;
}

cv::Mat InformationGainOctreeNBV::GetDebugImage(const Voxelgrid & environment) const
{
  const Voxelgrid leo = *(this->GetLastExpectedObservation());
  const Eigen::Quaternionf & axis = this->GetLastOrientation();
  const Eigen::Vector3f ax = axis * Eigen::Vector3f(0.0f, 0.0f, 1.0f);

  const uint64 width = environment.GetWidth();
  const uint64 height = environment.GetHeight();

  cv::Mat cv_color_leo;
  cv::Mat cv_leo = leo.ToOpenCVImage2D();
  cv::cvtColor(cv_leo, cv_color_leo, CV_GRAY2RGB);
  {
    const cv::Point origin(m_last_origin.x(), m_last_origin.y());
    const Eigen::Vector3f o2 = m_last_origin + ax * 15;
    const cv::Point dir(o2.x(), o2.y());
    cv::circle(cv_color_leo, origin,
               10, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
    cv::line(cv_color_leo, origin, dir, cv::Scalar(255, 0, 0));

    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (environment.at(x, y, 0))
        {
          if (leo.at(x, y, 0))
            cv_color_leo.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
          else
            cv_color_leo.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 127);
        }
      }
  }

  return cv_color_leo;
}

// -----------------------------------------

AutocompleteOctreeIGainNBVAdapter::AutocompleteOctreeIGainNBVAdapter(ros::NodeHandle & nh,
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
                                                                     const uint64_t sample_fixed_number_of_views):
  m_nh(nh), m_opencl(opencl), m_generate_single_image(generate_single_image), m_private_nh("~")
{
  m_is_3d = is_3d;

  m_use_octree_for_prediction = use_octree_for_prediction;
  m_use_octree_for_nbv = use_octree_for_nbv;

  m_sample_fixed_number_of_views = sample_fixed_number_of_views;

  int octree_max_layers;
  m_nh.param<int>("octree_max_layers", octree_max_layers, 6); // max octree depth

  m_octree_predict.reset(new OctreePredict(m_nh, m_use_octree_for_prediction));

  m_octree_opencl.reset(new OctreeRaycastOpenCL(m_nh, uint64(octree_max_layers)));

  m_information_gain_octree.reset(new InformationGainOctreeNBV(nh,
                                                               opencl,
                                                               *m_octree_opencl,
                                                               generate_single_image,
                                                               random_generator,
                                                               max_range,
                                                               min_range,
                                                               0.0f,
                                                               sensor_resolution,
                                                               sensor_focal_length,
                                                               is_simulated_sensor,
                                                               false,
                                                               m_is_3d,
                                                               use_octree_for_nbv,
                                                               accuracy_skip,
                                                               m_sample_fixed_number_of_views
                                                               ));
}

bool AutocompleteOctreeIGainNBVAdapter::Predict3d(const Voxelgrid &empty, const Voxelgrid &occupied, Voxelgrid &autocompleted)
{
  const bool success = m_octree_predict->Predict3d(empty, occupied, autocompleted);

  if (!success)
  {
    ROS_ERROR("AutocompleteOctreeIGainNBVAdapter: Predict3d: failed.");
    return false;
  }
  return true;
}

bool AutocompleteOctreeIGainNBVAdapter::Predict(const Voxelgrid & empty, const Voxelgrid & occupied, Voxelgrid & autocompleted)
{
  const bool success = m_octree_predict->Predict2d(empty, occupied, autocompleted);

  if (!success)
  {
    ROS_ERROR("AutocompleteOctreeIGainNBVAdapter: Predict: failed.");
    return false;
  }

  return true;
}

bool AutocompleteOctreeIGainNBVAdapter::GetNextBestView(const Voxelgrid & environment,
                                                        const Voxelgrid & empty,
                                                        const Voxelgrid & occupied,
                                                        const Voxelgrid & frontier,
                                                        const Vector3fVector & skip_origins,
                                                        const QuaternionfVector &skip_orientations,
                                                        Eigen::Vector3f & origin,
                                                        Eigen::Quaternionf &orientation,
                                                        ViewWithScoreVector * const all_views_with_scores)
{


  ROS_INFO("simulate_nbv_cycle: AutocompleteOctreeIGainNBVAdapter: GetNextBestView start.");

  Voxelgrid autocompleted;
  if (m_is_3d)
  {
    if (!Predict3d(empty, occupied, autocompleted))
      return false;
  }
  else
  {
    if (!Predict(empty, occupied, autocompleted))
      return false;
  }
  autocompleted.Clamp(0.0f, 1.0f); // clamp [0,1]
  //m_last_autocompleted_image = autocompleted;

//  autocompleted.Multiply(1.0f - m_probability_cnn_prediction_wrong);
//  autocompleted.Add(m_probability_cnn_prediction_wrong * m_a_priori_occupied_prob);
  autocompleted = *autocompleted.Or(occupied);
  autocompleted = *autocompleted.AndNot(empty);
  m_last_autocompleted_image = autocompleted;

  ROS_INFO("simulate_nbv_cycle: AutocompleteOctreeIGainNBVAdapter: computing information gain.");

  const uint64 max_layers = m_octree_predict->GetMaxLayers();

  bool r = m_information_gain_octree->GetNextBestView(environment, empty, autocompleted, occupied,
                                                      max_layers,
                                                      skip_origins, skip_orientations, origin, orientation,
                                                      all_views_with_scores);

  ROS_INFO("simulate_nbv_cycle: AutocompleteOctreeIGainNBVAdapter: GetNextBestView end.");
  return r;
}

bool AutocompleteOctreeIGainNBVAdapter::GetNextBestViewFromList(const Voxelgrid & environment,
                                                                const Voxelgrid & empty,
                                                                const Voxelgrid & occupied,
                                                                const Voxelgrid & frontier,
                                                                const Vector3fVector & origins,
                                                                const QuaternionfVector & orientations,
                                                                const bool combine_origins_orientations,
                                                                Eigen::Vector3f & origin,
                                                                Eigen::Quaternionf & orientation,
                                                                ViewWithScoreVector * const all_views_with_score)
{
  ROS_INFO("simulate_nbv_cycle: AutocompleteOctreeIGainNBVAdapter: GetNextBestView start.");

  Voxelgrid autocompleted;
  if (m_is_3d)
  {
    if (!Predict3d(empty, occupied, autocompleted))
      return false;
  }
  else
  {
    if (!Predict(empty, occupied, autocompleted))
      return false;
  }
  autocompleted.Clamp(0.0f, 1.0f); // clamp [0,1]
  //m_last_autocompleted_image = autocompleted;

//  autocompleted.Multiply(1.0f - m_probability_cnn_prediction_wrong);
//  autocompleted.Add(m_probability_cnn_prediction_wrong * m_a_priori_occupied_prob);
  autocompleted = *autocompleted.Or(occupied);
  autocompleted = *autocompleted.AndNot(empty);
  m_last_autocompleted_image = autocompleted;

  ROS_INFO("simulate_nbv_cycle: AutocompleteOctreeIGainNBVAdapter: computing information gain.");

  const uint64 max_layers = m_octree_predict->GetMaxLayers();

  bool r = m_information_gain_octree->GetNextBestViewFromList(environment, empty, autocompleted, occupied,
                                                              max_layers,
                                                              origins, orientations, combine_origins_orientations,
                                                              origin, orientation, all_views_with_score);

  ROS_INFO("simulate_nbv_cycle: AutocompleteOctreeIGainNBVAdapter: GetNextBestView end.");
  return r;
}

AutocompleteOctreeIGainNBVAdapter::StringStringMap AutocompleteOctreeIGainNBVAdapter::GetDebugInfo() const
{
  const StringStringMap & p_debug_info = m_octree_predict->GetDebugInfo();
  const StringStringMap & o_debug_info = m_information_gain_octree->GetDebugInfo();

  StringStringMap result;
  result.insert(p_debug_info.begin(), p_debug_info.end());
  result.insert(o_debug_info.begin(), o_debug_info.end());
  return result;
}

void AutocompleteOctreeIGainNBVAdapter::SaveDebugGrids(const std::string & prefix, const Voxelgrid & environment_image) const
{
  Voxelgrid autocompleted_scores = this->GetLastAutocompletedImage();
  autocompleted_scores.Save2D3D(prefix + "autocompleted", m_is_3d);
  if (!m_is_3d)
  {
    cv::Mat debug_image = this->GetDebugImage(environment_image);
    cv::imwrite(prefix + "leo.png", debug_image);
  }

  Voxelgrid::ConstPtr not_smoothed_scores = this->GetLastNotSmoothedScores();
  if (not_smoothed_scores && !not_smoothed_scores->IsEmpty())
  {
    not_smoothed_scores->Save2D3D(prefix + "last_not_smoothed", m_is_3d);
  }

  if (m_information_gain_octree && !(m_information_gain_octree->GetLastPredictedImage().empty()))
  {
    const cv::Mat & image = m_information_gain_octree->GetLastPredictedImage();
    const cv::Mat & initial_mask = m_information_gain_octree->GetLastInitialMask();
    const uint64 max_layers = m_information_gain_octree->GetLastMaxLayers();

    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Mat unified_image = channels[0];
    cv::Mat empty_or_not_occupied = channels[1];
    if (m_is_3d)
    {
      for (uint64 z = 0; z < unified_image.size[0]; z++)
        for (uint64 y = 0; y < unified_image.size[1]; y++)
          for (uint64 x = 0; x < unified_image.size[2]; x++)
          {
            if (empty_or_not_occupied.at<float>(z, y, x) < -0.5f)
              unified_image.at<float>(z, y, x) = 1.1f;
            if (empty_or_not_occupied.at<float>(z, y, x) > 0.5f)
              unified_image.at<float>(z, y, x) = -0.1f;
          }
    }
    else
    {
      for (uint64 y = 0; y < unified_image.rows; y++)
        for (uint64 x = 0; x < unified_image.cols; x++)
        {
          if (empty_or_not_occupied.at<float>(y, x) < -0.5f)
            unified_image.at<float>(y, x) = 1.1f;
          if (empty_or_not_occupied.at<float>(y, x) > 0.5f)
            unified_image.at<float>(y, x) = -0.1f;
        }
    }

    std::ofstream ofile(prefix + "octree.octree");
    if (m_is_3d)
    {
      const OctreeLoadSave::OctreeLevels octree_levels = image_to_octree::ImageToOctreeLevelsD<float, 3>(unified_image, initial_mask,
                                                                                                         max_layers, false);
      OctreeLoadSave::SerializeOctree3D<float>(ofile, octree_levels, NULL, NULL, NULL);
    }
    else
    {
      const OctreeLoadSave::OctreeLevels octree_levels = image_to_octree::ImageToOctreeLevelsD<float, 2>(unified_image, initial_mask,
                                                                                                         max_layers, false);
      OctreeLoadSave::SerializeOctree<float>(ofile, octree_levels, NULL, NULL, NULL);
    }
  }
}

Voxelgrid::Ptr AutocompleteOctreeIGainNBVAdapter::GetLastExpectedObservation() const
  {return m_information_gain_octree->GetLastExpectedObservation(); }
Eigen::Quaternionf AutocompleteOctreeIGainNBVAdapter::GetLastOrientation() const
  {return m_information_gain_octree->GetLastOrientation(); }
Voxelgrid::Ptr AutocompleteOctreeIGainNBVAdapter::GetLastNotSmoothedScores() const
  {return m_information_gain_octree->GetLastNotSmoothedScores(); }

cv::Mat AutocompleteOctreeIGainNBVAdapter::GetDebugImage(const Voxelgrid &environment) const
  {return m_information_gain_octree->GetDebugImage(environment); }

Voxelgrid AutocompleteOctreeIGainNBVAdapter::GetLastAutocompletedImage() const
  {return m_last_autocompleted_image; }

Voxelgrid AutocompleteOctreeIGainNBVAdapter::GetScores() const
  {return m_information_gain_octree->GetScores(); }
Voxelgrid4 AutocompleteOctreeIGainNBVAdapter::GetColorScores() const
  {return m_information_gain_octree->GetColorScores(); }

