

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <nbv_3d_cnn_octree_common/octree_load_save.h>
#include <nbv_3d_cnn_octree_common/image_to_octree.h>

#include <string>
#include <stdint.h>
#include <fstream>
#include <vector>
#include <array>

#define FAIL(...)  {ROS_FATAL(__VA_ARGS__); exit(1);}

using image_to_octree::At;

class ImageToOctree
{
  public:
  typedef uint64_t uint64;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::vector<int> IntVector;
  template <int N>
    using IntArray = std::array<int, N>;
  typedef uint8_t uint8;

  using OctreeLevels = OctreeLoadSave::OctreeLevels;
  using ImagePyramid = OctreeLoadSave::ImagePyramid;
  using SparseMask = OctreeLoadSave::SparseMask;
  template <typename T>
    using SparseImage = OctreeLoadSave::SparseImage<T>;

  ImageToOctree(ros::NodeHandle & nh): m_nh(nh)
  {
    int param_int;
    m_nh.param<std::string>("source_prefix", m_source_prefix, "");
    m_nh.param<std::string>("dest_prefix", m_dest_prefix, "");

    m_nh.param<bool>("is_3d", m_is_3d, false);

    m_nh.param<bool>("with_complete_gt", m_with_complete_gt, false);

    m_nh.param<int>("max_levels", param_int, 3);
    m_max_levels = param_int;

    m_timer = nh.createTimer(ros::Duration(0.0), &ImageToOctree::onTimer, this, false);

    m_counter = 0;
  }

  template <typename T>
  ImagePyramid ImageToImagePyramid(const cv::Mat & image_in, const uint64 max_layers)
  {
    return m_is_3d ? ImageToImagePyramidD<T, 3>(image_in, max_layers) :
                     ImageToImagePyramidD<T, 2>(image_in, max_layers);
  }

  template <typename T, int DIMS>
  ImagePyramid ImageToImagePyramidD(const cv::Mat & image_in, const uint64 max_layers)
  {
    ImagePyramid result;

    cv::Mat image = image_in.clone();
    cv::Mat sq_image;
    cv::pow(image, 2.0, sq_image);

    result.imgs.insert(result.imgs.begin(), image);
    result.sq_imgs.insert(result.sq_imgs.begin(), sq_image);

    for (uint64 i = 1; i < max_layers; i++)
    {
      image = image_to_octree::DownsampleImageAvg<T, DIMS>(image);
      sq_image = image_to_octree::DownsampleImageAvg<T, DIMS>(sq_image);

      result.imgs.insert(result.imgs.begin(), image); // smaller images go first
      result.sq_imgs.insert(result.sq_imgs.begin(), sq_image);
    }

    return result;
  }

  template <typename T>
  ImagePyramid InterestingImageToImagePyramid(const cv::Mat & image_in, const uint64 max_layers, const cv::Mat & interesting_output_mask)
  {
    return m_is_3d ? image_to_octree::InterestingImageToImagePyramidD<T, 3>(image_in, max_layers, interesting_output_mask) :
                     image_to_octree::InterestingImageToImagePyramidD<T, 2>(image_in, max_layers, interesting_output_mask);
  }

  template <typename T>
  OctreeLevels ImageToOctreeLevels(const cv::Mat & image, const cv::Mat & initial_mask, const uint64 max_layers,
                                   const bool mask_only = false)
  {
    if (!m_is_3d)
    {
      return image_to_octree::ImageToOctreeLevelsD<T, 2>(image, initial_mask, max_layers, mask_only);
    }
    else
    {
      return image_to_octree::ImageToOctreeLevelsD<T, 3>(image, initial_mask, max_layers, mask_only);
    }
  }

  cv::Mat LoadImage(const std::string & filename, const bool is_3d)
  {
    cv::Mat img;
    if (!is_3d)
    {
      const std::string fn = filename + ".png";
      img = cv::imread(fn, cv::IMREAD_GRAYSCALE);
      if (!img.data)
      {
        ROS_WARN("image_to_octree: unable to load image %s.", fn.c_str());
        return img;
      }
    }
    else
    {
      const std::string fn = filename + ".binvoxelgrid";
      rmonica_voxelgrid_common::Voxelgrid::Ptr vox =
        rmonica_voxelgrid_common::Voxelgrid::FromFileBinary(fn);
      if (!vox)
      {
        ROS_WARN("image_to_octree: unable to load voxelgrid %s.", fn.c_str());
        return img;
      }

      img = vox->ToOpenCVImage3DF();

      cv::Mat img_out;
      img.convertTo(img_out, CV_8UC1, 255.0f);
      img = img_out;
    }

    return img;
  }

  void onTimer(const ros::TimerEvent &)
  {
    const std::string gt_filename = m_source_prefix + std::to_string(m_counter) + "_environment";
    const std::string empty_filename = m_source_prefix + std::to_string(m_counter) + "_empty";
    const std::string occupied_filename = m_source_prefix + std::to_string(m_counter) + "_occupied";
    const std::string frontier_filename = m_source_prefix + std::to_string(m_counter) + "_frontier";

    ROS_INFO("image_to_octree: loading image %s", gt_filename.c_str());
    cv::Mat gt_img = LoadImage(gt_filename, m_is_3d);
    if (!gt_img.data)
    {
      ROS_FATAL("image_to_octree: unable to load image, ended.");
      m_timer.stop();
      return;
    }
    gt_img = image_to_octree::PadToGreaterPower2<uint8>(gt_img);

    cv::Mat unpad_empty_img = LoadImage(empty_filename, m_is_3d);
    if (!unpad_empty_img.data)
      FAIL("image_to_octree: unable to load image %s, terminating.", empty_filename.c_str());
    cv::Mat empty_img = image_to_octree::PadToGreaterPower2<uint8>(unpad_empty_img);

    cv::Mat unpad_occupied_img = LoadImage(occupied_filename, m_is_3d);
    if (!unpad_occupied_img.data)
      FAIL("image_to_octree: unable to load image %s, terminating.", occupied_filename.c_str());
    cv::Mat occupied_img = image_to_octree::PadToGreaterPower2<uint8>(unpad_occupied_img);

    cv::Mat unpad_frontier_img = LoadImage(frontier_filename, m_is_3d);
    if (!unpad_frontier_img.data)
      FAIL("image_to_octree: unable to load image %s, terminating.", frontier_filename.c_str());
    cv::Mat frontier_img = image_to_octree::PadToGreaterPower2<uint8>(unpad_frontier_img);

    cv::Mat gt_img_float;
    gt_img.convertTo(gt_img_float, CV_32FC1, 1.0f/255.0f);

    cv::Mat interesting_output_mask = 255 - unpad_empty_img - unpad_occupied_img;
    interesting_output_mask = image_to_octree::PadToGreaterPower2<uint8>(interesting_output_mask);
    cv::Mat uninteresting_output_mask = 255 - interesting_output_mask;

    cv::Mat interesting_input_mask = unpad_empty_img.clone();
    interesting_input_mask = 255;
    interesting_input_mask = image_to_octree::PadToGreaterPower2<uint8>(interesting_input_mask);

    cv::Mat interesting_output_mask_float;
    interesting_output_mask.convertTo(interesting_output_mask_float, CV_32FC1, 1.0f/255.0f);

    const uint64 max_levels = m_max_levels;

    const OctreeLoadSave::OctreeLevels gt_octree_levels = ImageToOctreeLevels<float>(gt_img_float, interesting_output_mask, max_levels);

    const OctreeLoadSave::ImagePyramid gt_image_pyramid = InterestingImageToImagePyramid<float>(gt_img_float, max_levels,
                                                                                              interesting_output_mask_float);

    {
      std::string gt_octree_filename_out = m_dest_prefix + std::to_string(m_counter) + "_gt_octree.octree";
      ROS_INFO("image_to_octree: saving octree file %s", gt_octree_filename_out.c_str());
      std::ofstream ofile(gt_octree_filename_out);
      if (!ofile)
        FAIL("Could not create file!");
      if (!m_is_3d)
        OctreeLoadSave::SerializeOctree<float>(ofile, gt_octree_levels, NULL, NULL, &gt_image_pyramid);
      else
        OctreeLoadSave::SerializeOctree3D<float>(ofile, gt_octree_levels, NULL, NULL, &gt_image_pyramid);
      if (!ofile)
        FAIL("Could not write file!");
    }

    // complete gt
    if (m_with_complete_gt)
    {
      cv::Mat complete_gt = gt_img_float;
      // the difference is that I use the input mask instead of the output mask
      // using this gives worse training results, apparently
      const OctreeLoadSave::OctreeLevels gt_octree_levels = ImageToOctreeLevels<float>(complete_gt, interesting_input_mask, max_levels);

      const OctreeLoadSave::ImagePyramid gt_image_pyramid = InterestingImageToImagePyramid<float>(complete_gt, max_levels,
                                                                                                interesting_output_mask_float);

      std::string gt_octree_filename_out = m_dest_prefix + std::to_string(m_counter) + "_complete_gt_octree.octree";
      ROS_INFO("image_to_octree: saving complete octree file %s", gt_octree_filename_out.c_str());
      std::ofstream ofile(gt_octree_filename_out);
      if (!ofile)
        FAIL("Could not create file!");
      if (!m_is_3d)
        OctreeLoadSave::SerializeOctree<float>(ofile, gt_octree_levels, NULL, NULL, &gt_image_pyramid);
      else
        OctreeLoadSave::SerializeOctree3D<float>(ofile, gt_octree_levels, NULL, NULL, &gt_image_pyramid);
      if (!ofile)
        FAIL("Could not write file!");
    }

    cv::Mat empty_img_float, frontier_img_float, occupied_img_float;
    empty_img.convertTo(empty_img_float, CV_32FC1, 1.0f/255.0f);
    frontier_img.convertTo(frontier_img_float, CV_32FC1, 1.0f/255.0f);
    occupied_img.convertTo(occupied_img_float, CV_32FC1, 1.0f/255.0f);
    std::vector<cv::Mat> channels = {empty_img_float, occupied_img_float};
    cv::Mat merged_input_image;
    cv::merge(channels, merged_input_image);
    const OctreeLevels input_octree_levels = ImageToOctreeLevels<cv::Vec2f>(merged_input_image, interesting_input_mask, max_levels);

    OctreeLevels output_uninteresting_masks;
    {
      cv::Mat uninteresting_output_mask_float;
      uninteresting_output_mask.convertTo(uninteresting_output_mask_float, CV_32FC1, 1.0f / 255.0);
      output_uninteresting_masks = ImageToOctreeLevels<float>(uninteresting_output_mask_float, uninteresting_output_mask,
                                                              max_levels, true);
    }

    const OctreeLevels occupied_octree_levels = ImageToOctreeLevels<float>(occupied_img_float, interesting_input_mask, max_levels);

    {
      std::string input_octree_filename_out = m_dest_prefix + std::to_string(m_counter) + "_input_octree.octree";
      ROS_INFO("image_to_octree: saving octree file %s", input_octree_filename_out.c_str());
      std::ofstream ifile(input_octree_filename_out);
      if (!ifile)
        FAIL("Could not create file!");
      if (!m_is_3d)
        OctreeLoadSave::SerializeOctree<cv::Vec2f>(ifile, input_octree_levels, &output_uninteresting_masks.img_masks, NULL, NULL);
      else
        OctreeLoadSave::SerializeOctree3D<cv::Vec2f>(ifile, input_octree_levels, &output_uninteresting_masks.img_masks, NULL, NULL);
      if (!ifile)
        FAIL("Could not write file!");
    }
    {
      std::string occupied_octree_filename_out = m_dest_prefix + std::to_string(m_counter) + "_occupied_octree.octree";
      ROS_INFO("image_to_octree: saving occupied octree file %s", occupied_octree_filename_out.c_str());
      std::ofstream ifile(occupied_octree_filename_out);
      if (!ifile)
        FAIL("Could not create file!");
      if (!m_is_3d)
        OctreeLoadSave::SerializeOctree<float>(ifile, occupied_octree_levels, NULL, NULL, NULL);
      else
        OctreeLoadSave::SerializeOctree3D<float>(ifile, occupied_octree_levels, NULL, NULL, NULL);
      if (!ifile)
        FAIL("Could not write file!");
    }

    if (!m_is_3d)
    {
      for (uint64 l = 0; l < max_levels; l++)
      {
        cv::Mat gt = gt_octree_levels.imgs[l].clone();
        cv::Mat gt_mask = gt_octree_levels.img_masks[l].clone();
        cv::Mat uninteresting_mask = output_uninteresting_masks.img_masks[l].clone();
        cv::Mat gtb;
        cv::Mat gt_maskb;

        gt.convertTo(gtb, CV_8UC1, 255);
        gt_mask.convertTo(gt_maskb, CV_8UC1, 255);
        uninteresting_mask.convertTo(uninteresting_mask, CV_8UC1, 255);

        std::string gt_filename_out = m_dest_prefix + "debug/" + std::to_string(m_counter) + "_" +
            std::to_string(l) + "_gt.png";
        std::string gt_mask_filename_out = m_dest_prefix + "debug/" + std::to_string(m_counter) + "_" +
            std::to_string(l) + "_gt_mask.png";
        std::string uninteresting_mask_filename_out = m_dest_prefix + "debug/" + std::to_string(m_counter) + "_" +
            std::to_string(l) + "_uninteresting.png";

        cv::imwrite(gt_filename_out, gtb);
        cv::imwrite(gt_mask_filename_out, gt_maskb);
        cv::imwrite(uninteresting_mask_filename_out, uninteresting_mask);
      }
    }

    m_counter++;
  }

  private:
  ros::NodeHandle & m_nh;

  std::string m_source_prefix;
  std::string m_dest_prefix;

  uint64 m_max_levels;

  bool m_is_3d;
  bool m_with_complete_gt;

  ros::Timer m_timer;

  uint64 m_counter;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "image_to_octree");

  ros::NodeHandle nh("~");

  ImageToOctree ito(nh);
  ros::spin();
}
