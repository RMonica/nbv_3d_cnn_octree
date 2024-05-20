#include "octree_to_image.h"
#include <nbv_3d_cnn_octree_common/octree_load_save.h>

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <stdint.h>
#include <fstream>
#include <vector>

class OctreeToImage
{
  public:
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  using OctreeLevels = OctreeLoadSave::OctreeLevels;
  using SparseMask = OctreeLoadSave::SparseMask;
  template <typename T>
    using SparseImage = OctreeLoadSave::SparseImage<T>;

  struct Segment
  {
    uint64 x1 = 0;
    uint64 y1 = 0;
    uint64 x2 = 0;
    uint64 y2 = 0;
    uint64 layer = 0;

    Segment(const uint64 x1, const uint64 y1, const uint64 x2, const uint64 y2, const uint64 layer):
      x1(x1), y1(y1), x2(x2), y2(y2), layer(layer) {}

    Segment() {}

    std::string ToTabString() const
    {
      return std::to_string(x1) + "\t" +
             std::to_string(y1) + "\t" +
             std::to_string(x2) + "\t" +
             std::to_string(y2) + "\t" +
             std::to_string(layer);
    }

    Segment operator*(const uint64 m) const
    {
      Segment result(x1 * m, y1 * m, x2 * m, y2 * m, layer);
      return result;
    }
  };
  typedef std::vector<Segment> SegmentVector;

  OctreeToImage(ros::NodeHandle & nh): m_nh(nh)
  {
    int param_int;
    m_nh.param<std::string>("source_prefix", m_source_prefix, "");
    m_nh.param<std::string>("source_suffix", m_source_suffix, "");
    m_nh.param<std::string>("dest_prefix", m_dest_prefix, "");
    m_nh.param<std::string>("dest_suffix", m_dest_suffix, "");

    m_nh.param<int>("crop_x", param_int, 0);
    m_crop_x = param_int;
    m_nh.param<int>("crop_y", param_int, 0);
    m_crop_y = param_int;

    m_timer = nh.createTimer(ros::Duration(0.0), &OctreeToImage::onTimer, this, false);


    m_nh.param<int>("first_counter", param_int, 0);
    m_first_counter = param_int;

    m_counter = m_first_counter;
  }

  template <typename T>
  void ProcessOctree(std::ifstream & ifile, const uint64 num_channels)
  {
    std::vector<OctreeLoadSave::SparseImage<T> > sparse_images;
    std::vector<OctreeLoadSave::SparseMask> uninteresting_masks;
    const bool success = OctreeLoadSave::DeserializeOctree<T>(ifile, sparse_images, uninteresting_masks, NULL, NULL);

    if (!success)
    {
      ROS_FATAL("octree_to_image: unable to load octree body, terminating.");
      exit(1);
    }

    const std::string dest_img_filename = m_dest_prefix + std::to_string(m_counter) + m_dest_suffix + ".png";
    const std::string dest_lines_filename = m_dest_prefix + std::to_string(m_counter) + m_dest_suffix + ".lines";

    cv::Mat reconst_image;

    const uint64 max_levels = sparse_images.size();

    SegmentVector segments;

    for (uint64 l = 0; l < max_levels; l++)
    {
      const OctreeLoadSave::SparseImage<T> & simg = sparse_images[l];
      cv::Mat mask;

      if (l == 0)
      {
        reconst_image = OctreeLoadSave::SparseImageToImage<T>(simg, mask);
      }
      else
      {
        cv::Mat reconst_image2;
        cv::resize(reconst_image, reconst_image2, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
        for (Segment & s : segments)
          s = s * 2;
        reconst_image = reconst_image2;
        cv::Mat new_image = OctreeLoadSave::SparseImageToImage<T>(simg, mask);
        reconst_image += new_image;
      }

      for (const OctreeLoadSave::Point2lu idx : simg.indices)
      {
        const uint64 x = idx.x;
        const uint64 y = idx.y;
        Segment top    (0 + x, 0 + y, 1 + x, 0 + y, l);
        Segment bottom (0 + x, 1 + y, 1 + x, 1 + y, l);
        Segment left   (0 + x, 0 + y, 0 + x, 1 + y, l);
        Segment right  (1 + x, 0 + y, 1 + x, 1 + y, l);
        segments.push_back(top);
        segments.push_back(bottom);
        segments.push_back(left);
        segments.push_back(right);
      }
    }

    if (m_crop_x && m_crop_y)
    {
      cv::Rect crop(0, 0, m_crop_x, m_crop_y);
      reconst_image = reconst_image(crop);

      SegmentVector old_segments;
      old_segments.swap(segments);
      for (const Segment & s : old_segments)
        if ((s.x1 <= m_crop_x && s.y1 <= m_crop_x) ||
            (s.x2 <= m_crop_y && s.y2 <= m_crop_y))
          segments.push_back(s);
    }

    if (num_channels == 2) // expand to 3 channels
    {
      std::vector<cv::Mat> channels;
      cv::split(reconst_image, channels);
      channels.push_back(cv::Mat(channels[0].rows, channels[0].cols, CV_32FC1, float(0)));
      cv::merge(channels, reconst_image);
    }

    cv::Mat reconst_image_int;
    reconst_image.convertTo(reconst_image_int, CV_8U, 255);
    ROS_INFO("octree_to_image: saving octree file %s", dest_img_filename.c_str());
    const bool write_success = cv::imwrite(dest_img_filename, reconst_image_int);
    if (!write_success)
    {
      ROS_FATAL("octree_to_image: unable to save file.");
      exit(1);
    }

    ROS_INFO("octree_to_image: saving segments file %s", dest_lines_filename.c_str());
    std::ofstream ofile(dest_lines_filename);
    for (SegmentVector::const_reverse_iterator iter = segments.rbegin(); iter != segments.rend(); iter++)
      ofile << iter->ToTabString() << "\n";
  }

  void onTimer(const ros::TimerEvent &)
  {
    const std::string source_filename = m_source_prefix + std::to_string(m_counter) + m_source_suffix + ".octree";

    std::ifstream ifile(source_filename, std::ios::binary);

    uint64 num_channels;
    {
      uint64 useless_version, useless_num_levels, useless_num_fields;
      bool is_3d;
      const bool success = OctreeLoadSave::DeserializeOctreeHeader(ifile, useless_version, useless_num_levels, num_channels,
                                                                   useless_num_fields, is_3d);

      if (!success && m_counter == m_first_counter)
      {
        ROS_FATAL("octree_to_image: unable to load first octree %s, terminating.", source_filename.c_str());
        exit(1);
      }

      if (!success)
      {
        ROS_WARN("octree_to_image: unable to load octree, ended.");
        m_timer.stop();
        return;
      }

      if (is_3d)
      {
        ROS_FATAL("octree_to_image: 3D octree detected, but only 2D octree is supported.");
        m_timer.stop();
        return;
      }

      ifile.seekg(0); // return to beginning of file for next call
    }

    ROS_INFO("octree_to_image: loading octree %s with %d channels", source_filename.c_str(), int(num_channels));

    if (num_channels == 1)
      ProcessOctree<float>(ifile, num_channels);
    else if (num_channels == 2)
      ProcessOctree<cv::Vec2f>(ifile, num_channels);
    else if (num_channels == 3)
      ProcessOctree<cv::Vec3f>(ifile, num_channels);
    else if (num_channels == 4)
      ProcessOctree<cv::Vec4f>(ifile, num_channels);
    else
    {
      ROS_FATAL("octree_to_image: unsupported number of channels: %d channels", int(num_channels));
      exit(1);
    }

    m_counter++;
  }

  private:
  ros::NodeHandle & m_nh;

  std::string m_source_prefix;
  std::string m_dest_prefix;
  std::string m_source_suffix;
  std::string m_dest_suffix;

  ros::Timer m_timer;

  uint64 m_counter;
  uint64 m_first_counter;

  uint64 m_crop_x;
  uint64 m_crop_y;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "octree_to_image");

  ros::NodeHandle nh("~");

  OctreeToImage oti(nh);
  ros::spin();
}

