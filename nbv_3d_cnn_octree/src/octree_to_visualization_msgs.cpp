#include <nbv_3d_cnn_octree_common/octree_load_save.h>
#include <nbv_3d_cnn_octree_common/image_to_octree.h>

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>

#include <string>
#include <stdint.h>
#include <fstream>
#include <vector>

template <typename T> uint64_t GetChannelsOfType() { return T::channels; }
template <> uint64_t GetChannelsOfType<float>() { return 1; }
template <typename T> float & GetAtOfType(T & f, const uint64_t i) { return f[i]; }
template <> float & GetAtOfType<float>(float & f, const uint64_t i) { return f; }

class OctreeToVisualizationMsgs
{
  public:
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  using OctreeLevels = OctreeLoadSave::OctreeLevels;
  using SparseMask3D = OctreeLoadSave::SparseMask3D;
  using SparseMask = OctreeLoadSave::SparseMask;
  using Point3lu = OctreeLoadSave::Point3lu;
  using Point2lu = OctreeLoadSave::Point2lu;
  template <typename T>
    using SparseImage3D = OctreeLoadSave::SparseImage3D<T>;
  template <typename T>
    using SparseImage = OctreeLoadSave::SparseImage<T>;

  typedef std::shared_ptr<std::ifstream> IFStreamPtr;

  static constexpr const char * const COLOR_MODE_VOXEL_SIZE            = "voxel_size";
  static constexpr const char * const COLOR_MODE_INTENSITY             = "intensity";
  static constexpr const char * const COLOR_MODE_FIXED                 = "fixed";
  static constexpr const char * const COLOR_MODE_VOXEL_SIZE_INTENSITY  = "voxel_size_intensity";
  static constexpr const char * const COLOR_MODE_INTENSITY_CH2_TO_RED  = "intensity_ch2_to_red";
  static constexpr const char * const COLOR_MODE_DEFAULT  = COLOR_MODE_VOXEL_SIZE;


  static constexpr const char * const COLOR_PALETTE_RAINBOW      = "rainbow";
  static constexpr const char * const COLOR_PALETTE_GREEN        = "green";
  static constexpr const char * const COLOR_PALETTE_REV_RAINBOW  = "reverse_rainbow";
  static constexpr const char * const COLOR_PALETTE_DEFAULT      = COLOR_PALETTE_RAINBOW;

  OctreeToVisualizationMsgs(ros::NodeHandle & nh): m_nh(nh)
  {
    int param_int;
    std::string param_string;
    m_nh.param<std::string>("source_filename", m_source_filename, "");
    m_nh.param<std::string>("source2_filename", m_source2_filename, "");


    m_nh.param<std::string>("dest_topic", m_dest_topic, "/octree_visualization");
    m_nh.param<std::string>("namespace", m_namespace, "/octree");
    m_nh.param<std::string>("frame_id", m_frame_id, "map");
    m_nh.param<float>("delay_timer", m_delay_timer, 1.0f);

    m_nh.param<float>("base_voxel_size", m_base_voxel_size, 0.01f);

    m_nh.param<float>("voxel_removal_threshold", m_voxel_removal_threshold, 0.5f);
    m_nh.param<float>("voxel_removal_threshold_max", m_voxel_removal_threshold_max, 2.0f);
    m_nh.param<bool>("negative", m_negative, false);

    m_nh.param<float>("voxel_ch2_removal_threshold", m_voxel_ch2_removal_threshold, 2.0f);
    m_nh.param<float>("voxel_ch2_removal_threshold_max", m_voxel_ch2_removal_threshold_max, -1.0f);

    m_nh.param<std::string>("color_palette", m_color_palette, COLOR_PALETTE_DEFAULT);

    m_nh.param<float>("z_offset", m_z_offset, 0.0f);

    m_nh.param<std::string>("color1", param_string, "1 1 1");
    try { m_color1 = StringToColor(param_string); }
    catch (const std::string e)
    {
      ROS_FATAL("octree_to_visualization_msgs: could not parse color 1: %s", e.c_str());
      std::exit(1);
    }

    m_nh.param<std::string>("color2", param_string, "1 0 0");
    try { m_color2 = StringToColor(param_string); }
    catch (const std::string e)
    {
      ROS_FATAL("octree_to_visualization_msgs: could not parse color 2: %s", e.c_str());
      std::exit(1);
    }

    m_nh.param<int>("channel", param_int, 0);
    m_channel = param_int;

    m_nh.param<std::string>("color_mode", m_color_mode, COLOR_MODE_DEFAULT);
    m_nh.param<float>("intensity_range", m_intensity_range, 1.0f);

    m_nh.param<bool>("with_size_border_color", m_with_size_border_color, false);

    {
      m_nh.param<std::string>("image_crop_2d", param_string, "100 100");
      std::istringstream istr(param_string);
      istr >> m_image_crop_2d.x() >> m_image_crop_2d.y();
      ROS_INFO_STREAM("octree_to_visualization_msgs: image_crop 2d: " << m_image_crop_2d.transpose());
    }

    {
      m_nh.param<std::string>("image_crop_3d", param_string, "100 100 100");
      std::istringstream istr(param_string);
      istr >> m_image_crop_3d.x() >> m_image_crop_3d.y() >> m_image_crop_3d.z();
      ROS_INFO_STREAM("octree_to_visualization_msgs: image_crop 3d: " << m_image_crop_3d.transpose());
    }

    m_marker_pub = m_nh.advertise<visualization_msgs::MarkerArray>(m_dest_topic, 1);

    m_timer = nh.createTimer(ros::Duration(m_delay_timer), &OctreeToVisualizationMsgs::onTimer, this, false);

    m_nh.param<int>("first_counter", param_int, 0);
    m_first_counter = param_int;

    m_counter = m_first_counter;
  }

  Eigen::Vector3f StringToColor(const std::string & str)
  {
    Eigen::Vector3f result;
    std::istringstream istr(str);
    istr >> result.x() >> result.y() >> result.z();
    if (!istr)
      throw std::string(std::string("could not parse color: ") + str);
    return result;
  }

  Eigen::Vector3f HSVtoRGB(const float hue, const float saturation, const float value)
  {
    cv::Mat m = cv::Mat(1, 1, CV_8UC3);
    m.at<cv::Vec3b>(0, 0) = cv::Vec3b(hue * 180, saturation * 255, value * 255);
    cv::Mat m_out;
    cv::cvtColor(m, m_out, cv::COLOR_HSV2RGB);
    return Eigen::Vector3f(m_out.at<cv::Vec3b>(0, 0)[0], m_out.at<cv::Vec3b>(0, 0)[1], m_out.at<cv::Vec3b>(0, 0)[2]) / 255.0f;
  }

  Eigen::Vector3f PointluToVector3f(const Point3lu & pt)
  {
    return Eigen::Vector3f(pt.x, pt.y, pt.z);
  }

  Eigen::Vector3f PointluToVector3f(const Point2lu & pt)
  {
    return Eigen::Vector3f(pt.x, pt.y, 0.0f);
  }

  template <typename T, bool IS_3D, std::enable_if_t<IS_3D, bool> = true >
  void ProcessOctree(std::ifstream & ifile, IFStreamPtr ifile2, const uint64 num_channels)
  {
    ProcessOctreeImpl<T, IS_3D, SparseImage3D<T>, SparseImage3D<float>, SparseMask3D>(ifile, ifile2, num_channels);
  }

  template <typename T, bool IS_3D, std::enable_if_t<!IS_3D, bool> = true >
  void ProcessOctree(std::ifstream & ifile, IFStreamPtr ifile2, const uint64 num_channels)
  {
    ProcessOctreeImpl<T, IS_3D, SparseImage<T>, SparseImage<float>, SparseMask>(ifile, ifile2, num_channels);
  }

  template <typename T, bool IS_3D, std::enable_if_t<IS_3D, bool> = true >
  SparseImage3D<T> ImageToSparseImage2D3D(const cv::Mat & img, const cv::Mat & mask)
  {
    return OctreeLoadSave::ImageToSparseImage3D<T>(img, mask);
  }

  template <typename T, bool IS_3D, std::enable_if_t<!IS_3D, bool> = true >
  SparseImage<T> ImageToSparseImage2D3D(const cv::Mat & img, const cv::Mat & mask)
  {
    return OctreeLoadSave::ImageToSparseImage<T>(img, mask);
  }

  template <typename T, bool IS_3D >
  bool DeserializeOctree(std::ifstream & ifile,
                         std::vector<SparseImage3D<T> > & sparse_images, std::vector<SparseMask3D > & uninteresting_masks)
  {
    return OctreeLoadSave::DeserializeOctree3D<T>(ifile, sparse_images, uninteresting_masks, NULL, NULL);
  }

  template <typename T, bool IS_3D >
  bool DeserializeOctree(std::ifstream & ifile,
                         std::vector<SparseImage<T> > & sparse_images, std::vector<SparseMask> & uninteresting_masks)
  {
    return OctreeLoadSave::DeserializeOctree<T>(ifile, sparse_images, uninteresting_masks, NULL, NULL);
  }

  template<typename T>
  cv::Mat DilateCross(const cv::Mat & empty, bool is_3d)
  {
    cv::Mat exp_empty;
    if (is_3d)
      exp_empty = DilateCross3D<T>(empty);
    else
      cv::dilate(empty, exp_empty, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));

    return exp_empty;
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

  Eigen::Vector3f GetColorForOctreeLevel(const float l, const float saturation, const float value)
  {
    if (m_color_palette == COLOR_PALETTE_RAINBOW)
      return HSVtoRGB(1.0f - l, saturation, value);
    if (m_color_palette == COLOR_PALETTE_REV_RAINBOW)
      return HSVtoRGB(l * 5.0f / 6.0f, saturation, value);
    if (m_color_palette == COLOR_PALETTE_GREEN)
      return HSVtoRGB(2.0f / 6.0f, saturation, value);

    ROS_FATAL("octree_to_visualization_msgs: unknown color palette: %s", m_color_palette.c_str());
    std::exit(1);
  }

  template <typename T, bool IS_3D, typename SparseImageType, typename SparseImageFloat, typename SparseMaskType>
  void ProcessOctreeImpl(std::ifstream & ifile, IFStreamPtr ifile2, const uint64 num_channels)
  {
    std::vector<SparseImageType> sparse_images;
    std::vector<SparseMaskType> uninteresting_masks;

    const bool success = DeserializeOctree<T, IS_3D>(ifile, sparse_images, uninteresting_masks);
    if (!success)
    {
      ROS_FATAL("octree_to_image: unable to load octree, terminating.");
      exit(1);
    }

    if (ifile2)
    {
      std::vector<SparseImageFloat> sparse_images2;
      std::vector<SparseMaskType> uninteresting_masks2;

      const bool success = DeserializeOctree<float, IS_3D>(*ifile2, sparse_images2, uninteresting_masks2);
      if (!success)
      {
        ROS_FATAL("octree_to_image: unable to load octree2, terminating.");
        exit(1);
      }

      const uint64 max_levels = sparse_images.size();
      cv::Mat image = image_to_octree::SparseImagesToImage<T>(sparse_images); // empty and frontiers
      cv::Mat occupied = image_to_octree::SparseImagesToImage<float>(sparse_images2); // occupied

      std::vector<cv::Mat> image_split;
      cv::split(image, image_split);
      cv::Mat empty = image_split[0];
      cv::Mat frontier = image_split[1];

      cv::Mat ones = cv::Mat::ones(empty.dims, empty.size, CV_32FC1);
      cv::Mat byte_ones = cv::Mat::ones(empty.dims, empty.size, CV_8UC1);

      cv::Mat unknown = ones - occupied - empty;
      unknown = unknown * 0.1f + occupied;

      cv::Mat complete;
      std::vector<cv::Mat> image_merge;
      image_merge.push_back(occupied);
      image_merge.push_back(unknown);
      cv::merge(image_merge, complete);

      if (IS_3D)
      {
        for (int z = 0; z < byte_ones.size[0]; z++)
          for (int y = 0; y < byte_ones.size[1]; y++)
            for (int x = 0; x < byte_ones.size[2]; x++)
            {
              if (x >= m_image_crop_3d.x() || y >= m_image_crop_3d.y() || z > m_image_crop_3d.z())
                byte_ones.at<uint8>(z, y, x) = 0;
            }
      }
      if (!IS_3D)
      {
        for (int y = 0; y < byte_ones.size[0]; y++)
          for (int x = 0; x < byte_ones.size[1]; x++)
          {
            if (x >= m_image_crop_2d.x() || y >= m_image_crop_2d.y())
              byte_ones.at<uint8>(y, x) = 0;
          }
      }

      const uint64 DIMS = IS_3D ? 3 : 2;
      OctreeLevels complete_octree = image_to_octree::ImageToOctreeLevelsD<T, DIMS>(complete, byte_ones, max_levels);

      sparse_images.clear();
      uninteresting_masks.clear();
      for (uint64 i = 0; i < max_levels; i++)
      {
        SparseImageType sparse_image = ImageToSparseImage2D3D<T, IS_3D>(complete_octree.imgs[i], complete_octree.img_masks[i]);

        sparse_images.push_back(sparse_image);
      }
    }

    const uint64 octree_levels = sparse_images.size();

    const ros::Time now = ros::Time::now();

    visualization_msgs::MarkerArray msg;

    visualization_msgs::Marker del_marker;
    del_marker.ns = m_namespace;
    del_marker.id = 0;
    del_marker.type = visualization_msgs::Marker::CUBE;
    del_marker.action = visualization_msgs::Marker::DELETEALL;

    msg.markers.push_back(del_marker);

    uint64 incremental_id = 1;
    const float BORDER_THICKNESS = IS_3D ? 0.001f : 0.1f;
    const float BORDER_WIDTH_RATIO = 0.1f;

    for (uint64 li = 0; li < octree_levels; li++)
    {
      float current_voxel_size = std::pow(2.0f, octree_levels - li - 1) * m_base_voxel_size;

      visualization_msgs::Marker border_markers[3][2];
      for (uint64 dim = 0; dim < 3; dim++)
        for (uint64 hor_vert = 0; hor_vert < 2; hor_vert++)
        {
          visualization_msgs::Marker & border_marker = border_markers[dim][hor_vert];
          Eigen::Vector3f cube_size = Eigen::Vector3f::Ones() * current_voxel_size;
          cube_size[dim] = BORDER_THICKNESS;
          cube_size[(dim + hor_vert + 1) % 3] = BORDER_WIDTH_RATIO * current_voxel_size;

          border_marker.header.frame_id = m_frame_id;
          border_marker.header.stamp = now;
          border_marker.header.seq = incremental_id;
          border_marker.ns = m_namespace;
          border_marker.id = incremental_id++;
          border_marker.type = visualization_msgs::Marker::CUBE_LIST;
          border_marker.action = visualization_msgs::Marker::ADD;
          border_marker.pose.orientation.w = 1.0f; // identity pose
          border_marker.scale.x = cube_size.x();
          border_marker.scale.y = cube_size.y();
          border_marker.scale.z = cube_size.z();

          {
            Eigen::Vector3f color = GetColorForOctreeLevel(float(li) / octree_levels, 1.0f, 1.0f);
            border_marker.color.r = color.x();
            border_marker.color.g = color.y();
            border_marker.color.b = color.z();
            border_marker.color.a = 1.0f;
          }
        }

      visualization_msgs::Marker marker;
      marker.header.frame_id = m_frame_id;
      marker.header.stamp = now;
      marker.header.seq = incremental_id;
      marker.ns = m_namespace;
      marker.id = incremental_id++;
      marker.type = visualization_msgs::Marker::CUBE_LIST;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.orientation.w = 1.0f; // identity pose
      marker.scale.x = current_voxel_size;
      marker.scale.y = current_voxel_size;
      marker.scale.z = IS_3D ? current_voxel_size : m_base_voxel_size;
      if (m_color_mode == COLOR_MODE_VOXEL_SIZE)
      {
        const float saturation = 0.7f;
        Eigen::Vector3f color = GetColorForOctreeLevel(float(li) / octree_levels, saturation, 1.0f);
        marker.color.r = color.x();
        marker.color.g = color.y();
        marker.color.b = color.z();
        marker.color.a = 1.0f;
      }

      const SparseImageType & level = sparse_images[li];

      uint64 channel = m_channel;

      if (channel >= GetChannelsOfType<T>())
      {
        ROS_ERROR("octree_to_visualization_msgs: requested channel %d does not exists, using 0 instead.", int(channel));
        channel = 0;
      }

      for (uint64 i = 0; i < level.indices.size(); i++)
      {
        const auto ipt = level.indices[i];
        T value = level.values[i];
        // do not add if value < 0.5
        {
          bool f_g = true;
          f_g = f_g && (GetAtOfType(value, channel) > m_voxel_removal_threshold &&
                        GetAtOfType(value, channel) < m_voxel_removal_threshold_max);
          if (m_color_mode == COLOR_MODE_INTENSITY_CH2_TO_RED)
            f_g = f_g || (GetAtOfType(value, 1) > m_voxel_removal_threshold);
          if (f_g == m_negative)
            continue;
        }
        Eigen::Vector3f pt = (PointluToVector3f(ipt) + Eigen::Vector3f(1.0f, 1.0f, IS_3D ? 1.0f : 0.0f) * 0.5f) * current_voxel_size;
        if (!IS_3D)
          pt.y() *= -1.0f;
        geometry_msgs::Point gpt;
        gpt.x = pt.x();
        gpt.y = pt.y();
        gpt.z = pt.z() + m_z_offset;

        marker.points.push_back(gpt);

        if (m_with_size_border_color)
        {
          for (uint64 sign_i = 0; sign_i < 2; sign_i++)
            for (uint64 fsign_i = 0; fsign_i < 2; fsign_i++)
              for (uint64 dim = 0; dim < 3; dim++)
                for (uint64 hor_vert = 0; hor_vert < 2; hor_vert++)
                {
                  if (!IS_3D && dim != 2)
                    continue; // display only top and bottom if 3D

                  visualization_msgs::Marker & border_marker = border_markers[dim][hor_vert];
                  const float sign = sign_i ? 1.0f : -1.0f;
                  const float fsign = fsign_i ? 1.0f : -1.0f;
                  Eigen::Vector3f position = Eigen::Vector3f::Zero();
                  if (IS_3D)
                    position[dim] = (current_voxel_size / 2.0f + BORDER_THICKNESS / 2.0f) * sign;
                  else
                    position[dim] = (m_base_voxel_size / 2.0f + BORDER_THICKNESS / 2.0f) * sign;
                  position[(dim + hor_vert + 1) % 3] = (current_voxel_size / 2.0f - BORDER_WIDTH_RATIO * current_voxel_size / 2.0f) * fsign;
                  position += pt;

                  geometry_msgs::Point bpt;
                  bpt.x = position.x();
                  bpt.y = position.y();
                  bpt.z = position.z() + m_z_offset;
                  border_marker.points.push_back(bpt);
                }
        }

        if (m_color_mode == COLOR_MODE_INTENSITY || m_color_mode == COLOR_MODE_VOXEL_SIZE_INTENSITY ||
            m_color_mode == COLOR_MODE_INTENSITY_CH2_TO_RED || m_color_mode == COLOR_MODE_FIXED)
        {
          const float v = GetAtOfType(value, channel);
          float intensity;
          if (!m_negative)
            intensity = (v - m_voxel_removal_threshold) / (1.0f - m_voxel_removal_threshold);
          else
            intensity = v / m_voxel_removal_threshold;
          intensity = m_intensity_range * intensity + (1.0f - m_intensity_range);

          std_msgs::ColorRGBA color;

          if (m_color_mode == COLOR_MODE_VOXEL_SIZE_INTENSITY)
          {
            float saturation = 1.0f;
//            if (v >= 1.0f)
//              saturation = 0.7f;
            Eigen::Vector3f hsv_color = GetColorForOctreeLevel(float(li) / octree_levels, saturation, intensity);
            color.r = hsv_color.x();
            color.g = hsv_color.y();
            color.b = hsv_color.z();
            color.a = 1.0f;
          }
          else if (m_color_mode == COLOR_MODE_INTENSITY)
          {
            color.r = intensity * m_color1.x() + (1.0f - intensity) * m_color2.x();
            color.g = intensity * m_color1.y() + (1.0f - intensity) * m_color2.y();
            color.b = intensity * m_color1.z() + (1.0f - intensity) * m_color2.z();
            color.a = 1.0f;
          }
          else if (m_color_mode == COLOR_MODE_INTENSITY_CH2_TO_RED)
          {
            const float v2 = GetAtOfType(value, 1);
            color.r = (1.0f - v2) * intensity * m_color1.x() + v2 * m_color2.x();
            color.g = (1.0f - v2) * intensity * m_color1.y() + v2 * m_color2.y();
            color.b = (1.0f - v2) * intensity * m_color1.z() + v2 * m_color2.z();
            color.a = 1.0f;
          }
          else if (m_color_mode == COLOR_MODE_FIXED)
          {
            color.r = m_color1.x();
            color.g = m_color1.y();
            color.b = m_color1.z();
            color.a = 1.0f;
          }

          marker.colors.push_back(color);
        }
      }

      ROS_INFO("octree_to_visualization_msgs: layer %d has %d cubes.", int(li), int(marker.points.size()));
      if (marker.points.size())
        msg.markers.push_back(marker);

      for (uint64 dim = 0; dim < 3; dim++)
        for (uint64 hor_vert = 0; hor_vert < 2; hor_vert++)
          if (border_markers[dim][hor_vert].points.size())
            msg.markers.push_back(border_markers[dim][hor_vert]);
    }

    m_marker_pub.publish(msg);
    ROS_INFO("octree_to_visualization_msgs: marker published.");
  }

  void onTimer(const ros::TimerEvent &)
  {
    const std::string source_filename = m_source_filename;

    std::ifstream ifile(source_filename, std::ios::binary);

    ROS_INFO("octree_to_visualization_msgs: loading octree '%s'", source_filename.c_str());

    uint64 num_channels;
    bool is_3d;
    {
      uint64 useless_version, useless_num_levels, useless_num_fields;
      const bool success = OctreeLoadSave::DeserializeOctreeHeader(ifile, useless_version, useless_num_levels, num_channels,
                                                                   useless_num_fields, is_3d);

      if (!success && m_counter == m_first_counter)
      {
        ROS_FATAL("octree_to_visualization_msgs: unable to load first octree '%s', terminating.", source_filename.c_str());
        exit(1);
      }

      if (!success)
      {
        ROS_WARN("octree_to_visualization_msgs: unable to load octree, ended.");
        m_timer.stop();
        return;
      }

      ifile.seekg(0); // return to beginning of file for next call
    }

    IFStreamPtr ifile2;
    if (!m_source2_filename.empty())
    {
      ROS_INFO("octree_to_visualization_msgs: loading second octree '%s'.", m_source2_filename.c_str());
      ifile2.reset(new std::ifstream(m_source2_filename, std::ios::binary));
      if (!*ifile2)
      {
        ROS_FATAL("octree_to_visualization_msgs: unable to load second octree '%s', terminating.", m_source2_filename.c_str());
        exit(1);
      }
    }

    ROS_INFO("octree_to_visualization_msgs: loading octree %s with %d channels", source_filename.c_str(), int(num_channels));

    if (!is_3d)
    {
      ROS_INFO("octree_to_visualization_msgs: processing 2D octree.");
      if (num_channels == 1)
        ProcessOctree<float, false>(ifile, ifile2, num_channels);
      else if (num_channels == 2)
        ProcessOctree<cv::Vec2f, false>(ifile, ifile2, num_channels);
      else if (num_channels == 3)
        ProcessOctree<cv::Vec3f, false>(ifile, ifile2, num_channels);
      else if (num_channels == 4)
        ProcessOctree<cv::Vec4f, false>(ifile, ifile2, num_channels);
      else
      {
        ROS_FATAL("octree_to_visualization_msgs: unsupported number of channels: %d channels", int(num_channels));
        exit(1);
      }
    }
    else
    {
      ROS_INFO("octree_to_visualization_msgs: processing 3D octree.");
      if (num_channels == 1)
        ProcessOctree<float, true>(ifile, ifile2, num_channels);
      else if (num_channels == 2)
        ProcessOctree<cv::Vec2f, true>(ifile, ifile2, num_channels);
      else if (num_channels == 3)
        ProcessOctree<cv::Vec3f, true>(ifile, ifile2, num_channels);
      else if (num_channels == 4)
        ProcessOctree<cv::Vec4f, true>(ifile, ifile2, num_channels);
      else
      {
        ROS_FATAL("octree_to_visualization_msgs: unsupported number of channels: %d channels", int(num_channels));
        exit(1);
      }
    }

    m_counter++;
    m_timer.stop();
  }

  private:
  ros::NodeHandle & m_nh;

  std::string m_source_filename;
  std::string m_source2_filename;
  std::string m_dest_topic;

  std::string m_namespace;
  std::string m_frame_id;

  ros::Timer m_timer;
  float m_delay_timer;

  ros::Publisher m_marker_pub;

  uint64 m_counter;
  uint64 m_first_counter;

  uint64 m_channel;

  std::string m_color_mode;
  bool m_with_size_border_color;
  std::string m_color_palette;

  Eigen::Vector3f m_color1;
  Eigen::Vector3f m_color2;

  Eigen::Vector2i m_image_crop_2d;
  Eigen::Vector3i m_image_crop_3d;

  float m_z_offset;

  float m_base_voxel_size;
  float m_voxel_removal_threshold;
  float m_voxel_removal_threshold_max;
  float m_voxel_ch2_removal_threshold;
  float m_voxel_ch2_removal_threshold_max;
  bool m_negative;
  float m_intensity_range;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "octree_to_visualization_msgs");

  ros::NodeHandle nh("~");

  OctreeToVisualizationMsgs otvm(nh);
  ros::spin();
}

