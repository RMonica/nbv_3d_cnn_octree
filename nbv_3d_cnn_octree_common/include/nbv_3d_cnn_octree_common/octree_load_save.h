#ifndef OCTREE_LOAD_SAVE_H
#define OCTREE_LOAD_SAVE_H

#include <ostream>
#include <istream>

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <stdint.h>
#include <vector>
#include <typeinfo>
#include <array>

namespace OctreeLoadSave
{
typedef uint8_t uint8;
typedef uint64_t uint64;

struct OctreeLevels
{
  std::vector<cv::Mat> imgs;
  std::vector<cv::Mat> img_masks;
};

struct ImagePyramid
{
  std::vector<cv::Mat> imgs;      // means
  std::vector<cv::Mat> sq_imgs;   // means of squares
  std::vector<cv::Mat> interest_weights;
};

struct Point2lu
{
  uint64 y = 0;
  uint64 x = 0;

  Point2lu() {}
  Point2lu(uint64 y, uint64 x) {this->x = x; this->y = y; }

  std::array<int, 2> ToIntArray() const {return {int(y), int(x)}; }

  bool operator<(const Point2lu & other) const
  {
    if (y < other.y) return true;
    if (y > other.y) return false;
    if (x < other.x) return true;
    if (x > other.x) return false;
    return false;
  }

  bool operator==(const Point2lu & other) const
  {
    if (y != other.y) return false;
    if (x != other.x) return false;
    return true;
  }

  Point2lu operator/(const uint64 other) const
  {
    Point2lu result = *this;
    result.x /= other;
    result.y /= other;
    return result;
  }

  Point2lu operator*(const uint64 other) const
  {
    Point2lu result = *this;
    result.x *= other;
    result.y *= other;
    return result;
  }
};

struct Point3lu
{
  uint64 x = 0;
  uint64 y = 0;
  uint64 z = 0;

  Point3lu() {}
  Point3lu(uint64 z, uint64 y, uint64 x) {this->x = x; this->y = y; this->z = z; }

  std::array<int, 3> ToIntArray() const {return {int(z), int(y), int(x)}; }

  bool operator<(const Point3lu & other) const
  {
    if (z < other.z) return true;
    if (z > other.z) return false;
    if (y < other.y) return true;
    if (y > other.y) return false;
    if (x < other.x) return true;
    if (x > other.x) return false;
    return false;
  }

  bool operator==(const Point3lu & other) const
  {
    if (x != other.x) return false;
    if (y != other.y) return false;
    if (z != other.z) return false;
    return true;
  }

  Point3lu operator/(const uint64 other) const
  {
    Point3lu result = *this;
    result.x /= other;
    result.y /= other;
    result.z /= other;
    return result;
  }

  Point3lu operator*(const uint64 other) const
  {
    Point3lu result = *this;
    result.x *= other;
    result.y *= other;
    result.z *= other;
    return result;
  }
};

template <typename T>
struct SparseImage
{
  uint64 width = 0;
  uint64 height = 0;

  std::array<int, 2> GetSizeArray() const {return {int(height), int(width)}; }

  std::vector<Point2lu> indices;
  std::vector<T> values;
};

template <typename T>
struct SparseImage3D
{
  uint64 width = 0;
  uint64 height = 0;
  uint64 depth = 0;

  std::array<int, 3> GetSizeArray() const {return {int(depth), int(height), int(width)}; }

  std::vector<Point3lu> indices;
  std::vector<T> values;
};

struct SparseMask
{
  uint64 width = 0;
  uint64 height = 0;

  std::array<int, 2> GetSizeArray() const {return {int(height), int(width)}; }

  std::vector<Point2lu> indices;

  template <typename T>
  void CopyDimsFromImage(const SparseImage<T> & si)
  {
    width = si.width;
    height = si.height;
  }
};

struct SparseMask3D
{
  uint64 width = 0;
  uint64 height = 0;
  uint64 depth = 0;

  std::array<int, 3> GetSizeArray() const {return {int(depth), int(height), int(width)}; }

  std::vector<Point3lu> indices;

  template <typename T>
  void CopyDimsFromImage(const SparseImage3D<T> & si)
  {
    width = si.width;
    height = si.height;
    depth = si.depth;
  }
};

template <typename T>
SparseImage<T> ImageToSparseImage(const cv::Mat & img, const cv::Mat & mask)
{
  const uint64 width = img.cols;
  const uint64 height = img.rows;

  SparseImage<T> result;
  result.width = width;
  result.height = height;

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (mask.at<uint8>(y, x))
      {
        result.indices.push_back(Point2lu(y, x));
        result.values.push_back(img.at<T>(y, x));
      }
    }

  return result;
}

template <typename T>
SparseImage3D<T> ImageToSparseImage3D(const cv::Mat & img, const cv::Mat & mask)
{
  const uint64 width = img.size[2];
  const uint64 height = img.size[1];
  const uint64 depth = img.size[0];

  SparseImage3D<T> result;
  result.width = width;
  result.height = height;
  result.depth = depth;

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (mask.at<uint8>(z, y, x))
        {
          result.indices.push_back(Point3lu(z, y, x));
          result.values.push_back(img.at<T>(z, y, x));
        }
      }

  return result;
}

template <typename T>
SparseImage3D<T> ImageToSparseImage3DAutoMask(const cv::Mat & img)
{
  const uint64 width = img.size[2];
  const uint64 height = img.size[1];
  const uint64 depth = img.size[0];

  SparseImage3D<T> result;
  result.width = width;
  result.height = height;
  result.depth = depth;

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (img.at<T>(z, y, x) != T(T() * 0))
        {
          result.indices.push_back(Point3lu(z, y, x));
          result.values.push_back(img.at<T>(z, y, x));
        }
      }

  return result;
}

template <typename T>
SparseImage<T> ImageToSparseImageAutoMask(const cv::Mat & img)
{
  const uint64 width = img.cols;
  const uint64 height = img.rows;

  SparseImage<T> result;
  result.width = width;
  result.height = height;

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (img.at<T>(y, x) != T(T() * 0))
      {
        result.indices.push_back(Point2lu(y, x));
        result.values.push_back(img.at<T>(y, x));
      }
    }

  return result;
}

template <typename T>
cv::Mat SparseImageToImage(const SparseImage<T> & simg, cv::Mat & mask)
{
  const uint64 width = simg.width;
  const uint64 height = simg.height;

  cv::Mat result;
  if (typeid(T) == typeid(float))
    result = cv::Mat(height, width, CV_32FC1, 0.0f);
  else if (typeid(T) == typeid(cv::Vec2f))
    result = cv::Mat(height, width, CV_32FC2, cv::Vec2f(0.0f, 0.0f));
  else if (typeid(T) == typeid(cv::Vec3f))
    result = cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
  else if (typeid(T) == typeid(cv::Vec4f))
    result = cv::Mat(height, width, CV_32FC4, cv::Vec4f(0.0f, 0.0f, 0.0f, 0.0f));
  else
  {
    ROS_FATAL("SparseImageToImage: Unknown Type Id %s", typeid(T).name());
    exit(1);
  }

  mask = cv::Mat(height, width, CV_8UC1, uint8(false));

  const uint64 size = simg.indices.size();

  for (uint64 i = 0; i < size; i++)
  {
    const uint64 x = simg.indices[i].x;
    const uint64 y = simg.indices[i].y;
    const T value = simg.values[i];

    result.at<T>(y, x) = value;
    mask.at<uint8>(y, x) = true;
  }

  return result;
}

template <typename T>
cv::Mat SparseImageToImage3D(const SparseImage3D<T> & simg, cv::Mat & mask)
{
  const uint64 width = simg.width;
  const uint64 height = simg.height;
  const uint64 depth = simg.depth;

  int sizes[] = {int(depth), int(height), int(width)};

  cv::Mat result;
  if (typeid(T) == typeid(float))
    result = cv::Mat(3, sizes, CV_32FC1, 0.0f);
  else if (typeid(T) == typeid(cv::Vec2f))
    result = cv::Mat(3, sizes, CV_32FC2, cv::Vec2f(0.0f, 0.0f));
  else if (typeid(T) == typeid(cv::Vec3f))
    result = cv::Mat(3, sizes, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
  else if (typeid(T) == typeid(cv::Vec4f))
    result = cv::Mat(3, sizes, CV_32FC4, cv::Vec4f(0.0f, 0.0f, 0.0f, 0.0f));
  else
  {
    ROS_FATAL("SparseImageToImage3D: Unknown Type Id %s", typeid(T).name());
    exit(1);
  }

  mask = cv::Mat(3, sizes, CV_8UC1, uint8(false));

  const uint64 size = simg.indices.size();

  for (uint64 i = 0; i < size; i++)
  {
    const uint64 x = simg.indices[i].x;
    const uint64 y = simg.indices[i].y;
    const uint64 z = simg.indices[i].z;
    const T value = simg.values[i];

    result.at<T>(z, y, x) = value;
    mask.at<uint8>(z, y, x) = true;
  }

  return result;
}

SparseMask MaskToSparseMask(const cv::Mat & mask);
SparseMask3D MaskToSparseMask3D(const cv::Mat & mask);

enum class SerializeOctreeField
{
  FIELD_OCTREE_LEVELS = 0,
  FIELD_FOCUS_MASKS   = 1,
  FIELD_IMAGE_PYRAMID = 2,
  FIELD_WEIGHTED_IMAGE_PYRAMID = 3,
};

template <typename T>
void WriteSparseImage(std::ostream & ofile, const SparseImage<T> & sparse_img)
{
  ofile.write((const char *)&sparse_img.height, sizeof(sparse_img.height));
  ofile.write((const char *)&sparse_img.width, sizeof(sparse_img.width));

  const uint64 size = sparse_img.indices.size();
  ofile.write((const char *)&size, sizeof(size));
  for (uint64 i = 0; i < size; i++)
  {
    const uint64 y = sparse_img.indices[i].y;
    ofile.write((const char *)&y, sizeof(y));
  }
  for (uint64 i = 0; i < size; i++)
  {
    const uint64 x = sparse_img.indices[i].x;
    ofile.write((const char *)&x, sizeof(x));
  }
  for (uint64 i = 0; i < size; i++)
  {
    const T v = sparse_img.values[i];
    ofile.write((const char *)&v, sizeof(v));
  }
}

template <typename T>
void WriteSparseImage3D(std::ostream & ofile, const SparseImage3D<T> & sparse_img)
{
  ofile.write((const char *)&sparse_img.depth, sizeof(sparse_img.depth));
  ofile.write((const char *)&sparse_img.height, sizeof(sparse_img.height));
  ofile.write((const char *)&sparse_img.width, sizeof(sparse_img.width));

  const uint64 size = sparse_img.indices.size();
  ofile.write((const char *)&size, sizeof(size));
  for (uint64 i = 0; i < size; i++)
  {
    const uint64 z = sparse_img.indices[i].z;
    ofile.write((const char *)&z, sizeof(z));
  }
  for (uint64 i = 0; i < size; i++)
  {
    const uint64 y = sparse_img.indices[i].y;
    ofile.write((const char *)&y, sizeof(y));
  }
  for (uint64 i = 0; i < size; i++)
  {
    const uint64 x = sparse_img.indices[i].x;
    ofile.write((const char *)&x, sizeof(x));
  }
  for (uint64 i = 0; i < size; i++)
  {
    const T v = sparse_img.values[i];
    ofile.write((const char *)&v, sizeof(v));
  }
}

template <typename T>
void ReadSparseImage(std::istream & ifile, SparseImage<T> & sparse_img)
{
  uint64 height, width, size;
  ifile.read((char *)&height, sizeof(height));
  ifile.read((char *)&width, sizeof(width));
  ifile.read((char *)&size, sizeof(size));

  sparse_img.height = height;
  sparse_img.width = width;

  sparse_img.indices.resize(size);
  sparse_img.values.resize(size);

  for (uint64 i = 0; i < size; i++)
  {
    uint64 y;
    ifile.read((char *)&y, sizeof(y));
    sparse_img.indices[i].y = y;
  }
  for (uint64 i = 0; i < size; i++)
  {
    uint64 x;
    ifile.read((char *)&x, sizeof(x));
    sparse_img.indices[i].x = x;
  }
  for (uint64 i = 0; i < size; i++)
  {
    T v;
    ifile.read((char *)&v, sizeof(v));
    sparse_img.values[i] = v;
  }
}

template <typename T>
void ReadSparseImage3D(std::istream & ifile, SparseImage3D<T> & sparse_img)
{
  uint64 depth, height, width, size;
  ifile.read((char *)&depth, sizeof(depth));
  ifile.read((char *)&height, sizeof(height));
  ifile.read((char *)&width, sizeof(width));
  ifile.read((char *)&size, sizeof(size));

  sparse_img.depth = depth;
  sparse_img.height = height;
  sparse_img.width = width;

  sparse_img.indices.resize(size);
  sparse_img.values.resize(size);

  for (uint64 i = 0; i < size; i++)
  {
    uint64 z;
    ifile.read((char *)&z, sizeof(z));
    sparse_img.indices[i].z = z;
  }
  for (uint64 i = 0; i < size; i++)
  {
    uint64 y;
    ifile.read((char *)&y, sizeof(y));
    sparse_img.indices[i].y = y;
  }
  for (uint64 i = 0; i < size; i++)
  {
    uint64 x;
    ifile.read((char *)&x, sizeof(x));
    sparse_img.indices[i].x = x;
  }
  for (uint64 i = 0; i < size; i++)
  {
    T v;
    ifile.read((char *)&v, sizeof(v));
    sparse_img.values[i] = v;
  }
}

template <typename T>
void SerializeOctree(std::ostream & ofile, const OctreeLevels & octree_levels, const std::vector<cv::Mat> * const focus_masks,
                     const ImagePyramid * const image_pyramid, const ImagePyramid * const weighted_image_pyramid)
{
  typedef SerializeOctreeField Field;

  const std::string magic = "OCTREE2D";

  ofile << magic;
  const uint64 version = 1;
  ofile.write((const char *)&version, sizeof(version));

  const uint64 num_levels = octree_levels.imgs.size();
  ofile.write((const char *)&num_levels, sizeof(num_levels));

  const uint64 type_size = sizeof(T) / sizeof(float);
  ofile.write((const char *)&type_size, sizeof(type_size));

  uint64 num_fields = 1;
  if (focus_masks)
    num_fields += 1;
  if (image_pyramid)
    num_fields += 1;
  if (weighted_image_pyramid)
    num_fields += 1;
  ofile.write((const char *)&num_fields, sizeof(num_fields));

  {
    const uint64 octree_levels_field = uint64(Field::FIELD_OCTREE_LEVELS);
    ofile.write((const char *)&octree_levels_field, sizeof(octree_levels_field));

    for (uint64 l = 0; l < num_levels; l++)
    {
      const cv::Mat img = octree_levels.imgs[l];
      const cv::Mat mask = octree_levels.img_masks[l];
      SparseImage<T> sparse_img = ImageToSparseImage<T>(img, mask);

      WriteSparseImage<T>(ofile, sparse_img);
    }
  }

  if (focus_masks)
  {
    const uint64 field = uint64(Field::FIELD_FOCUS_MASKS);
    ofile.write((const char *)&field, sizeof(field));

    for (uint64 l = 0; l < num_levels; l++)
    {
      const cv::Mat & mask = (*focus_masks)[l];
      const SparseMask smask = MaskToSparseMask(mask);

      ofile.write((const char *)&smask.height, sizeof(smask.height));
      ofile.write((const char *)&smask.width, sizeof(smask.width));

      const uint64 size = smask.indices.size();
      ofile.write((const char *)&size, sizeof(size));
      for (uint64 i = 0; i < size; i++)
      {
        const uint64 y = smask.indices[i].y;
        ofile.write((const char *)&y, sizeof(y));
      }
      for (uint64 i = 0; i < size; i++)
      {
        const uint64 x = smask.indices[i].x;
        ofile.write((const char *)&x, sizeof(x));
      }
    }
  }

  if (image_pyramid)
  {
    const uint64 field = uint64(Field::FIELD_IMAGE_PYRAMID);
    ofile.write((const char *)&field, sizeof(field));

    for (uint64 l = 0; l < num_levels; l++)
    {
      const cv::Mat img = image_pyramid->imgs[l];
      const cv::Mat img_sq = image_pyramid->sq_imgs[l];
      std::vector<cv::Mat> mats;
      mats.push_back(img);
      mats.push_back(img_sq);
      cv::Mat merged;
      cv::merge(mats, merged);

      const SparseImage<cv::Vec2f> sp_img = ImageToSparseImageAutoMask<cv::Vec2f>(merged);

      WriteSparseImage<cv::Vec2f>(ofile, sp_img);
    }
  }

  if (weighted_image_pyramid)
  {
    const uint64 field = uint64(Field::FIELD_WEIGHTED_IMAGE_PYRAMID);
    ofile.write((const char *)&field, sizeof(field));

    for (uint64 l = 0; l < num_levels; l++)
    {
      const cv::Mat img = weighted_image_pyramid->imgs[l];
      const cv::Mat img_sq = weighted_image_pyramid->sq_imgs[l];
      const cv::Mat weights = weighted_image_pyramid->interest_weights[l];
      std::vector<cv::Mat> mats;
      mats.push_back(img);
      mats.push_back(img_sq);
      mats.push_back(weights);
      cv::Mat merged;
      cv::merge(mats, merged);

      const SparseImage<cv::Vec3f> sp_img = ImageToSparseImageAutoMask<cv::Vec3f>(merged);

      WriteSparseImage<cv::Vec3f>(ofile, sp_img);
    }
  }
}

template <typename T>
void SerializeOctree3D(std::ostream & ofile, const OctreeLevels & octree_levels, const std::vector<cv::Mat> * const focus_masks,
                       const ImagePyramid * const image_pyramid, const ImagePyramid * const weighted_image_pyramid)
{
  typedef SerializeOctreeField Field;

  const std::string magic = "OCTREE3D";

  ofile << magic;
  const uint64 version = 1;
  ofile.write((const char *)&version, sizeof(version));

  const uint64 num_levels = octree_levels.imgs.size();
  ofile.write((const char *)&num_levels, sizeof(num_levels));

  const uint64 type_size = sizeof(T) / sizeof(float);
  ofile.write((const char *)&type_size, sizeof(type_size));

  uint64 num_fields = 1;
  if (focus_masks)
    num_fields += 1;
  if (image_pyramid)
    num_fields += 1;
  if (weighted_image_pyramid)
    num_fields += 1;
  ofile.write((const char *)&num_fields, sizeof(num_fields));

  {
    const uint64 octree_levels_field = uint64(Field::FIELD_OCTREE_LEVELS);
    ofile.write((const char *)&octree_levels_field, sizeof(octree_levels_field));

    for (uint64 l = 0; l < num_levels; l++)
    {
      const cv::Mat img = octree_levels.imgs[l];
      const cv::Mat mask = octree_levels.img_masks[l];
      SparseImage3D<T> sparse_img = ImageToSparseImage3D<T>(img, mask);

      WriteSparseImage3D<T>(ofile, sparse_img);
    }
  }

  if (focus_masks)
  {
    const uint64 field = uint64(Field::FIELD_FOCUS_MASKS);
    ofile.write((const char *)&field, sizeof(field));

    for (uint64 l = 0; l < num_levels; l++)
    {
      const cv::Mat & mask = (*focus_masks)[l];
      const SparseMask3D smask = MaskToSparseMask3D(mask);

      ofile.write((const char *)&smask.depth, sizeof(smask.depth));
      ofile.write((const char *)&smask.height, sizeof(smask.height));
      ofile.write((const char *)&smask.width, sizeof(smask.width));

      const uint64 size = smask.indices.size();
      ofile.write((const char *)&size, sizeof(size));
      for (uint64 i = 0; i < size; i++)
      {
        const uint64 z = smask.indices[i].z;
        ofile.write((const char *)&z, sizeof(z));
      }
      for (uint64 i = 0; i < size; i++)
      {
        const uint64 y = smask.indices[i].y;
        ofile.write((const char *)&y, sizeof(y));
      }
      for (uint64 i = 0; i < size; i++)
      {
        const uint64 x = smask.indices[i].x;
        ofile.write((const char *)&x, sizeof(x));
      }
    }
  }

  if (image_pyramid)
  {
    const uint64 field = uint64(Field::FIELD_IMAGE_PYRAMID);
    ofile.write((const char *)&field, sizeof(field));

    for (uint64 l = 0; l < num_levels; l++)
    {
      const cv::Mat img = image_pyramid->imgs[l];
      const cv::Mat img_sq = image_pyramid->sq_imgs[l];
      std::vector<cv::Mat> mats;
      mats.push_back(img);
      mats.push_back(img_sq);
      cv::Mat merged;
      cv::merge(mats, merged);

      const SparseImage3D<cv::Vec2f> sp_img = ImageToSparseImage3DAutoMask<cv::Vec2f>(merged);

      WriteSparseImage3D<cv::Vec2f>(ofile, sp_img);
    }
  }

  if (weighted_image_pyramid)
  {
    const uint64 field = uint64(Field::FIELD_WEIGHTED_IMAGE_PYRAMID);
    ofile.write((const char *)&field, sizeof(field));

    for (uint64 l = 0; l < num_levels; l++)
    {
      const cv::Mat img = weighted_image_pyramid->imgs[l];
      const cv::Mat img_sq = weighted_image_pyramid->sq_imgs[l];
      const cv::Mat weights = weighted_image_pyramid->interest_weights[l];
      std::vector<cv::Mat> mats;
      mats.push_back(img);
      mats.push_back(img_sq);
      mats.push_back(weights);
      cv::Mat merged;
      cv::merge(mats, merged);

      const SparseImage3D<cv::Vec3f> sp_img = ImageToSparseImage3DAutoMask<cv::Vec3f>(merged);

      WriteSparseImage3D<cv::Vec3f>(ofile, sp_img);
    }
  }
}

bool DeserializeOctreeHeader(std::istream & ifile, uint64 & version, uint64 & num_levels, uint64 & num_channels,
                             uint64 & num_fields, bool &is_3d);

template <typename T>
bool DeserializeOctree(std::istream & ifile, std::vector<SparseImage<T> > & octree_levels,
                       std::vector<SparseMask> & focus_masks, ImagePyramid * image_pyramid, ImagePyramid * weighted_image_pyramid)
{
  typedef SerializeOctreeField Field;

  if (!ifile)
  {
    ROS_ERROR("DeserializeOctree: cannot open file.");
    return false;
  }

  focus_masks.clear();
  octree_levels.clear();

  uint64 version;
  uint64 num_levels;
  uint64 num_channels;
  uint64 num_fields;
  bool is_3d;
  if (!DeserializeOctreeHeader(ifile, version, num_levels, num_channels, num_fields, is_3d))
    return false;

  if (is_3d)
  {
    ROS_ERROR("DeserializeOctree: expected 2d octree, 3d octree header found.");
    return false;
  }

  const uint64 type_size = sizeof(T) / sizeof(float);
  if (type_size != num_channels)
  {
    ROS_ERROR("DeserializeOctree: expected %d channels from template, got %d", int(type_size), int(num_channels));
    return false;
  }

  for (uint64 f = 0; f < num_fields; f++)
  {
    uint64 field;
    ifile.read((char *)&field, sizeof(field));

    if (field == uint64(Field::FIELD_OCTREE_LEVELS))
    {
      octree_levels.clear();
      for (uint64 l = 0; l < num_levels; l++)
      {
        SparseImage<T> sparse_img;
        ReadSparseImage<T>(ifile, sparse_img);

        octree_levels.push_back(sparse_img);
      }
    }
    else if (field == uint64(Field::FIELD_FOCUS_MASKS))
    {
      focus_masks.clear();
      for (uint64 l = 0; l < num_levels; l++)
      {
        uint64 height, width, size;
        ifile.read((char *)&height, sizeof(height));
        ifile.read((char *)&width, sizeof(width));
        ifile.read((char *)&size, sizeof(size));

        SparseMask smask;
        smask.height = height;
        smask.width = width;
        smask.indices.resize(size);

        for (uint64 i = 0; i < size; i++)
        {
          uint64 y;
          ifile.read((char *)&y, sizeof(y));
          smask.indices[i].y = y;
        }
        for (uint64 i = 0; i < size; i++)
        {
          uint64 x;
          ifile.read((char *)&x, sizeof(x));
          smask.indices[i].x = x;
        }

        focus_masks.push_back(smask);
      }
    }
    else if (field == uint64(Field::FIELD_IMAGE_PYRAMID))
    {
      if (image_pyramid) // discard image pyramid if not required
      {
        image_pyramid->imgs.resize(num_levels);
        image_pyramid->sq_imgs.resize(num_levels);
      }

      for (uint64 l = 0; l < num_levels; l++)
      {
        SparseImage<cv::Vec2f> sparse_img;
        ReadSparseImage(ifile, sparse_img);

        cv::Mat useless_mask;
        cv::Mat merged = SparseImageToImage(sparse_img, useless_mask);

        std::vector<cv::Mat> split;
        cv::split(merged, split);
        cv::Mat img = split[0];
        cv::Mat sq_img = split[1];

        if (image_pyramid)
        {
          image_pyramid->imgs[l] = img;
          image_pyramid->sq_imgs[l] = sq_img;
        }
      }
    }
    else if (field == uint64(Field::FIELD_WEIGHTED_IMAGE_PYRAMID))
    {
      if (weighted_image_pyramid) // discard image pyramid if not required
      {
        weighted_image_pyramid->imgs.resize(num_levels);
        weighted_image_pyramid->sq_imgs.resize(num_levels);
        weighted_image_pyramid->interest_weights.resize(num_levels);
      }

      for (uint64 l = 0; l < num_levels; l++)
      {
        SparseImage<cv::Vec3f> sparse_img;
        ReadSparseImage(ifile, sparse_img);

        cv::Mat useless_mask;
        cv::Mat merged = SparseImageToImage(sparse_img, useless_mask);

        std::vector<cv::Mat> split;
        cv::split(merged, split);
        cv::Mat img = split[0];
        cv::Mat sq_img = split[1];
        cv::Mat weights = split[2];

        if (weighted_image_pyramid)
        {
          weighted_image_pyramid->imgs[l] = img;
          weighted_image_pyramid->sq_imgs[l] = sq_img;
          weighted_image_pyramid->interest_weights[l] = weights;
        }
      }
    }
    else
    {
      ROS_ERROR("DeserializeOctree: unknown field: %d.", int(field));
      return false;
    }
  }

  return true;
}

template <typename T>
bool DeserializeOctree3D(std::istream & ifile, std::vector<SparseImage3D<T> > & octree_levels,
                         std::vector<SparseMask3D> & focus_masks, ImagePyramid * image_pyramid, ImagePyramid * weighted_image_pyramid)
{
  typedef SerializeOctreeField Field;

  if (!ifile)
  {
    ROS_ERROR("DeserializeOctree: cannot open file.");
    return false;
  }

  focus_masks.clear();
  octree_levels.clear();

  uint64 version;
  uint64 num_levels;
  uint64 num_channels;
  uint64 num_fields;
  bool is_3d;
  if (!DeserializeOctreeHeader(ifile, version, num_levels, num_channels, num_fields, is_3d))
    return false;

  if (!is_3d)
  {
    ROS_ERROR("DeserializeOctree: expected 3d octree, 2d octree header found.");
    return false;
  }

  const uint64 type_size = sizeof(T) / sizeof(float);
  if (type_size != num_channels)
  {
    ROS_ERROR("DeserializeOctree: expected %d channels from template, got %d", int(type_size), int(num_channels));
    return false;
  }

  for (uint64 f = 0; f < num_fields; f++)
  {
    uint64 field;
    ifile.read((char *)&field, sizeof(field));

    if (field == uint64(Field::FIELD_OCTREE_LEVELS))
    {
      octree_levels.clear();
      for (uint64 l = 0; l < num_levels; l++)
      {
        SparseImage3D<T> sparse_img;
        ReadSparseImage3D<T>(ifile, sparse_img);

        octree_levels.push_back(sparse_img);
      }
    }
    else if (field == uint64(Field::FIELD_FOCUS_MASKS))
    {
      focus_masks.clear();
      for (uint64 l = 0; l < num_levels; l++)
      {
        uint64 depth, height, width, size;
        ifile.read((char *)&depth, sizeof(depth));
        ifile.read((char *)&height, sizeof(height));
        ifile.read((char *)&width, sizeof(width));
        ifile.read((char *)&size, sizeof(size));

        SparseMask3D smask;
        smask.depth = depth;
        smask.height = height;
        smask.width = width;
        smask.indices.resize(size);

        for (uint64 i = 0; i < size; i++)
        {
          uint64 z;
          ifile.read((char *)&z, sizeof(z));
          smask.indices[i].z = z;
        }
        for (uint64 i = 0; i < size; i++)
        {
          uint64 y;
          ifile.read((char *)&y, sizeof(y));
          smask.indices[i].y = y;
        }
        for (uint64 i = 0; i < size; i++)
        {
          uint64 x;
          ifile.read((char *)&x, sizeof(x));
          smask.indices[i].x = x;
        }

        focus_masks.push_back(smask);
      }
    }
    else if (field == uint64(Field::FIELD_IMAGE_PYRAMID))
    {
      if (image_pyramid) // discard image pyramid if not required
      {
        image_pyramid->imgs.resize(num_levels);
        image_pyramid->sq_imgs.resize(num_levels);
      }

      for (uint64 l = 0; l < num_levels; l++)
      {
        SparseImage3D<cv::Vec2f> sparse_img;
        ReadSparseImage3D(ifile, sparse_img);

        cv::Mat useless_mask;
        cv::Mat merged = SparseImageToImage3D(sparse_img, useless_mask);

        std::vector<cv::Mat> split;
        cv::split(merged, split);
        cv::Mat img = split[0];
        cv::Mat sq_img = split[1];

        if (image_pyramid)
        {
          image_pyramid->imgs[l] = img;
          image_pyramid->sq_imgs[l] = sq_img;
        }
      }
    }
    else if (field == uint64(Field::FIELD_WEIGHTED_IMAGE_PYRAMID))
    {
      if (weighted_image_pyramid) // discard image pyramid if not required
      {
        weighted_image_pyramid->imgs.resize(num_levels);
        weighted_image_pyramid->sq_imgs.resize(num_levels);
        weighted_image_pyramid->interest_weights.resize(num_levels);
      }

      for (uint64 l = 0; l < num_levels; l++)
      {
        SparseImage3D<cv::Vec3f> sparse_img;
        ReadSparseImage3D(ifile, sparse_img);

        cv::Mat useless_mask;
        cv::Mat merged = SparseImageToImage3D(sparse_img, useless_mask);

        std::vector<cv::Mat> split;
        cv::split(merged, split);
        cv::Mat img = split[0];
        cv::Mat sq_img = split[1];
        cv::Mat weights = split[2];

        if (weighted_image_pyramid)
        {
          weighted_image_pyramid->imgs[l] = img;
          weighted_image_pyramid->sq_imgs[l] = sq_img;
          weighted_image_pyramid->interest_weights[l] = weights;
        }
      }
    }
    else
    {
      ROS_ERROR("DeserializeOctree: unknown field: %d.", int(field));
      return false;
    }
  }

  return true;
}

}

#endif // OCTREE_LOAD_SAVE_H
