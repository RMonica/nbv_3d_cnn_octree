#ifndef SIMULATE_NBV_CYCLE_OCTREE_PREDICTION_H
#define SIMULATE_NBV_CYCLE_OCTREE_PREDICTION_H

#include <nbv_3d_cnn/voxelgrid.h>
#include "simulate_nbv_cycle_octree.h"

#include <ros/ros.h>

// STL
#include <string>
#include <stdint.h>
#include <vector>
#include <memory>

// OpenCV
#include <opencv2/core/core.hpp>

template <int DIMS>
class IOctreePrediction
{
  public:
  typedef simulate_nbv_cycle_octree::Octree<float, DIMS> Octree;
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  virtual ~IOctreePrediction() {}

  virtual const cv::Mat & GetImage() = 0;
  virtual const Octree & GetOctree(const uint64 max_layers) = 0;
  virtual const Voxelgrid & GetVoxelgrid() = 0;

  typedef std::shared_ptr<IOctreePrediction> Ptr;
};

template <int DIMS>
const std::shared_ptr<Voxelgrid> OctreePredictionToVoxelgrid(IOctreePrediction<DIMS> & prediction)
{
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  const cv::Mat image = prediction.GetImage();

  int sizes[DIMS];
  for (uint64 i = 0; i < DIMS; i++)
    sizes[i] = image.size[i];

  const uint64 width = sizes[DIMS - 1];
  const uint64 height = sizes[DIMS - 2];
  const uint64 depth = (DIMS >= 3) ? sizes[DIMS - 3] : 1;

  std::shared_ptr<Voxelgrid> result(new Voxelgrid(width, height, depth));

  if (DIMS == 2)
  {
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        result->at(x, y, 0) = image.at<float>(y, x);
  }
  else
  {
    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
          result->at(x, y, z) = image.at<float>(z, y, x);
  }

  return result;
}

template <int DIMS>
class ImagePrediction: public IOctreePrediction<DIMS>
{
  public:
  ImagePrediction(const cv::Mat image): m_image(image) {}

  typedef typename IOctreePrediction<DIMS>::Octree Octree;
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  virtual const cv::Mat & GetImage()
  {
    return m_image;
  }

  virtual const Voxelgrid & GetVoxelgrid()
  {
    if (m_cached_voxelgrid)
      return *m_cached_voxelgrid;

    m_cached_voxelgrid = OctreePredictionToVoxelgrid(*this);

    return *m_cached_voxelgrid;
  }

  virtual const Octree & GetOctree(const uint64 max_layers)
  {
    if (m_cached_octree)
      return *m_cached_octree;

    const simulate_nbv_cycle_octree::Octree<float, DIMS> oct = simulate_nbv_cycle_octree::ImageToOctree<float, DIMS>(m_image, max_layers);

    m_cached_octree.reset(new Octree(oct));
    return *m_cached_octree;
  }

  typedef std::shared_ptr<ImagePrediction> Ptr;

  private:
  cv::Mat m_image;

  std::shared_ptr<Voxelgrid> m_cached_voxelgrid;
  std::shared_ptr<Octree> m_cached_octree;
};

template <int DIMS>
class OctreePrediction: public IOctreePrediction<DIMS>
{
  public:
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  typedef typename IOctreePrediction<DIMS>::Octree Octree;
  typedef std::array<uint64, DIMS> SizeArray;
  typedef std::array<int, DIMS> IntArray;

  OctreePrediction(const Octree & octree, const IntArray & dense_size): m_octree(octree), m_dense_size(dense_size) {}

  virtual const cv::Mat & GetImage()
  {
    if (m_image_cache)
      return *m_image_cache;

    const IntArray env_size = m_dense_size;
    const simulate_nbv_cycle_octree::OctreeLevels octree_levels = simulate_nbv_cycle_octree::OctreeToOctreeLevels(m_octree);

    cv::Mat img = octree_levels.imgs[0].clone();
    for (int i = 1; i < octree_levels.imgs.size(); i++)
    {
      img = image_to_octree::UpsampleImage<float, DIMS>(img);
      img = img + octree_levels.imgs[i];
    }

    img = image_to_octree::CropImage<float, DIMS>(img, image_to_octree::IntArrayFilledWith<DIMS>(0), env_size);

    m_image_cache.reset(new cv::Mat());
    *m_image_cache = img;

    return *m_image_cache;
  }

  virtual const Voxelgrid & GetVoxelgrid()
  {
    if (m_cached_voxelgrid)
      return *m_cached_voxelgrid;

    m_cached_voxelgrid = OctreePredictionToVoxelgrid(*this);

    return *m_cached_voxelgrid;
  }

  virtual const Octree & GetOctree(const uint64 max_layers)
  {
    return m_octree;
  }

  typedef std::shared_ptr<OctreePrediction> Ptr;

  private:

  Octree m_octree;
  IntArray m_dense_size;

  std::shared_ptr<cv::Mat> m_image_cache;
  std::shared_ptr<Voxelgrid> m_cached_voxelgrid;
};

#endif // SIMULATE_NBV_CYCLE_OCTREE_PREDICTION_H
