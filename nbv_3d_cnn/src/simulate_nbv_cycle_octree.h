#ifndef SIMULATE_NBV_CYCLE_OCTREE_H
#define SIMULATE_NBV_CYCLE_OCTREE_H

#include <stdint.h>
#include <vector>
#include <sstream>
#include <map>
#include <unordered_map>
#include <functional>

#include <ros/ros.h>

#include <nbv_3d_cnn_octree_common/image_to_octree.h>

namespace simulate_nbv_cycle_octree
{

typedef uint8_t uint8;
typedef uint32_t uint32;
typedef int32_t int32;
typedef uint64_t uint64;
typedef int64_t int64;

typedef std::vector<uint32> Uint32Vector;

using OctreeLevels = image_to_octree::OctreeLevels;

template <int DIMS>
using IntArray = image_to_octree::IntArray<DIMS>;

using image_to_octree::ForeachSize;
using image_to_octree::ArrToIntArray;
using image_to_octree::DivIntArray;
using image_to_octree::MulIntArray;
using image_to_octree::UpsampleImage;
using image_to_octree::IntArrayFilledWith;

template <typename Scalar, int DIMS>
struct OctreeCell
{
  uint32 children[1 << DIMS];
  Scalar value;

  OctreeCell()
  {
    for (uint64 i = 0; i < 1 << DIMS; i++)
      children[i] = -1;
    value = Scalar() * std::numeric_limits<float>::quiet_NaN();
  }

  typedef std::vector<OctreeCell> Vector;
};

template <typename Scalar, int DIMS>
struct Octree
{
  const uint32 num_levels;
  const uint32 num_children = (1 << DIMS);
  const IntArray<DIMS> dense_size;
  typedef OctreeCell<Scalar, DIMS> Cell;

  typedef std::vector<Scalar> ScalarVector;
  typedef std::vector<uint32> Uint32Vector;

  std::vector<Uint32Vector> level_children;
  std::vector<ScalarVector> level_scalars;

  Uint32Vector top_level_image;

  explicit Octree(const int num_levels, const IntArray<DIMS> & dense_size):
    num_levels(num_levels), dense_size(dense_size)
  {
    level_children.resize(num_levels);
    level_scalars.resize(num_levels);

    const IntArray<DIMS> base_sizes = DivIntArray<DIMS>(dense_size, (1 << (num_levels - 1)));
    uint64 base_prod = 1;
    for (int i = 0; i < DIMS; i++)
      base_prod *= base_sizes[i];
    top_level_image.resize(base_prod, uint32(-1));
  }

  Octree & operator=(const Octree & other)
  {
    if (num_levels != other.num_levels)
    {
      ROS_FATAL("Octree: operator=: mismatched num levels: %d and %d", int(num_levels), int(other.num_levels));
      std::exit(1);
    }
    if (dense_size != other.dense_size)
    {
      ROS_FATAL("Octree: operator=: mismatched dense_size.");
      std::exit(1);
    }

    level_children = other.level_children;
    level_scalars = other.level_scalars;
    top_level_image = other.top_level_image;

    return *this;
  }

  uint64 TopLevelImageIndex(const IntArray<DIMS> & idx) const
  {
    const IntArray<DIMS> base_sizes = DivIntArray<DIMS>(dense_size, (1 << (num_levels - 1)));

    uint64 index = idx[0];
    for (uint64 i = 0; i < (DIMS - 1); i++)
      index = index * base_sizes[i] + idx[i + 1];
    return index;
  }

  uint32 & TopLevelImageAt(const IntArray<DIMS> & idx) {return top_level_image[TopLevelImageIndex(idx)]; }
  uint32 TopLevelImageAt(const IntArray<DIMS> & idx) const {return top_level_image[TopLevelImageIndex(idx)]; }

  uint64 GetTotalCells() const
  {
    uint64 result = 0;
    for (const ScalarVector & sv : level_scalars)
      result += sv.size();
    return result;
  }

  uint32 LLAddCell(int level, const Cell & cell)
  {
    uint32 idx = level_scalars[level].size();
    level_scalars[level].push_back(cell.value);
    for (uint32 i = 0; i < num_children; i++)
      level_children[level].push_back(cell.children[i]);
    return idx;
  }

  Cell LLGetCell(int level, const uint32 index) const
  {
    Cell result;
    result.value = level_scalars[level][index];
    for (uint32 i = 0; i < num_children; i++)
      result.children[i] = level_children[level][index * num_children + i];
    return result;
  }

  void LLSetCell(int level, const uint32 index, const Cell & new_cell)
  {
    level_scalars[level][index] = new_cell.value;
    for (uint32 i = 0; i < num_children; i++)
      level_children[level][index * num_children + i] = new_cell.children[i];
  }

  static uint32 LLChildCoordToIndex(const IntArray<DIMS> & idx)
  {
    uint32 result = 0;
    for (int i = 0; i < DIMS; i++)
      result = result * 2 + idx[i];
    return result;
  }

  static IntArray<DIMS> LLChildIndexToCoord(const uint32 & idx)
  {
    IntArray<DIMS> result;
    for (int i = 0; i < DIMS; i++)
      result[i] = !!(idx & (1 << (DIMS - i - 1)));
    return result;
  }
};

/*
struct OctreeLevels
{
  std::vector<cv::Mat> imgs;
  std::vector<cv::Mat> img_masks;
};
*/

template <typename Scalar, int DIMS>
const Scalar & At(const cv::Mat & img, const int * i)
{
  uint64 dims = img.dims;
  if (dims == 2)
    return img.at<Scalar>(i[0], i[1]);
  if (dims == 3)
    return img.at<Scalar>(i[0], i[1], i[2]);
  ROS_FATAL("OctreeLevelsToOctree: At: unknown dims: %d", int(dims));
  std::exit(1);
}

template <typename Scalar, int DIMS>
Scalar & At(cv::Mat & img, const int * i)
{
  uint64 dims = img.dims;
  if (dims == 2)
    return img.at<Scalar>(i[0], i[1]);
  if (dims == 3)
    return img.at<Scalar>(i[0], i[1], i[2]);
  ROS_FATAL("OctreeLevelsToOctree: At: unknown dims: %d", int(dims));
  std::exit(1);
}

template <typename Scalar, int DIMS> // returns index of created cell
uint32 OctreeLevelsToOctreeKernel(const OctreeLoadSave::OctreeLevels & octree_levels,
                                  uint64 level, IntArray<DIMS> coords,
                                  Octree<Scalar, DIMS> & octree_out
                                  )
{
  typedef Octree<Scalar, DIMS> MyOctree;
  typedef typename MyOctree::Cell Cell;

  const uint8 has_value = At<uint8, DIMS>(octree_levels.img_masks[level], coords.data());

  bool any_valid = false;
  Cell c;
  if (has_value)
  {
    for (uint64 i = 0; i < octree_out.num_children; i++)
      c.children[i] = -1;
    c.value = At<Scalar, DIMS>(octree_levels.imgs[level], coords.data());
    any_valid = true;
  }
  else
  {
    if ((level + 1) == octree_out.num_levels)
    {
      for (uint64 i = 0; i < octree_out.num_children; i++)
        c.children[i] = -1; // reached last level with no information
    }
    else
    {
      // recursion
      for (uint64 i = 0; i < octree_out.num_children; i++)
      {
        IntArray<DIMS> new_coords = MulIntArray<DIMS>(coords, 2);
        for (uint64 d = 0; d < DIMS; d++)
          new_coords[d] += int(!!(i & (1 << (DIMS - d - 1))));
        const uint32 child_idx = OctreeLevelsToOctreeKernel<Scalar, DIMS>(octree_levels, level + 1, new_coords, octree_out);
        c.children[i] = child_idx;
        if (child_idx != uint32(-1))
          any_valid = true;
      }
    }
    c.value = Scalar() * std::numeric_limits<float>::quiet_NaN();
  }

  if (!any_valid)
    return -1; // no valid children or value, prune cell

  const uint32 idx = octree_out.LLAddCell(level, c);
  return idx;
}

template <typename Scalar, int DIMS>
Octree<Scalar, DIMS> OctreeLevelsToOctree(const OctreeLoadSave::OctreeLevels & octree_levels)
{
  typedef Octree<Scalar, DIMS> MyOctree;
  typedef typename MyOctree::Cell Cell;

  const uint64 num_levels = octree_levels.imgs.size();

  IntArray<DIMS> sizes = ArrToIntArray<DIMS>(octree_levels.img_masks[0].size);
  IntArray<DIMS> tot_sizes = MulIntArray<DIMS>(sizes, (1 << (num_levels - 1)));
  MyOctree result(num_levels, tot_sizes);

  ForeachSize<DIMS>(sizes, 1, [&](const IntArray<DIMS> & i) -> bool {
    uint32 c_i = i[0];
    for (uint64 d = 0; d < (DIMS - 1); d++)
      c_i = c_i * sizes[d] + i[d + 1];

    result.top_level_image[c_i] = OctreeLevelsToOctreeKernel<Scalar, DIMS>(octree_levels, 0, i, result);
    return true;
  });

  return result;
}

template <typename Scalar, int DIMS>
void OctreeToOctreeLevelsKernel(const Octree<Scalar, DIMS> & octree,
                                const IntArray<DIMS> & coords,
                                const uint32 c_i,
                                const uint32 l,
                                OctreeLevels & octree_levels)
{
  typedef Octree<Scalar, DIMS> MyOctree;
  typedef typename MyOctree::Cell Cell;

  const uint32 num_levels = octree.num_levels;

  if (l >= num_levels)
  {
    ROS_ERROR("OctreeToOctreeLevelsKernel: malformed octree: reached level %d, but there are only %d levels.",
              int(l), int(num_levels));
    return;
  }

  const Cell c = octree.LLGetCell(l, c_i);

  bool leaf = true;
  for (uint64 i = 0; i < octree.num_children; i++)
  {
    if (c.children[i] != uint32(-1))
    {
      // recursive call
      const uint32 c_c_i = c.children[i];
      IntArray<DIMS> child_coords = MulIntArray<DIMS>(coords, 2);
      for (uint64 d = 0; d < DIMS; d++)
        child_coords[d] += !!(i & (1 << (DIMS - d - 1)));
      OctreeToOctreeLevelsKernel<Scalar, DIMS>(octree, child_coords, c_c_i, l + 1, octree_levels);

      leaf = false;
    }
  }

  if (leaf)
  {
    // if leaf, store value
    At<Scalar, DIMS>(octree_levels.imgs[l], coords.data()) = c.value;
    At<uint8, DIMS>(octree_levels.img_masks[l], coords.data()) = uint8(true);
  }
}

template <typename Scalar, int DIMS>
OctreeLevels OctreeToOctreeLevels(const Octree<Scalar, DIMS> & octree)
{
  OctreeLevels octree_levels;
  const uint32 num_levels = octree.num_levels;

  octree_levels.imgs.resize(num_levels);
  octree_levels.img_masks.resize(num_levels);

  int type;
  switch (sizeof(Scalar) / sizeof(float))
  {
    case 1: type = CV_32FC1; break;
    case 2: type = CV_32FC2; break;
    case 3: type = CV_32FC3; break;
    case 4: type = CV_32FC4; break;
    default:
      ROS_FATAL("OctreeToOctreeLevels: unsupported type with %d floats, max 4 floats.", int(sizeof(Scalar) / sizeof(float)));
      std::exit(1);
  }

  const IntArray<DIMS> dense_size = octree.dense_size;
  IntArray<DIMS> curr_size = dense_size;
  for (uint32 l = 0; l < num_levels; l++)
  {
    octree_levels.imgs[num_levels - l - 1] = cv::Mat(DIMS, curr_size.data(), type, Scalar() * 0.0f);
    octree_levels.img_masks[num_levels - l - 1] = cv::Mat(DIMS, curr_size.data(), type, uint8(0));

    curr_size = DivIntArray<DIMS>(curr_size, 2);
  }

  const IntArray<DIMS> base_sizes = DivIntArray<DIMS>(dense_size, (1 << (num_levels - 1)));
  const int width = base_sizes[DIMS - 1];
  const int height = base_sizes[DIMS - 2];
  const int depth = DIMS >= 3 ? base_sizes[DIMS - 3] : 1;
  for (int z = 0; z < depth; z++)
    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++)
      {
        int xyz[3] = {x, y, z};
        IntArray<DIMS> coords;
        for (uint64 i = 0; i < DIMS; i++)
          coords[DIMS - i - 1] = xyz[i];

        uint32 c_i = octree.TopLevelImageAt(coords);
        if (c_i != uint32(-1))
          OctreeToOctreeLevelsKernel<Scalar, DIMS>(octree, coords, c_i, 0, octree_levels);
      }

  return octree_levels;
}

template <typename Scalar, int DIMS, typename SparseImageType >
Octree<Scalar, DIMS> SparseOctreeLevelsToOctree(const std::vector<SparseImageType> & sparse_images)
{
  const uint64 num_levels = sparse_images.size();
  Octree<Scalar, DIMS> result(num_levels, sparse_images.back().GetSizeArray());

  struct MyHash
  {
    std::size_t operator()(const IntArray<DIMS> & k) const
    {
      std::size_t result = 0;
      for (int i = 0; i < DIMS; i++)
        result = result ^ (std::hash<int>()(k[i]) << (i*8));
      return result;
    }
  };

  typedef OctreeCell<Scalar, DIMS> Cell;
  typedef IntArray<DIMS> DimsIntArray;
  typedef std::unordered_map<DimsIntArray, uint32, MyHash> DimsIntArrayUint32Map;
  typedef std::pair<DimsIntArray, uint32> DimsIntArrayUint32Pair;

  // maps 3D position of the cell to its index in the next octree level
  DimsIntArrayUint32Map level_cache;

  // fill the levels
  for (uint64 level = sparse_images.size(); level > 0; level--)
  {
    const uint64 l = level - 1;
    const SparseImageType & sparse_image = sparse_images[l];

    DimsIntArrayUint32Map next_level_cache;
    for (typename DimsIntArrayUint32Map::const_iterator iter = level_cache.begin(); iter != level_cache.end(); iter++)
    {
      const DimsIntArray coords = iter->first;
      const DimsIntArray parent_coords = image_to_octree::DivIntArray<DIMS>(coords, 2);
      const DimsIntArray child_coords = image_to_octree::RemainderIntArray<DIMS>(coords, 2);
      uint32 parent_index;
      const typename DimsIntArrayUint32Map::const_iterator parent_coords_iter = next_level_cache.find(parent_coords);
      if (parent_coords_iter != next_level_cache.end())
        parent_index = parent_coords_iter->second;
      else
      {
        Cell new_cell;
        parent_index = result.LLAddCell(l, new_cell);
        next_level_cache.insert(DimsIntArrayUint32Pair(parent_coords, parent_index));
      }

      Cell cell = result.LLGetCell(l, parent_index);
      cell.children[result.LLChildCoordToIndex(child_coords)] = iter->second;
      result.LLSetCell(l, parent_index, cell);
    }

    for (uint64 i = 0; i < sparse_image.indices.size(); i++)
    {
      Cell new_cell;
      new_cell.value = sparse_image.values[i];
      const uint32 new_idx = result.LLAddCell(l, new_cell);
      next_level_cache.insert(DimsIntArrayUint32Pair(sparse_image.indices[i].ToIntArray(), new_idx));
    }

    level_cache.swap(next_level_cache);
  }

  // fill the top-level image
  for (typename DimsIntArrayUint32Map::const_iterator iter = level_cache.begin(); iter != level_cache.end(); iter++)
  {
    result.TopLevelImageAt(iter->first) = iter->second;
  }

  return result;
}

template <typename ScalarIn, typename ScalarOut, int DIMS>
// true: outputs value, false: outputs index, if false index may be -1 if pruned branch
bool OctreeUnaryOpKernel(const Octree<ScalarIn, DIMS> & octree, const std::function<ScalarOut(ScalarIn)> & op,
                         const uint32 l, const uint32 c_i, Octree<ScalarOut, DIMS> & octree_out,
                         ScalarOut & value_out, uint32 & index_out)
{
  typedef Octree<ScalarIn, DIMS> MyOctreeIn;
  typedef typename MyOctreeIn::Cell CellIn;

  typedef Octree<ScalarOut, DIMS> MyOctreeOut;
  typedef typename MyOctreeOut::Cell CellOut;

  const uint32 num_levels = octree.num_levels;
  const uint32 num_children = octree.num_children;

  if (l >= num_levels)
  {
    ROS_ERROR("OctreeUnaryOpKernel: malformed octree: reached level %d, but there are only %d levels.",
              int(l), int(num_levels));
    value_out = ScalarOut() * 0.0f;
    return true;
  }

  const CellIn cell = octree.LLGetCell(l, c_i);
  CellOut cell_out;
  bool any_children = false;
  ScalarOut child_values[num_children];
  uint32 child_indices[num_children];
  for (uint32 i = 0; i < num_children; i++)
    child_indices[i] = uint32(-1);
  bool has_value[num_children];
  for (uint32 i = 0; i < num_children; i++)
    has_value[i] = false;

  for (uint32 i = 0; i < num_children; i++)
  {
    const uint32 c_c_i = cell.children[i];
    if (c_c_i == uint32(-1))
    {
      has_value[i] = false;
      child_indices[i] = uint32(-1);
      continue; // not set -> not set
    }

    any_children = true;

    ScalarOut child_value;
    uint32 child_index;
    const bool value_or_not_index = OctreeUnaryOpKernel(octree, op, l + 1, c_c_i, octree_out, child_value, child_index);
    has_value[i] = value_or_not_index;
    if (value_or_not_index)
      child_values[i] = child_value;
    else
      child_indices[i] = child_index;
  }

  if (!any_children)
  {
    value_out = op(cell.value);
    return true;
  }
  else //if (any_children)
  {
    bool all_values_equal = true;
    for (uint32 i = 0; i < num_children && all_values_equal; i++)
      if (!has_value[i] || child_values[i] != child_values[0])
        all_values_equal = false;

    if (all_values_equal)
    {
      value_out = child_values[0];
      return true;
    }
    else //if (!all_values_equal)
    {
      for (uint32 i = 0; i < num_children; i++)
      {
        if (!has_value[i])
          cell_out.children[i] = child_indices[i];
        else
        {
          CellOut child_cell;
          child_cell.value = child_values[i];
          cell_out.children[i] = octree_out.LLAddCell(l + 1, child_cell);
        }
      }

      index_out = octree_out.LLAddCell(l, cell_out);
      return false;
    }
  }
}

template <typename ScalarIn, typename ScalarOut, int DIMS>
Octree<ScalarOut, DIMS> OctreeUnaryOp(const Octree<ScalarIn, DIMS> & octree, const std::function<ScalarOut(ScalarIn)> & op)
{
  typedef Octree<ScalarIn, DIMS> MyOctreeIn;
  typedef typename MyOctreeIn::Cell CellIn;

  typedef Octree<ScalarOut, DIMS> MyOctreeOut;
  typedef typename MyOctreeOut::Cell CellOut;

  const uint64 num_levels = octree.num_levels;
  const IntArray<DIMS> dense_size = octree.dense_size;
  const IntArray<DIMS> base_size = DivIntArray<DIMS>(dense_size, (1 << (num_levels - 1)));

  MyOctreeOut octree_out(num_levels, dense_size);

  for (uint64 l = 0; l < num_levels; l++)
  {
    octree_out.level_children[l].reserve(octree.level_children[l].size());
    octree_out.level_scalars[l].reserve(octree.level_scalars[l].size());
  }

  ForeachSize<DIMS>(base_size, 1, [&](const IntArray<DIMS> & coords) -> bool {

    const uint32 c_i = octree.TopLevelImageAt(coords);
    if (c_i == uint32(-1))
    {
      octree_out.TopLevelImageAt(coords) = uint32(-1);
      return true; // not set -> not set
    }

    ScalarOut child_value;
    uint32 child_index;
    const bool value_or_not_index = OctreeUnaryOpKernel(octree, op, 0, c_i, octree_out, child_value, child_index);

    if (value_or_not_index)
    {
      CellOut cell_out;
      cell_out.value = child_value;
      octree_out.TopLevelImageAt(coords) = octree_out.LLAddCell(0, cell_out);
    }
    else
    {
      octree_out.TopLevelImageAt(coords) = child_index;
    }

    return true;
  });

  return octree_out;
}

template <typename Scalar, int DIMS>
// true: outputs value, false: outputs index, if false index may be -1 if pruned branch
bool ImageToOctreeKernel(const cv::Mat & image, const uint64 num_levels, const IntArray<DIMS> & image_size,
                         const uint32 l, const IntArray<DIMS> & coords, Octree<Scalar, DIMS> & octree_out,
                         Scalar & value_out, uint32 & index_out)
{
  typedef Octree<Scalar, DIMS> MyOctree;
  typedef typename MyOctree::Cell Cell;

  const uint32 num_children = octree_out.num_children;

  // check if out of base image
  for (uint64 i = 0; i < DIMS; i++)
    if (coords[i] * (1 << (num_levels - 1 - l)) >= image_size[i])
    {
      index_out = -1;
      return false; // out of image, set to not set
    }

  if (l + 1 == num_levels) // reached max resolution
  {
    value_out = At<Scalar, DIMS>(image, coords.data()); // just return image value
    return true;
  }

  Cell cell_out;
  Scalar child_values[num_children];
  uint32 child_indices[num_children];
  for (uint32 i = 0; i < num_children; i++)
    child_indices[i] = uint32(-1);
  bool has_value[num_children];
  for (uint32 i = 0; i < num_children; i++)
    has_value[i] = false;

  for (uint32 i = 0; i < num_children; i++)
  {
    IntArray<DIMS> child_coords = MulIntArray<DIMS>(coords, 2);
    for (uint64 d = 0; d < DIMS; d++)
      child_coords[d] += !!(i & (1 << (DIMS - d - 1)));

    Scalar child_value;
    uint32 child_index;
    const bool value_or_not_index = ImageToOctreeKernel<Scalar, DIMS>
        (image, num_levels, image_size, l + 1, child_coords, octree_out, child_value, child_index);
    has_value[i] = value_or_not_index;
    if (value_or_not_index)
      child_values[i] = child_value;
    else
      child_indices[i] = child_index;
  }

  bool all_values_equal = true;
  for (uint32 i = 0; i < num_children && all_values_equal; i++)
    if (!has_value[i] || child_values[i] != child_values[0])
      all_values_equal = false;

  if (all_values_equal)
  {
    value_out = child_values[0];
    return true;
  }
  else //if (!all_values_equal)
  {
    for (uint32 i = 0; i < num_children; i++)
    {
      if (!has_value[i])
        cell_out.children[i] = child_indices[i];
      else
      {
        Cell child_cell;
        child_cell.value = child_values[i];
        cell_out.children[i] = octree_out.LLAddCell(l + 1, child_cell);
      }
    }

    index_out = octree_out.LLAddCell(l, cell_out);
    return false;
  }
}

template <typename Scalar, int DIMS>
Octree<Scalar, DIMS> ImageToOctree(const cv::Mat & image, const uint64 num_levels)
{
  typedef Octree<Scalar, DIMS> MyOctree;
  typedef typename MyOctree::Cell Cell;

  uint64 greater_power2 = 1;
  while ([greater_power2, image]() -> bool {
           for (int i = 0; i < DIMS; i++)
             if (image.size[i] > greater_power2)
               return true;
           return false;
         }())
    greater_power2 *= 2;

  IntArray<DIMS> image_size;
  for (uint64 i = 0; i < DIMS; i++)
    image_size[i] = image.size[i];
  const IntArray<DIMS> dense_size = IntArrayFilledWith<DIMS>(greater_power2);
  const IntArray<DIMS> base_size = DivIntArray<DIMS>(dense_size, (1 << (num_levels - 1)));

  MyOctree octree_out(num_levels, dense_size);

  ForeachSize<DIMS>(base_size, 1, [&](const IntArray<DIMS> & coords) -> bool {
    // check if out of base image
    for (uint64 i = 0; i < DIMS; i++)
      if ((coords[i] * (1 << (num_levels - 1))) >= image_size[i])
      {
        octree_out.TopLevelImageAt(coords) = uint32(-1);
        return true; // out of image, set to not set
      }

    Scalar child_value;
    uint32 child_index;
    const bool value_or_not_index = ImageToOctreeKernel<Scalar, DIMS>
        (image, num_levels, image_size, 0, coords, octree_out, child_value, child_index);

    if (value_or_not_index)
    {
      Cell cell_out;
      cell_out.value = child_value;
      octree_out.TopLevelImageAt(coords) = octree_out.LLAddCell(0, cell_out);
    }
    else
    {
      octree_out.TopLevelImageAt(coords) = child_index;
    }

    return true;
  });

  return octree_out;
}

template <typename Scalar1In, typename Scalar2In, typename ScalarOut, int DIMS>
// true: outputs value, false: outputs index, if false index may be -1 if pruned branch
bool OctreeBinaryOpKernel(const Octree<Scalar1In, DIMS> & octree1, const Octree<Scalar1In, DIMS> & octree2,
                          const std::function<ScalarOut(Scalar1In, Scalar2In)> & op,
                          const uint32 l1, const uint32 c_i1, const bool is_frozen_1,
                          const uint32 l2, const uint32 c_i2, const bool is_frozen_2,
                          Octree<ScalarOut, DIMS> & octree_out,
                          ScalarOut & value_out, uint32 & index_out)
{
  typedef Octree<Scalar1In, DIMS> MyOctree1In;
  typedef typename MyOctree1In::Cell Cell1In;

  typedef Octree<Scalar2In, DIMS> MyOctree2In;
  typedef typename MyOctree2In::Cell Cell2In;

  typedef Octree<ScalarOut, DIMS> MyOctreeOut;
  typedef typename MyOctreeOut::Cell CellOut;

  const uint32 num_levels = octree1.num_levels;
  const uint32 num_children = octree1.num_children;

  if (l1 >= num_levels || l2 >= num_levels)
  {
    ROS_ERROR("OctreeBinaryOpKernel: malformed octree: reached levels (%d, %d), but there are only %d levels.",
              int(l1), int(l2), int(num_levels));
    value_out = ScalarOut() * 0.0f;
    return true;
  }

  const uint32 l = std::max<uint32>(l1, l2);

  const bool valid_cell1 = (c_i1 != uint32(-1));
  const bool valid_cell2 = (c_i2 != uint32(-1));

  Cell1In cell1, cell2;
  if (valid_cell1)
    cell1 = octree1.LLGetCell(l1, c_i1);
  if (valid_cell2)
    cell2 = octree2.LLGetCell(l2, c_i2);

  CellOut cell_out;
  bool any_children = false;
  ScalarOut child_values[num_children];
  uint32 child_indices[num_children];
  for (uint32 i = 0; i < num_children; i++)
    child_indices[i] = uint32(-1);
  bool child_has_value[num_children];
  for (uint32 i = 0; i < num_children; i++)
    child_has_value[i] = false;

  for (uint32 i = 0; i < num_children; i++)
  {
    const uint32 c_c_i1 = (valid_cell1 && !is_frozen_1) ? cell1.children[i] : uint32(-1);
    const uint32 c_c_i2 = (valid_cell2 && !is_frozen_2) ? cell2.children[i] : uint32(-1);

    if (c_c_i1 == uint32(-1) && c_c_i2 == uint32(-1))
    {
      child_has_value[i] = false;
      child_indices[i] = uint32(-1);
      continue; // both not set -> not set
    }

    any_children = true;

    const bool is_frozen_child1 = is_frozen_1 || (c_c_i1 == uint32(-1));
    const bool is_frozen_child2 = is_frozen_2 || (c_c_i2 == uint32(-1));

    const uint32 child_l1 = !is_frozen_child1 ? l1 + 1 : l1;
    const uint32 child_l2 = !is_frozen_child2 ? l2 + 1 : l2;

    const uint32 child_c_i1 = !is_frozen_child1 ? c_c_i1 : c_i1;
    const uint32 child_c_i2 = !is_frozen_child2 ? c_c_i2 : c_i2;


    ScalarOut child_value;
    uint32 child_index;
    const bool value_or_not_index = OctreeBinaryOpKernel(octree1, octree2, op,
                                                         child_l1, child_c_i1, is_frozen_child1,
                                                         child_l2, child_c_i2, is_frozen_child2,
                                                         octree_out, child_value, child_index);
    child_has_value[i] = value_or_not_index;
    if (value_or_not_index)
      child_values[i] = child_value;
    else
      child_indices[i] = child_index;
  }

  if (!any_children)
  {
    const Scalar1In nan1 = Scalar1In() * std::numeric_limits<float>::quiet_NaN();
    const Scalar2In nan2 = Scalar2In() * std::numeric_limits<float>::quiet_NaN();
    value_out = op(valid_cell1 ? cell1.value : nan1, valid_cell2 ? cell2.value : nan2);
    return true;
  }
  else //if (any_children)
  {
    bool all_values_equal = true;
    for (uint32 i = 0; i < num_children && all_values_equal; i++)
      if (!child_has_value[i] || child_values[i] != child_values[0])
        all_values_equal = false;

    if (all_values_equal)
    {
      value_out = child_values[0];
      return true;
    }
    else //if (!all_values_equal)
    {
      for (uint32 i = 0; i < num_children; i++)
      {
        if (!child_has_value[i])
          cell_out.children[i] = child_indices[i];
        else
        {
          CellOut child_cell;
          child_cell.value = child_values[i];
          cell_out.children[i] = octree_out.LLAddCell(l + 1, child_cell);
        }
      }

      index_out = octree_out.LLAddCell(l, cell_out);
      return false;
    }
  }
}

template <typename Scalar1In, typename Scalar2In, typename ScalarOut, int DIMS>
Octree<ScalarOut, DIMS> OctreeBinaryOp(const Octree<Scalar1In, DIMS> & octree1, const Octree<Scalar2In, DIMS> & octree2,
                                       const std::function<ScalarOut(Scalar1In, Scalar2In)> & op)
{
  typedef Octree<ScalarOut, DIMS> MyOctreeOut;
  typedef typename MyOctreeOut::Cell CellOut;

  const uint64 num_levels = octree1.num_levels;
  const IntArray<DIMS> dense_size = octree1.dense_size;
  const IntArray<DIMS> base_size = DivIntArray<DIMS>(dense_size, (1 << (num_levels - 1)));

  MyOctreeOut octree_out(num_levels, dense_size);

  for (uint64 l = 0; l < num_levels; l++)
  {
    octree_out.level_children[l].reserve(std::max(octree1.level_children[l].size(), octree2.level_children[l].size()));
    octree_out.level_scalars[l].reserve(std::max(octree1.level_scalars[l].size(), octree2.level_scalars[l].size()));
  }

  ForeachSize<DIMS>(base_size, 1, [&](const IntArray<DIMS> & coords) -> bool {

    const uint32 c_i1 = octree1.TopLevelImageAt(coords);
    const uint32 c_i2 = octree2.TopLevelImageAt(coords);
    if (c_i1 == uint32(-1) && c_i2 == uint32(-1))
    {
      octree_out.TopLevelImageAt(coords) = uint32(-1);
      return true; // both not set -> not set
    }

    ScalarOut child_value;
    uint32 child_index;
    const bool value_or_not_index = OctreeBinaryOpKernel(octree1, octree2, op,
                                                         0, c_i1, false,
                                                         0, c_i2, false,
                                                         octree_out, child_value, child_index);

    if (value_or_not_index)
    {
      CellOut cell_out;
      cell_out.value = child_value;
      octree_out.TopLevelImageAt(coords) = octree_out.LLAddCell(0, cell_out);
    }
    else
    {
      octree_out.TopLevelImageAt(coords) = child_index;
    }

    return true;
  });

  return octree_out;
}

template <typename Scalar, int DIMS>
Octree<Scalar, DIMS> OctreePrune(const Octree<Scalar, DIMS> & oct)
{
  return OctreeUnaryOp<Scalar, Scalar, DIMS>(oct, [](float a) -> float {return a; });
}

template <typename Scalar, int DIMS>
Scalar OctreeAt(const Octree<Scalar, DIMS> & octree, const IntArray<DIMS> coords)
{
  typedef Octree<Scalar, DIMS> MyOctree;
  typedef typename MyOctree::Cell Cell;

  const uint64 num_levels = octree.num_levels;

  const IntArray<DIMS> tot_sizes = octree.dense_size;
  const IntArray<DIMS> base_sizes = DivIntArray<DIMS>(tot_sizes, (1 << (num_levels - 1)));

  const IntArray<DIMS> small_i = DivIntArray<DIMS>(coords, (1 << (num_levels - 1)));

  uint32 c_i = small_i[0];
  for (uint64 i = 0; i < (DIMS - 1); i++)
    c_i = c_i * base_sizes[i] + small_i[i + 1];
  if (octree.top_level_image[c_i] == uint32(-1))
    return Scalar() * std::numeric_limits<Scalar>::quiet_NaN();
  c_i = octree.top_level_image[c_i];

  for (uint64 l = 0; l < num_levels; l++)
  {
    const Cell c = octree.LLGetCell(l, c_i);

    IntArray<DIMS> c_l_i = DivIntArray<DIMS>(coords, (1 << (num_levels - l - 2)));
    for (uint64 d = 0; d < DIMS; d++)
      c_l_i[d] = int(c_l_i[d] & 1);

    uint64 child_idx = 0;
    for (uint64 d = 0; d < DIMS; d++)
      child_idx += c_l_i[d] << (DIMS - d - 1);

    c_i = c.children[child_idx];

    if (c_i == uint32(-1))
      return c.value;
  }

  return Scalar() * std::numeric_limits<Scalar>::quiet_NaN();
}

template <typename Scalar, int DIMS>
Uint32Vector SerializeOctreeToUint32(const Octree<Scalar, DIMS> & octree)
{
  /*
  Serialized octree data structure:
  uint num_levels
  uint base_image_depth, base_image_height, base_image_width
  uint first_children_pointer
  uint first_value_pointer
  uint[] base_image
  uint[] children
  uint[] values
  */

  const uint32 HEADER_SIZE = 6;
  const uint32 num_children = (1 << DIMS);
  const uint32 value_size = (sizeof(Scalar) + sizeof(uint32) - 1) / sizeof(uint32);

  const uint32 num_levels = octree.num_levels;

  uint32 total_children = 0;
  for (uint64 l = 0; l < num_levels; l++)
  {
    total_children += octree.level_children[l].size() / num_children;
  }
  uint32 total_values = 0;
  for (uint64 l = 0; l < num_levels; l++)
  {
    total_values += octree.level_scalars[l].size();
  }
  uint32 base_image_size = 1;
  const IntArray<DIMS> base_sizes = DivIntArray<DIMS>(octree.dense_size, (1 << (num_levels - 1)));
  for (int s : base_sizes)
    base_image_size *= s;

  Uint32Vector result;
  result.resize(HEADER_SIZE + base_image_size + total_children * num_children + total_values * value_size);

  // header
  uint64 header_idx = 0;
  result[header_idx++] = octree.num_levels;
  for (int32 i = 0; i < 3 - DIMS; i++)
    result[header_idx++] = 1; // pad sizes with ones if < 3
  for (int32 i = 0; i < DIMS; i++)
    result[header_idx++] = base_sizes[i];

  const uint32 first_children_pointer = HEADER_SIZE + base_image_size;
  result[header_idx++] = first_children_pointer;
  const uint32 first_value_pointer = HEADER_SIZE + base_image_size + total_children * num_children;
  result[header_idx++] = first_value_pointer;

  if (header_idx != HEADER_SIZE)
    ROS_FATAL("SerializeOctreeToUint32: header size mismatch, expected %d, got %d", int(HEADER_SIZE), int(header_idx));

  for (uint64 i = 0; i < base_image_size; i++)
    result[HEADER_SIZE + i] = octree.top_level_image[i];

  uint64 cumulative_children_size = 0;
  for (uint32 l = 0; l < num_levels; l++)
  {
    const uint64 this_level_size = octree.level_children[l].size() / num_children;
    for (uint64 i = 0; i < this_level_size; i++)
      for (uint64 c = 0; c < num_children; c++)
      {
        uint32 child = octree.level_children[l][i * num_children + c];
        if (child != uint32(-1))
          child += cumulative_children_size + this_level_size;
        result[first_children_pointer + cumulative_children_size * num_children + i * num_children + c] = child;
      }

    cumulative_children_size += this_level_size;
  }

  uint64 cumulative_values_size = 0;
  for (uint32 l = 0; l < num_levels; l++)
  {
    const uint64 this_level_size = octree.level_scalars[l].size();
    for (uint64 i = 0; i < this_level_size; i++)
      for (uint64 c = 0; c < value_size; c++)
      {
        const uint32 * const v = reinterpret_cast<const uint32 *>(&(octree.level_scalars[l][i]));
        result[first_value_pointer + cumulative_values_size * value_size + i * value_size + c] = v[c];
      }

    cumulative_values_size += this_level_size;
  }

  return result;
}

template <typename Scalar, int DIMS>
Scalar SerializedOctreeAt(const Uint32Vector & octree, const int z, const int y, const int x)
{
  const uint32 HEADER_SIZE = 6;
  const uint32 num_children = (1 << DIMS);
  const uint32 value_size = (sizeof(Scalar) + sizeof(uint32) - 1) / sizeof(uint32);

  const int SERIALIZED_DIMS = 3; // all serialized octrees are 3D for simplicity

  const uint64 num_levels             = octree[0];
  const uint64 base_image_depth       = octree[1];
  const uint64 base_image_height      = octree[2];
  const uint64 base_image_width       = octree[3];
  const uint64 first_children_pointer = octree[4];
  const uint64 first_value_pointer    = octree[5];

  const uint32 * const base_image = &(octree[HEADER_SIZE]);
  const uint32 * const children = &(octree[first_children_pointer]);
  const uint32 * const values = &(octree[first_value_pointer]);

  const IntArray<SERIALIZED_DIMS> base_sizes = {int(base_image_depth), int(base_image_height), int(base_image_width)};
  const IntArray<SERIALIZED_DIMS> coords = {z, y, x};
  const IntArray<SERIALIZED_DIMS> small_coords = DivIntArray<SERIALIZED_DIMS>(coords, (1 << (num_levels - 1)));

  uint32 c_i = small_coords[0] * base_sizes[2] * base_sizes[1] + small_coords[1] * base_sizes[2] + small_coords[2];
  if (base_image[c_i] == uint32(-1))
    return Scalar() * std::numeric_limits<float>::quiet_NaN();
  c_i = base_image[c_i];

  for (uint64 l = 0; l < num_levels; l++)
  {
    IntArray<SERIALIZED_DIMS> c_l_i = DivIntArray<SERIALIZED_DIMS>(coords, (1 << (num_levels - l - 2)));
    for (uint64 d = 0; d < SERIALIZED_DIMS; d++)
      c_l_i[d] = int(c_l_i[d] & 1);

    uint64 child_idx = 0;
    for (uint64 d = 0; d < SERIALIZED_DIMS; d++)
      child_idx += c_l_i[d] << (SERIALIZED_DIMS - d - 1);

    const uint32 * const c_children = &(children[c_i * num_children]);
    const uint32 * const c_value = &(values[c_i * value_size]);

    c_i = c_children[child_idx];

    if (c_i == uint32(-1))
    {
      const Scalar result = *reinterpret_cast<const Scalar *>(c_value);
      return result;
    }
  }

  return Scalar() * std::numeric_limits<float>::quiet_NaN();
}

// this is to allow usage of 3 coords for 2d octrees
template <typename Scalar, int DIMS>
Scalar OctreeAt3D(const simulate_nbv_cycle_octree::Octree<Scalar, DIMS> & octree, const int z, const int y, const int x)
{
  ROS_FATAL("OctreeAt3D: OctreeAt called with unsupported DIMS == %d", DIMS);
  exit(1);
}

template <>
inline float OctreeAt3D<float, 2>(const simulate_nbv_cycle_octree::Octree<float, 2> & octree, const int z, const int y, const int x)
{
  if (z != 0)
  {
    ROS_FATAL("OctreeAt3D: OctreeAt called with DIMS == 2 && z != 0");
    exit(1);
  }

  IntArray<2> coords = {y, x};
  return OctreeAt<float, 2>(octree, coords);
}

template <>
inline float OctreeAt3D<float, 3>(const simulate_nbv_cycle_octree::Octree<float, 3> & octree, const int z, const int y, const int x)
{
  IntArray<3> coords = {z, y, x};
  return OctreeAt<float, 3>(octree, coords);
}

}

#endif // SIMULATE_NBV_CYCLE_OCTREE_H
