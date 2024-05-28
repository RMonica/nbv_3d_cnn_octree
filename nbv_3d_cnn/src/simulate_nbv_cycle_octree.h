#ifndef SIMULATE_NBV_CYCLE_OCTREE_H
#define SIMULATE_NBV_CYCLE_OCTREE_H

#include <stdint.h>
#include <vector>
#include <sstream>
#include <map>
#include <unordered_map>

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

template <int DIMS>
using IntArray = image_to_octree::IntArray<DIMS>;

using image_to_octree::ForeachSize;
using image_to_octree::ArrToIntArray;
using image_to_octree::DivIntArray;
using image_to_octree::MulIntArray;
using image_to_octree::UpsampleImage;

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
  IntArray<DIMS> dense_size;
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

  uint64 TopLevelImageIndex(const IntArray<DIMS> & idx)
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
