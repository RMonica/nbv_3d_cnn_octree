#ifndef IMAGE_TO_OCTREE_H
#define IMAGE_TO_OCTREE_H

#include "octree_load_save.h"

#include <string>
#include <stdint.h>
#include <fstream>
#include <vector>
#include <array>

namespace image_to_octree
{
template <int N>
  using IntArray = std::array<int, N>;
typedef uint8_t uint8;
typedef uint64_t uint64;

using OctreeLevels = OctreeLoadSave::OctreeLevels;
using ImagePyramid = OctreeLoadSave::ImagePyramid;
using SparseMask = OctreeLoadSave::SparseMask;
template <typename T>
  using SparseImage = OctreeLoadSave::SparseImage<T>;
template <typename T>
  using SparseImage3D = OctreeLoadSave::SparseImage3D<T>;

template <int N>
bool operator<(const IntArray<N> & a, const IntArray<N> & b)
{
  for (uint64 i = 0; i < N; i++)
  {
    if (a[i] < b[i]) return true;
    if (a[i] > b[i]) return false;
  }
  return false;
}

template <int N>
IntArray<N> ArrToIntArray(const int v[N])
{
  IntArray<N> result;
  for (uint64 i = 0; i < N; i++)
    result[i] = v[i];
  return result;
}

template <int N>
IntArray<N> DivIntArray(IntArray<N> v, int n)
{
  IntArray<N> result;
  for (uint64 i = 0; i < N; i++)
    result[i] = v[i] / n;
  return result;
}

template <int N>
IntArray<N> MulIntArray(IntArray<N> v, int n)
{
  IntArray<N> result;
  for (uint64 i = 0; i < N; i++)
    result[i] = v[i] * n;
  return result;
}

template <int N>
IntArray<N> AddIntArray(IntArray<N> v1, IntArray<N> v2)
{
  IntArray<N> result;
  for (uint64 i = 0; i < N; i++)
    result[i] = v1[i] + v2[i];
  return result;
}

template <int N>
IntArray<N> RemainderIntArray(IntArray<N> v1, int n)
{
  IntArray<N> result;
  for (uint64 i = 0; i < N; i++)
    result[i] = v1[i] % n;
  return result;
}

template <int N>
IntArray<N> IntArrayFilledWith(int v)
{
  IntArray<N> result;
  for (uint64 i = 0; i < N; i++)
    result[i] = v;
  return result;
}

// return false to break
template <int DIMS>
void ForeachSize(const IntArray<DIMS> & sizes, const int inc, std::function<bool(const IntArray<DIMS> &)> op)
{
  IntArray<DIMS> i;
  i.fill(0);
  while (true)
  {
    if (!op(i))
      return; // break

    i[sizes.size() - 1] += inc;
    int idx = sizes.size() - 1;
    while (i[idx] >= sizes[idx])
    {
      if (idx == 0)
        return; // ended
      i[idx] = 0;
      idx--;
      i[idx] += inc;
    }
  }
}

template <typename Scalar>
Scalar & At(cv::Mat & img, const int * i)
{
  uint64 dims = img.dims;
  if (dims == 2)
    return img.at<Scalar>(i[0], i[1]);
  if (dims == 3)
    return img.at<Scalar>(i[0], i[1], i[2]);
  ROS_FATAL("image_to_octree: At: unknown dims: %d", int(dims));
  std::exit(1);
}

template <typename Scalar>
const Scalar & At(const cv::Mat & img, const int * i)
{
  uint64 dims = img.dims;
  if (dims == 2)
    return img.at<Scalar>(i[0], i[1]);
  if (dims == 3)
    return img.at<Scalar>(i[0], i[1], i[2]);
  ROS_FATAL("image_to_octree: At: unknown dims: %d", int(dims));
  std::exit(1);
}

template <typename T, int DIMS>
cv::Mat UpsampleImage(const cv::Mat & img)
{
  IntArray<DIMS> sizes;
  for (uint64 i = 0; i < DIMS; i++)
    sizes[i] = img.size[i];
  IntArray<DIMS> double_sizes;
  for (uint64 i = 0; i < DIMS; i++)
    double_sizes[i] = sizes[i] * 2;

  cv::Mat result = cv::Mat(DIMS, double_sizes.data(), img.type());

  ForeachSize<DIMS>(double_sizes, 1, [&](const IntArray<DIMS> & i) -> bool {
    IntArray<DIMS> i2;
    for (uint64 h = 0; h < DIMS; h++)
      i2[h] = i[h] / 2;
    At<T>(result, i.data()) = At<T>(img, i2.data());
    return true;
  });

  return result;
}

template <typename T>
cv::Mat UpsampleImage2D3D(const cv::Mat & img, const bool is_3d)
{
  if (is_3d)
    return UpsampleImage<T, 3>(img);
  else
    return UpsampleImage<T, 2>(img);
}

// 2d version
template <typename T>
cv::Mat SparseImagesToImage(const std::vector<SparseImage<T> > & sparse_images)
{
  const uint64 max_levels = sparse_images.size();
  cv::Mat reconst_image;
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
      reconst_image = UpsampleImage<T, 2>(reconst_image);
      cv::Mat new_image = OctreeLoadSave::SparseImageToImage<T>(simg, mask);
      reconst_image += new_image;
    }
  }

  return reconst_image;
}

// 3d version
template <typename T>
cv::Mat SparseImagesToImage(const std::vector<SparseImage3D<T> > & sparse_images)
{
  const uint64 max_levels = sparse_images.size();
  cv::Mat reconst_image;
  for (uint64 l = 0; l < max_levels; l++)
  {
    const OctreeLoadSave::SparseImage3D<T> & simg = sparse_images[l];
    cv::Mat mask;

    if (l == 0)
    {
      reconst_image = OctreeLoadSave::SparseImageToImage3D<T>(simg, mask);
    }
    else
    {
      reconst_image = UpsampleImage<T, 3>(reconst_image);
      cv::Mat new_image = OctreeLoadSave::SparseImageToImage3D<T>(simg, mask);
      reconst_image += new_image;
    }
  }

  return reconst_image;
}

template <typename T, int DIMS>
cv::Mat CropImage(const cv::Mat & img, const IntArray<DIMS> & min_bounds, const IntArray<DIMS> & sizes)
{
  cv::Mat result(DIMS, sizes.data(), img.type());

  ForeachSize<DIMS>(sizes, 1, [&](const IntArray<DIMS> & i) -> bool {
    IntArray<DIMS> i2;
    for (uint64 h = 0; h < DIMS; h++)
      i2[h] = i[h] + min_bounds[h];
    At<T>(result, i.data()) = At<T>(img, i2.data());
    return true;
  });

  return result;
}

template <typename T, int DIMS>
cv::Mat DownsampleImage(const cv::Mat & img)
{
  IntArray<DIMS> sizes;
  for (uint64 i = 0; i < DIMS; i++)
    sizes[i] = img.size[i];
  IntArray<DIMS> half_sizes;
  for (uint64 i = 0; i < DIMS; i++)
    half_sizes[i] = sizes[i] / 2;

  cv::Mat result = cv::Mat(DIMS, half_sizes.data(), img.type());

  ForeachSize<DIMS>(half_sizes, 1, [&](const IntArray<DIMS> & i) -> bool {
    IntArray<DIMS> i2;
    for (uint64 h = 0; h < DIMS; h++)
      i2[h] = i[h] * 2;
    At<T>(result, i.data()) = At<T>(img, i2.data());
    return true;
  });

  return result;
}

template <typename T, int DIMS>
cv::Mat DownsampleImageAvg(const cv::Mat & img, const uint64 window = 2)
{
  const uint64 dims = DIMS;
  IntArray<DIMS> sizes;
  for (uint64 i = 0; i < dims; i++)
    sizes[i] = img.size[i];
  IntArray<DIMS> half_sizes;
  for (uint64 i = 0; i < dims; i++)
    half_sizes[i] = sizes[i] / window;

  cv::Mat result = cv::Mat(dims, half_sizes.data(), img.type());

  IntArray<DIMS> wsize; wsize.fill(window);

  ForeachSize<DIMS>(half_sizes, 1, [&](const IntArray<DIMS> & i) -> bool {
    T v = T(0);
    ForeachSize<DIMS>(wsize, 1, [&](const IntArray<DIMS> & d) -> bool{
      IntArray<DIMS> ni;
      for (uint64 nii = 0; nii < dims; nii++)
        ni[nii] = i[nii] * window + d[nii];
      v += At<T>(img, ni.data());
      return true;
    });

    At<T>(result, i.data()) = v / std::pow(window, dims);
    return true;
  });
  return result;
}

template <typename T, int DIMS>
cv::Mat DownsampleImageMin(const cv::Mat & img, const float default_value = 1.0f, const uint64 window = 2)
{
  const uint64 dims = DIMS;
  IntArray<DIMS> sizes;
  for (uint64 i = 0; i < dims; i++)
    sizes[i] = img.size[i];
  IntArray<DIMS> half_sizes;
  for (uint64 i = 0; i < dims; i++)
    half_sizes[i] = sizes[i] / window;

  cv::Mat result = cv::Mat(dims, half_sizes.data(), img.type());

  IntArray<DIMS> wsize; wsize.fill(window);

  ForeachSize<DIMS>(half_sizes, 1, [&](const IntArray<DIMS> & i) -> bool {
    T v = T(default_value);
    ForeachSize<DIMS>(wsize, 1, [&](const IntArray<DIMS> & d) -> bool{
      IntArray<DIMS> ni;
      for (uint64 nii = 0; nii < dims; nii++)
        ni[nii] = i[nii] * window + d[nii];
      const T nv = At<T>(img, ni.data());
      v = std::min(v, nv);
      return true;
    });

    At<T>(result, i.data()) = v;
    return true;
  });
  return result;
}

template <typename T, int DIMS>
cv::Mat DownsampleImageSum(const cv::Mat & img, const uint64 window = 2)
{
  const uint64 dims = DIMS;
  IntArray<DIMS> sizes;
  for (uint64 i = 0; i < dims; i++)
    sizes[i] = img.size[i];
  IntArray<DIMS> half_sizes;
  for (uint64 i = 0; i < dims; i++)
    half_sizes[i] = sizes[i] / window;

  cv::Mat result = cv::Mat(dims, half_sizes.data(), img.type());

  IntArray<DIMS> wsize; wsize.fill(window);

  ForeachSize<DIMS>(half_sizes, 1, [&](const IntArray<DIMS> & i) -> bool {
    T v = T(0);
    ForeachSize<DIMS>(wsize, 1, [&](const IntArray<DIMS> & d) -> bool{
      IntArray<DIMS> ni;
      for (uint64 nii = 0; nii < dims; nii++)
        ni[nii] = i[nii] * window + d[nii];
      v += At<T>(img, ni.data());
      return true;
    });

    At<T>(result, i.data()) = v;
    return true;
  });
  return result;
}

template <int DIMS>
OctreeLevels MaskToOctreeLevelsMinD(const cv::Mat & initial_mask, const uint64 max_layers)
{
  OctreeLevels result;

  cv::Mat mask_available = initial_mask.clone();

  IntArray<DIMS> sizes;
  for (uint64 i = 0; i < DIMS; i++)
    sizes[i] = initial_mask.size[i];

  uint64 layer_count = 1;

  while ([&]() -> bool { // if any dimension is == 1, cannot reduce further, so return
           for (int v : sizes) if (v <= 1) return false;
           return true;
         }())
  {
    cv::Mat img_available;
    mask_available.convertTo(img_available, CV_32FC1);
    cv::min(img_available, 1.0f, img_available);

    result.imgs.insert(result.imgs.begin(), img_available);
    result.img_masks.insert(result.img_masks.begin(), mask_available);

    mask_available = DownsampleImageMin<uint8, DIMS>(mask_available, 255);

    for (uint64 nidx = 0; nidx < DIMS; nidx++)
      sizes[nidx] = sizes[nidx] / 2;
    layer_count += 1;
    if (layer_count > max_layers)
      break;
  }

  return result;
}

template <typename T, int DIMS>
OctreeLevels ImageToOctreeLevelsD(const cv::Mat & image, const cv::Mat & initial_mask, const uint64 max_layers,
                                  const bool mask_only = false)
{
  OctreeLevels result;

  cv::Mat img = image.clone();
  cv::Mat mask_available = initial_mask.clone();
  uint64 dims = DIMS;
  IntArray<DIMS> sizes;
  for (uint64 i = 0; i < dims; i++)
    sizes[i] = img.size[i];
  uint64 layer_count = 1;

  IntArray<DIMS> wsize; wsize.fill(2);

  while ([&]() -> bool { // if any dimension is == 1, cannot reduce further, so return
           for (int v : sizes) if (v <= 1) return false;
           return true;
         }())
  {
    cv::Mat input = cv::Mat::zeros(dims, sizes.data(), image.type());
    cv::Mat input_mask = cv::Mat::zeros(dims, sizes.data(), CV_8UC1);
    cv::Mat next_img = DownsampleImage<T, DIMS>(img);
    cv::Mat next_mask_available = DownsampleImage<uint8, DIMS>(mask_available);
    ForeachSize<DIMS>(sizes, 2, [&](const IntArray<DIMS> & i) -> bool
    {
      T first_v;
      float first_m = std::numeric_limits<float>::quiet_NaN();
      bool found_different = false;
      for (int v : sizes)
        if (v <= 2) found_different = true; // last layer -> output all
      if (layer_count >= max_layers)
        found_different = true; // last layer -> output all
      ForeachSize<DIMS>(wsize, 1, [&](const IntArray<DIMS> & d) -> bool
      {
        IntArray<DIMS> ni;
        for (uint64 nidx = 0; nidx < dims; nidx++)
          ni[nidx] = d[nidx] + i[nidx];
        const bool m = At<uint8>(mask_available, ni.data());
        const T v = At<T>(img, ni.data());
        if (std::isnan(first_m))
        {
          first_v = v;
          first_m = m;
        }
        if (first_m != m)
          found_different = true;
        if (!mask_only && first_v != v)
          found_different = true;
        return true;
      });
      if (found_different)
      {
        IntArray<DIMS> half_i;
        for (uint64 nidx = 0; nidx < dims; nidx++)
          half_i[nidx] = i[nidx] / 2;
        At<uint8>(next_mask_available, half_i.data()) = false;
        ForeachSize<DIMS>(wsize, 1, [&](const IntArray<DIMS> & d) -> bool
        {
          IntArray<DIMS> ni;
          for (uint64 nidx = 0; nidx < dims; nidx++)
            ni[nidx] = d[nidx] + i[nidx];
          if (At<uint8>(mask_available, ni.data()))
          {
            At<T>(input, ni.data()) = At<T>(img, ni.data());
            At<uint8>(input_mask, ni.data()) = true;
          }
          return true;
        });
      }
      return true;
    });

    result.imgs.insert(result.imgs.begin(), input);
    result.img_masks.insert(result.img_masks.begin(), input_mask);

    for (uint64 nidx = 0; nidx < dims; nidx++)
      sizes[nidx] = sizes[nidx] / 2;
    img = next_img;
    mask_available = next_mask_available;
    layer_count += 1;
    if (layer_count > max_layers)
      break;
  }

  return result;
}

template <typename T>
cv::Mat PadToSquareSize(const cv::Mat img, const uint64 new_size, const T default_value = (T() * 0))
{
  if (img.dims == 2)
  {
    cv::Mat result = cv::Mat::zeros(new_size, new_size, img.type());
    cv::Rect roi(0, 0, img.cols, img.rows);
    img.copyTo(result(roi));
    return result;
  }
  else
  {
    int sizes[] = {int(new_size), int(new_size), int(new_size)};
    cv::Mat result = cv::Mat::zeros(3, sizes, img.type());
    result = default_value;
    for (uint64 z = 0; z < img.size[0]; z++)
      for (uint64 y = 0; y < img.size[1]; y++)
        for (uint64 x = 0; x < img.size[2]; x++)
        {
          result.at<T>(z, y, x) = img.at<T>(z, y, x);
        }
    return result;
  }
}

template <typename T>
cv::Mat PadToGreaterPower2(const cv::Mat img, const T default_value = (T() * 0))
{
  uint64 greater_power2 = 1;
  while ([greater_power2, img]() -> bool {
           for (int i = 0; i < img.dims; i++)
             if (img.size[i] > greater_power2)
               return true;
           return false;
         }())
    greater_power2 *= 2;
  return PadToSquareSize<T>(img, greater_power2, default_value);
}

template <typename T, int DIMS>
ImagePyramid InterestingImageToImagePyramidD(const cv::Mat & image_in, const uint64 max_layers, const cv::Mat & interesting_output_mask_in)
{
  ImagePyramid result;

  cv::Mat image = image_in.clone();
  cv::Mat interesting_output_mask = interesting_output_mask_in.clone();
  image = image.mul(interesting_output_mask);

  cv::Mat sq_image;
  cv::pow(image, 2.0, sq_image);

  result.imgs.insert(result.imgs.begin(), image);
  result.sq_imgs.insert(result.sq_imgs.begin(), sq_image);
  result.interest_weights.insert(result.interest_weights.begin(), interesting_output_mask);

  for (uint64 i = 1; i < max_layers; i++)
  {
    image = DownsampleImageSum<T, DIMS>(image);
    sq_image = DownsampleImageSum<T, DIMS>(sq_image);
    interesting_output_mask = DownsampleImageSum<T, DIMS>(interesting_output_mask);

    cv::Mat nonzero_interesting_output_mask = interesting_output_mask.clone();
    IntArray<DIMS> sizes;
    for (uint64 i = 0; i < DIMS; i++)
      sizes[i] = nonzero_interesting_output_mask.size[i];
    ForeachSize<DIMS>(sizes, 1, [&](const IntArray<DIMS> & i) -> bool {
      if (At<T>(nonzero_interesting_output_mask, i.data()) <= 0.0f)
        At<T>(nonzero_interesting_output_mask, i.data()) = T(1);
      return true;
    });

    cv::Mat avg_image = image.mul(T(1)/nonzero_interesting_output_mask);
    cv::Mat avg_sq_image = sq_image.mul(T(1)/nonzero_interesting_output_mask);

    result.imgs.insert(result.imgs.begin(), avg_image); // smaller images go first
    result.sq_imgs.insert(result.sq_imgs.begin(), avg_sq_image);
    result.interest_weights.insert(result.interest_weights.begin(), interesting_output_mask);
  }

  return result;
}

}


#endif // IMAGE_TO_OCTREE_H
