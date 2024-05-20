#include <nbv_3d_cnn_octree_common/octree_load_save.h>

namespace OctreeLoadSave
{

SparseMask MaskToSparseMask(const cv::Mat & mask)
{
  const uint64 width = mask.cols;
  const uint64 height = mask.rows;

  SparseMask result;
  result.width = width;
  result.height = height;

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (mask.at<uint8>(y, x))
        result.indices.push_back(Point2lu(y, x));
    }

  return result;
}

SparseMask3D MaskToSparseMask3D(const cv::Mat & mask)
{
  const uint64 width = mask.size[2];
  const uint64 height = mask.size[1];
  const uint64 depth = mask.size[0];

  SparseMask3D result;
  result.width = width;
  result.height = height;
  result.depth = depth;

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (mask.at<uint8>(z, y, x))
          result.indices.push_back(Point3lu(z, y, x));
      }

  return result;
}

bool DeserializeOctreeHeader(std::istream & ifile, uint64 & version, uint64 & num_levels, uint64 & num_channels, uint64 & num_fields,
                             bool & is_3d)
{
  if (!ifile)
  {
    ROS_ERROR("DeserializeOctreeHeader: cannot open file.");
    return false;
  }

  std::string magic = "OCTREE2D";
  std::string magic3d = "OCTREE3D";

  std::vector<uint8> buffer(magic.size() + 1, 0); // zero-terminated
  ifile.read((char *)(buffer.data()), magic.size());
  std::string maybe_magic = (char *)(buffer.data());

  if (maybe_magic != magic && maybe_magic != magic3d)
  {
    ROS_ERROR("DeserializeOctreeHeader: expected magic %s or %s, got %s", magic.c_str(), magic3d.c_str(), maybe_magic.c_str());
    return false;
  }

  is_3d = (maybe_magic == magic3d);

  const uint64 ok_version = 1;
  ifile.read((char *)&version, sizeof(version));
  if (version != ok_version)
  {
    ROS_ERROR("DeserializeOctreeHeader: mismatched version: expected %d, got %d", int(ok_version), int(version));
    return false;
  }

  ifile.read((char *)&num_levels, sizeof(num_levels));

  ifile.read((char *)&num_channels, sizeof(num_channels));

  ifile.read((char *)&num_fields, sizeof(num_fields));

  if (!ifile)
  {
    ROS_ERROR("DeserializeOctreeHeader: unable to read num_fields.");
    return false;
  }

  return true;
}

}
