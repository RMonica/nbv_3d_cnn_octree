#include <pcl/io/ply_io.h>

#include <string>
#include <iostream>

#include "octree.h"

int main(int argc, char ** argv)
{
  if (argc < 3)
  {
    std::cout << "Usage: octree_to_points source.octree dest.ply" << std::endl;
  }

  const std::string source_filename = argv[1];
  const std::string dest_filename = argv[2];

  Octree octree;
  const bool success = octree.read_octree(source_filename);
  if (!success)
  {
    std::cout << "Could not read octree: " << source_filename << std::endl;
    std::exit(1);
  }

  const OctreeInfo & info = octree.get_info();
  std::cout << "Octree depth: " << info.depth() << std::endl;
  for (int d = 0; d < 10; d++)
    std::cout << "Node num (depth " << d << "): " << info.node_num(d) << std::endl;

  const int depth_start = 0;
  const int depth_end = 10;
  Points points;
  octree.octree2pts(points, depth_start, depth_end);
  points.write_ply(dest_filename);

  return 0;
}
