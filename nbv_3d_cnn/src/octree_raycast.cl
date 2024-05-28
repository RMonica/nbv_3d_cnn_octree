#ifndef __OPENCL_VERSION__
  #define global
  #define __global
  #define kernel
#endif

#define SQR(x) ((x)*(x))

#ifndef NULL
  #define NULL (0)
#endif

bool Equal2(int2 a, int2 b)
{
  return a.x == b.x && a.y == b.y;
}

bool Equal3(int3 a, int3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool Equal3f(float3 a, float3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

int2 FloatToInt2(float2 f)
{
  return (int2)(f.x, f.y);
}

int2 UintToInt2(uint2 f)
{
  return (int2)(f.x, f.y);
}

float3 UintToFloat3(uint3 f)
{
  return (float3)(f.x, f.y, f.z);
}

float3 IntToFloat3(int3 f)
{
  return (float3)(f.x, f.y, f.z);
}

int3 FloatToInt3(float3 f)
{
  return (int3)(f.x, f.y, f.z);
}

float3 Float4ToFloat3(float4 f)
{
  return (float3)(f.x, f.y, f.z);
}

void SetElem3f(float3 * v, int index, float value)
{
  switch (index)
  {
    case 0: v->x = value; break;
    case 1: v->y = value; break;
    case 2: v->z = value; break;
  }
}

float GetElem3f(const float3 v, int index)
{
  switch (index)
  {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
  }

  return v.x;
}

void SetElem3i(int3 * v, int index, int value)
{
  switch (index)
  {
    case 0: v->x = value; break;
    case 1: v->y = value; break;
    case 2: v->z = value; break;
  }
}

int GetElem3i(const int3 v, int index)
{
  switch (index)
  {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
  }

  return v.x;
}

void SetElem3u(uint3 * v, int index, uint value)
{
  switch (index)
  {
    case 0: v->x = value; break;
    case 1: v->y = value; break;
    case 2: v->z = value; break;
  }
}

uint GetElem3u(const uint3 v, int index)
{
  switch (index)
  {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
  }

  return v.x;
}

float4 quat_Inverse(const float4 q)
{
  return (float4)(-q.xyz, q.w);
}

float4 quat_Mult(const float4 q1, const float4 q2)
{
  float4 q;
  q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
  q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
  q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
  q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);

  return q;
}

float3 quat_ApplyRotation(const float4 q, const float3 v)
{
  const float4 ev = (float4)(v.xyz, 0.0f);
  const float4 iq = quat_Inverse(q);
  const float4 result = quat_Mult(quat_Mult(q, ev), iq);
  return result.xyz;
}

void EvaluateRayNoOctree(global const float * occupied_environment,
                         global const float * empty_environment,
                         const uint width,
                         const uint height,
                         const uint depth,
                         const uchar is_3d,
                         const float sensor_focal_length,
                         float3 origin,
                         float3 ray_direction,
                         const float max_range,
                         const float min_range,
                         const float a_priori_occupied_prob,
                         float * hits,
                         float * miss)
{
  if (Equal3f(ray_direction, (float3)(0, 0, 0)))
  {
    *hits = 0.0f;
    *miss = 1.0f;
    return; // invalid
  }

  *hits = 0.0f;
  *miss = 0.0f;

  float prob_in_min_range = 1.0f;
  uint min_range_i = (uint)min_range;
  for (uint z = 0; z < min_range_i; z++)
  {
    float3 pt = ray_direction * (float)(z) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);

    float oe;
    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
    {
      oe = 0.0;
    }
    else
    {
      oe = occupied_environment[ipt.z * width * height + ipt.y * width + ipt.x];
    }

    prob_in_min_range = (1.0 - oe) * prob_in_min_range;
  }

  uint max_range_i = (uint)max_range;
  float occluded = 0.0f;
  for (uint z = 0; z < max_range_i; z++)
  {
    float3 pt = ray_direction * (float)(z) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);

    float dimensions = (depth <= 1) ? 1.0 : 2.0;
    float pw = pow((float)(z) / sensor_focal_length, dimensions);
    float distance_weight = min(pw, 1.0f);

    float oe, ee, ue;
    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
    {
      oe = 0.0;
      ee = 1.0;
      ue = 0.0;
    }
    else
    {
      oe = occupied_environment[ipt.z * width * height + ipt.y * width + ipt.x];
      ee = empty_environment[ipt.z * width * height + ipt.y * width + ipt.x] > 0.5f ? 1.0f : 0.0f;
      ue = (empty_environment[ipt.z * width * height + ipt.y * width + ipt.x] > -0.5f) &&
           (empty_environment[ipt.z * width * height + ipt.y * width + ipt.x] < 0.5f) ? 1.0f : 0.0f;
    }


    float prob_unknown = ue;
    float prob_unknown_and_reachable = prob_unknown * (1.0 - occluded) * prob_in_min_range;
    *hits += distance_weight * prob_unknown_and_reachable;
    *miss += distance_weight * (1.0f - prob_unknown_and_reachable);

    float prob_occluding_if_unknown = prob_unknown * a_priori_occupied_prob;
    float prob_occluding_if_occupied = oe;
    // new_reachable = old_reachable * empty_prob
    // ->
    // (1 - new_occluded) = (1 - old_occluded) * (1 - occupied_prob)
    occluded = 1.0f - (1.0f - occluded) * (1.0f - prob_occluding_if_occupied - prob_occluding_if_unknown);
  }
}

void kernel SimulateMultiViewWithInformationGain(global const float * occupied_environment,
                                                 global const float * empty_environment,
                                                 const uint width,
                                                 const uint height,
                                                 const uint depth,
                                                 const uchar is_3d,
                                                 const float sensor_focal_length,
                                                 global const float4 * origins,
                                                 global const float4 * orientations,
                                                 global const float4 * local_orientations,
                                                 const uint num_orientations,
                                                 const uint num_local_orientations,
                                                 const float max_range,
                                                 const float min_range,
                                                 const float a_priori_occupied_prob,
                                                 global float * pixel_score
                                                 )
{
  const uint ray_id = get_global_id(0);
  const uint local_orientation_id = ray_id % num_local_orientations;
  const uint local_view_id = ray_id / num_local_orientations;

  float3 origin;
  float4 orientation;
  if (!num_orientations)
  {
    origin = Float4ToFloat3(origins[ray_id / num_local_orientations]);
    orientation = orientations[ray_id / num_local_orientations];
  }
  else
  {
    origin = Float4ToFloat3(origins[(ray_id / num_local_orientations) / num_orientations]);
    orientation = orientations[(ray_id / num_local_orientations) % num_orientations];
  }

  const float3 local_ray_dir = Float4ToFloat3(local_orientations[local_orientation_id]);
  const float3 ray_direction = quat_ApplyRotation(orientation, local_ray_dir);

  float ray_hits, ray_miss;
  EvaluateRayNoOctree(occupied_environment, empty_environment, width, height, depth,
                      is_3d, sensor_focal_length, origin, ray_direction, max_range, min_range,
                      a_priori_occupied_prob, &ray_hits, &ray_miss);

  float score = 0.0f;
  if (ray_hits + ray_miss > 0.0)
    score = ray_hits / (ray_hits + ray_miss);
  atomic_add(&(pixel_score[local_view_id]), (uint)(score * 1000.0f));
}

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

global float * SerializedOctreeAt(global uint * const octree,
                                  const uchar is_3d,
                                  const uint value_size,
                                  const int3 coords,
                                  uint * log2_voxel_size_voxels
                                 )
{
  const uint HEADER_SIZE = 6;
  const uint num_children = is_3d ? 8 : 4;

  //const uint num_levels             = octree[0];
  const uint num_levels             = NUM_LEVELS;
  const uint base_image_depth       = octree[1];
  const uint base_image_height      = octree[2];
  const uint base_image_width       = octree[3];
  const uint first_children_pointer = octree[4];
  const uint first_value_pointer    = octree[5];

  global uint * const base_image = octree + HEADER_SIZE;
  global uint * const children = octree + first_children_pointer;
  global uint * const values = octree + first_value_pointer;

  const uint small_divisor = (1 << (num_levels - 1));
  const uint3 small_coords = (uint3)(coords.x / small_divisor, coords.y / small_divisor, coords.z / small_divisor);

  uint c_i = small_coords.x + small_coords.y * base_image_width + small_coords.z * base_image_width * base_image_height;
  if (base_image[c_i] == (uint)(-1))
  {
    return NULL;
  }
  c_i = base_image[c_i];

  for (uint l = 0; l < num_levels; l++)
  {
    global uint * const c_value = values + (c_i * value_size);

    if (l + 2 > num_levels)
      c_i = (uint)(-1);
    else
    {
      const uint curr_divisor = (1 << (num_levels - l - 2));
      uint3 c_l_i = (uint3)((coords.x / curr_divisor) & 1, (coords.y / curr_divisor) & 1, (coords.z / curr_divisor) & 1);

      uint child_idx = GetElem3u(c_l_i, 0) + GetElem3u(c_l_i, 1) * 2 + GetElem3u(c_l_i, 2) * 4;

      global uint * const c_children = children + (c_i * num_children);
      c_i = c_children[child_idx];
    }

    if (c_i == (uint)(-1))
    {
      if (log2_voxel_size_voxels)
        (*log2_voxel_size_voxels) = num_levels - l - 1;

      return (global float *)c_value;
    }
  }

  return NULL;
}

float ln(float v)
{
  return log2(v)/log2(2.718281828f);
}

int ComputeZInc(const uint log2_voxel_size_voxels, const int3 ipt, const float3 ray_direction)
{
  if (log2_voxel_size_voxels == 0) // we are in a 1x1 voxel, jump is 1
    return 1;

  // we are in a large voxel
  uint mask = ~((1 << log2_voxel_size_voxels) - 1);
  int3 this_voxel_origin = (int3)(ipt.x & mask, ipt.y & mask, ipt.z & mask);

  int voxel_size_voxels = 1 << log2_voxel_size_voxels;

  // find closest distance so that ray leaves voxel
  int min_dist = 20000; // initialize to large value
  #pragma unroll
  for (int i = 0; i < 3; i++)
  {
    const float dir = GetElem3f(ray_direction, i);
    if (fabs(dir) < 0.0001f) // too close to zero, no decision
      continue;
    int dist;
    // if dir > 0, compute wrt end of voxel
    const int inc = (dir > 0.0f) ? voxel_size_voxels : -1;
    dist = (int)ceil((float)(GetElem3i(this_voxel_origin, i) + inc - GetElem3i(ipt, i)) / dir);
    if (dist < min_dist)
      min_dist = dist;
  }

  const int z_inc = max(min_dist, 0); // increase z to leave this voxel
  return z_inc;
}

void EvaluateRayOctree(global uint * occupied_octree,
                       const uint width,
                       const uint height,
                       const uint depth,
                       const uchar is_3d,
                       const float sensor_focal_length,
                       float3 origin,
                       float3 ray_direction,
                       const float max_range,
                       const float min_range,
                       const float a_priori_occupied_prob,
                       float * hits,
                       float * miss)
{
  if (Equal3f(ray_direction, (float3)(0, 0, 0)))
  {
    *hits = 0.0f;
    *miss = 1.0f;
    return; // invalid
  }

  *hits = 0.0f;
  *miss = 0.0f;

  float prob_in_min_range = 1.0f;
  uint min_range_i = (uint)min_range;
  for (uint z = 0; z < min_range_i; z++)
  {
    float3 pt = ray_direction * (float)(z) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);
    int z_inc = 0;
    float jump_size = 1.0f;

    float oe;
    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
    {
      oe = 0.0;
    }
    else
    {
      uint log2_voxel_size_voxels;
      global const float * octree_at = SerializedOctreeAt(occupied_octree, is_3d, 2, ipt, &log2_voxel_size_voxels);
      oe = octree_at[0];

      if (log2_voxel_size_voxels != 0)
      {
        int maybe_z_inc = ComputeZInc(log2_voxel_size_voxels, ipt, ray_direction);
        if (maybe_z_inc + z > min_range_i)
          maybe_z_inc = min_range_i - z;
        z_inc = max(maybe_z_inc - 1, 0);
        jump_size = (float)maybe_z_inc;
      }
    }

    prob_in_min_range = pow(1.0f - oe, jump_size) * prob_in_min_range;

    z += z_inc;
  }

  uint max_range_i = (uint)max_range;
  float occluded = 0.0f;
  for (uint z = 0; z < max_range_i; z++)
  {
    float3 pt = ray_direction * (float)(z) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);

    float oe, ee, ue;
    float jump_size = 1.0f;
    int z_inc = 0;
    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
    {
      oe = 0.0;
      ee = 1.0;
      ue = 0.0;
    }
    else
    {
      uint log2_voxel_size_voxels;
      global const float * octree_at = SerializedOctreeAt(occupied_octree, is_3d, 2, ipt, &log2_voxel_size_voxels);
      oe = octree_at[0];
      ee = (octree_at[1] > 0.5f) ? 1.0f : 0.0f;
      ue = (octree_at[1] > -0.5f) && (octree_at[1] < 0.5f) ? 1.0f : 0.0f;

      if (log2_voxel_size_voxels != 0)
      {
        int maybe_z_inc = ComputeZInc(log2_voxel_size_voxels, ipt, ray_direction);
        if (maybe_z_inc + z > max_range_i)
          maybe_z_inc = max_range_i - z;
        z_inc = max(maybe_z_inc - 1, 0);
        jump_size = (float)maybe_z_inc;
      }
    }

    float dimensions = (!is_3d) ? 1.0 : 2.0;
    float pw = pow(((float)(z) + jump_size / 2) / sensor_focal_length, dimensions);
    float distance_weight = min(pw, 1.0f);

    float prob_unknown = ue;

    float prob_occluding_if_occupied = oe;
    float prob_next_reachable = 1.0f - prob_occluding_if_occupied;

    float local_hits = 0.0f;
    const float PRECISION = 0.0001f;
    float prob_unknown_and_reachable = prob_unknown * (1.0 - occluded) * prob_in_min_range;
    if (prob_next_reachable < (1.0f - PRECISION))
      local_hits = prob_unknown_and_reachable * ((1.0f - pow(prob_next_reachable, jump_size)) / (1.0f - prob_next_reachable));
    else
      local_hits = prob_unknown_and_reachable * jump_size;
    *hits += distance_weight * local_hits;
    *miss += distance_weight * (jump_size - local_hits);

    // new_reachable = old_reachable * pow(empty_prob, jump_size)
    // ->
    // (1 - new_occluded) = (1 - old_occluded) * pow((1 - occupied_prob), jump_size)
    occluded = 1.0f - (1.0f - occluded) * pow(prob_next_reachable, jump_size);
    z += z_inc;
  }
}

void kernel SimulateMultiRayWithInformationGain(global uint * occupied_octree,
                                                const uint width,
                                                const uint height,
                                                const uint depth,
                                                const uchar is_3d,
                                                const uint rays_size,
                                                const float sensor_focal_length,
                                                global const float * origins,
                                                global const float * orientations,
                                                const float max_range,
                                                const float min_range,
                                                const float a_priori_occupied_prob,
                                                global float * hits,
                                                global float * miss
                                                )
{
  const int ray_id = get_global_id(0);

  const float3 origin = (float3)(origins[ray_id], origins[ray_id + rays_size],
                                 origins[ray_id + 2 * rays_size]);

  const float3 ray_direction = (float3)(orientations[ray_id], orientations[ray_id + rays_size],
                                      orientations[ray_id + 2 * rays_size]);

  float ray_hits, ray_miss;
  EvaluateRayOctree(occupied_octree, width, height, depth, is_3d, sensor_focal_length, origin, ray_direction, max_range, min_range,
              a_priori_occupied_prob, &ray_hits, &ray_miss);

  hits[ray_id] = ray_hits;
  miss[ray_id] = ray_miss;
}

void kernel SimulateMultiViewWithInformationGainOctree(global uint * occupied_octree,
                                                       const uint width,
                                                       const uint height,
                                                       const uint depth,
                                                       const uchar is_3d,
                                                       const float sensor_focal_length,
                                                       global const float4 * origins,
                                                       global const float4 * orientations,
                                                       global const float4 * local_orientations,
                                                       const uint num_orientations,
                                                       const uint num_local_orientations,
                                                       const float max_range,
                                                       const float min_range,
                                                       const float a_priori_occupied_prob,
                                                       global uint * pixel_score
                                                       )
{
  const uint ray_id = get_global_id(0);
  const uint local_orientation_id = ray_id % num_local_orientations;
  const uint local_view_id = ray_id / num_local_orientations;

  float3 origin;
  float4 orientation;
  if (!num_orientations)
  {
    origin = Float4ToFloat3(origins[ray_id / num_local_orientations]);
    orientation = orientations[ray_id / num_local_orientations];
  }
  else
  {
    origin = Float4ToFloat3(origins[(ray_id / num_local_orientations) / num_orientations]);
    orientation = orientations[(ray_id / num_local_orientations) % num_orientations];
  }

  const float3 local_ray_dir = Float4ToFloat3(local_orientations[local_orientation_id]);
  const float3 ray_direction = quat_ApplyRotation(orientation, local_ray_dir);

  float ray_hits, ray_miss;
  EvaluateRayOctree(occupied_octree, width, height, depth, is_3d, sensor_focal_length, origin, ray_direction, max_range, min_range,
              a_priori_occupied_prob, &ray_hits, &ray_miss);

  float score = 0.0f;
  if (ray_hits + ray_miss > 0.0)
    score = ray_hits / (ray_hits + ray_miss);
  atomic_add(&(pixel_score[local_view_id]), (uint)(score * 1000.0f));
}

void kernel FillUint(
                     const uint c,
                     global uint * to_be_filled
                     )
{
  const int x = get_global_id(0);
  to_be_filled[x] = c;
}

void kernel FillFloat(
                      const float c,
                      global float * to_be_filled
                      )
{
  const int x = get_global_id(0);
  to_be_filled[x] = c;
}
