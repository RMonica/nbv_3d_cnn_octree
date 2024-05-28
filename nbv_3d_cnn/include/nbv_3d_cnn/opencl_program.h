#ifndef OPENCL_PROGRAM_H
#define OPENCL_PROGRAM_H

#include <vector>
#include <memory>
#include <stdint.h>

#include <ros/ros.h>

// OpenCL
#include <CL/cl2.hpp>

#include <Eigen/Dense>

class OpenCLProgram
{
  public:
  explicit OpenCLProgram(ros::NodeHandle & nh, const std::string & program_source);

  typedef std::shared_ptr<cl::Context> CLContextPtr;
  typedef std::shared_ptr<cl::CommandQueue> CLCommandQueuePtr;
  typedef std::shared_ptr<cl::Buffer> CLBufferPtr;
  typedef std::shared_ptr<cl::Device> CLDevicePtr;
  typedef std::shared_ptr<cl::Program> CLProgramPtr;
  typedef std::shared_ptr<cl::Kernel> CLKernelPtr;
  typedef std::vector<cl_float, Eigen::aligned_allocator<cl_float> > CLFloatVector;
  typedef std::vector<cl_float2, Eigen::aligned_allocator<cl_float2> > CLFloat2Vector;
  typedef std::vector<cl_float4, Eigen::aligned_allocator<cl_float4> > CLFloat4Vector;
  typedef std::vector<cl_int2, Eigen::aligned_allocator<cl_int2> > CLInt2Vector;
  typedef std::vector<cl_int3, Eigen::aligned_allocator<cl_int3> > CLInt3Vector;
  typedef std::vector<cl_float, Eigen::aligned_allocator<cl_uchar> > CLUCharVector;
  typedef std::vector<cl_float3, Eigen::aligned_allocator<cl_float3> > CLFloat3Vector;
  typedef std::vector<cl_int, Eigen::aligned_allocator<cl_int> > CLInt32Vector;
  typedef std::vector<cl_uint, Eigen::aligned_allocator<cl_uint> > CLUInt32Vector;
  typedef std::vector<cl_ushort, Eigen::aligned_allocator<cl_ushort> > CLUInt16Vector;
  typedef std::vector<cl_ushort2, Eigen::aligned_allocator<cl_ushort2> > CLUShort2Vector;

  typedef uint64_t uint64;
  typedef uint8_t uint8;

  CLProgramPtr GetProgram() const {return m_opencl_program; }
  CLCommandQueuePtr GetQueue() const {return m_opencl_command_queue; }
  CLContextPtr GetContext() const {return m_opencl_context; }
  CLDevicePtr GetDevice() const {return m_opencl_device; }

  CLBufferPtr CreateBuffer(const CLContextPtr context,
                                 const size_t size,
                                 const std::string name) const;

  static cl_float2 EToCL(const Eigen::Vector2f & v)
  {
    cl_float2 result;
    result.s[0] = v[0];
    result.s[1] = v[1];
    return result;
  }

  static cl_int2 EToCL(const Eigen::Vector2i & v)
  {
    cl_int2 result;
    result.s[0] = v[0];
    result.s[1] = v[1];
    return result;
  }

  static cl_float3 EToCL(const Eigen::Vector3f & v)
  {
    cl_float3 result;
    result.s[0] = v[0];
    result.s[1] = v[1];
    result.s[2] = v[2];
    return result;
  }

  static cl_float4 EToCL(const Eigen::Vector4f & v)
  {
    cl_float4 result;
    result.s[0] = v[0];
    result.s[1] = v[1];
    result.s[2] = v[2];
    result.s[3] = v[3];
    return result;
  }

  static cl_int3 EToCL(const Eigen::Vector3i & v)
  {
    cl_int3 result;
    result.s[0] = v[0];
    result.s[1] = v[1];
    result.s[2] = v[2];
    return result;
  }

  static cl_float4 EToCL(const Eigen::Quaternionf & q)
  {
    cl_float4 result;
    result.s[0] = q.x();
    result.s[1] = q.y();
    result.s[2] = q.z();
    result.s[3] = q.w();
    return result;
  }

  static cl_float4 Float3ToFloat4(const cl_float3 f)
  {
    cl_float4 result;
    result.s[0] = f.s[0];
    result.s[1] = f.s[1];
    result.s[2] = f.s[2];
    result.s[3] = 0.0f;
    return result;
  }

  protected:
  CLContextPtr m_opencl_context;
  CLCommandQueuePtr m_opencl_command_queue;
  CLDevicePtr m_opencl_device;
  CLProgramPtr m_opencl_program;

  private:
  ros::NodeHandle & m_nh;
};

#endif // OPENCL_PROGRAM_H
