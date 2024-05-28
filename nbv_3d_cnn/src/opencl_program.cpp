#include <nbv_3d_cnn/opencl_program.h>

#include "generate_test_dataset.h"

OpenCLProgram::OpenCLProgram(ros::NodeHandle & nh, const std::string & source): m_nh(nh)
{
  std::string param_string;

  std::string platform_name;
  m_nh.param<std::string>(PARAM_NAME_OPENCL_PLATFORM_NAME, platform_name, PARAM_DEFAULT_OPENCL_PLATFORM_NAME);
  std::string device_name;
  m_nh.param<std::string>(PARAM_NAME_OPENCL_DEVICE_NAME, device_name, PARAM_DEFAULT_OPENCL_DEVICE_NAME);

  cl_device_type device_type;
  m_nh.param<std::string>(PARAM_NAME_OPENCL_DEVICE_TYPE, param_string, PARAM_DEFAULT_OPENCL_DEVICE_TYPE);
  if (param_string == PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL)
    device_type = CL_DEVICE_TYPE_ALL;
  else if (param_string == PARAM_VALUE_OPENCL_DEVICE_TYPE_CPU)
    device_type = CL_DEVICE_TYPE_CPU;
  else if (param_string == PARAM_VALUE_OPENCL_DEVICE_TYPE_GPU)
    device_type = CL_DEVICE_TYPE_GPU;
  else
  {
    ROS_ERROR("generate_test_dataset_opencl: invalid parameter opencl_device_type, value '%s', using '%s' instead.",
              param_string.c_str(), PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL);
    device_type = CL_DEVICE_TYPE_ALL;
  }

  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  if (all_platforms.empty())
  {
    ROS_ERROR("generate_test_dataset_opencl: opencl: no platforms found.");
    exit(1);
  }

  {
    std::string all_platform_names;
    for (uint64 i = 0; i < all_platforms.size(); i++)
      all_platform_names += "\n  -- " + all_platforms[i].getInfo<CL_PLATFORM_NAME>();
    ROS_INFO_STREAM("generate_test_dataset_opencl: opencl: found platforms:" << all_platform_names);
  }
  uint64 platform_id = 0;
  if (!platform_name.empty())
  {
    ROS_INFO("generate_test_dataset_opencl: looking for matching platform: %s", platform_name.c_str());
    for (uint64 i = 0; i < all_platforms.size(); i++)
    {
      const std::string plat = all_platforms[i].getInfo<CL_PLATFORM_NAME>();
      if (plat.find(platform_name) != std::string::npos)
      {
        ROS_INFO("generate_test_dataset_opencl: found matching platform: %s", plat.c_str());
        platform_id = i;
        break;
      }
    }
  }

  cl::Platform default_platform = all_platforms[platform_id];
  ROS_INFO_STREAM("generate_test_dataset_opencl: using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>());

  std::vector<cl::Device> all_devices;
  default_platform.getDevices(device_type, &all_devices);
  if (all_devices.empty())
  {
      ROS_INFO("generate_test_dataset_opencl: no devices found.");
      exit(1);
  }
  {
    std::string all_device_names;
    for (uint64 i = 0; i < all_devices.size(); i++)
      all_device_names += "\n  -- " + all_devices[i].getInfo<CL_DEVICE_NAME>();
    ROS_INFO_STREAM("generate_test_dataset_opencl: found devices:" << all_device_names);
  }
  uint64 device_id = 0;
  if (!device_name.empty())
  {
    ROS_INFO("generate_test_dataset_opencl: looking for matching device: %s", device_name.c_str());
    for (uint64 i = 0; i < all_devices.size(); i++)
    {
      const std::string dev = all_devices[i].getInfo<CL_DEVICE_NAME>();
      if (dev.find(device_name) != std::string::npos)
      {
        ROS_INFO("generate_test_dataset_opencl: found matching device: %s", dev.c_str());
        device_id = i;
        break;
      }
    }
  }

  cl::Device default_device = all_devices[device_id];
  m_opencl_device = CLDevicePtr(new cl::Device(default_device));
  ROS_INFO_STREAM("generate_test_dataset_opencl: using device: " << default_device.getInfo<CL_DEVICE_NAME>());

  m_opencl_context = CLContextPtr(new cl::Context({*m_opencl_device}));

  m_opencl_command_queue = CLCommandQueuePtr(new cl::CommandQueue(*m_opencl_context,*m_opencl_device));

  cl::Program::Sources sources;
  sources.push_back({source.c_str(),source.length()});

  ROS_INFO("generate_test_dataset_opencl: building program... ");
  m_opencl_program = CLProgramPtr(new cl::Program(*m_opencl_context,sources));
  if (m_opencl_program->build({*m_opencl_device}) != CL_SUCCESS)
  {
    ROS_ERROR_STREAM("generate_test_dataset_opencl: error building opencl_program: " <<
                     m_opencl_program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*m_opencl_device));
    exit(1);
  }
  ROS_INFO("generate_test_dataset_opencl: initialized.");
}

OpenCLProgram::CLBufferPtr OpenCLProgram::CreateBuffer(const CLContextPtr context,
                                                       const size_t size,
                                                       const std::string name) const
{
  cl_int err;
  CLBufferPtr buf = CLBufferPtr(new cl::Buffer(*context,CL_MEM_READ_WRITE,
                                    size, NULL, &err));
  if (err != CL_SUCCESS)
  {
    ROS_ERROR("could not allocate buffer '%s' of size %u, error %d", name.c_str(), unsigned(size), int(err));
  }
  return buf;
}
