#include "cuda_common.cuh"

using namespace std;

void queryDevice() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  float kilo = (1 << 10);
  float mega = (1 << 20);
  float giga = (1 << 30);

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    cout << "Device Number: " << i << endl;
    cout << "  Device name: " << prop.name << endl;
    cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
    cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
    cout << "  Peak Memory Bandwidth (GB/s): "
         << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / mega
         << endl;
    cout << "  Total Global Memory (GB): " << prop.totalGlobalMem / giga
         << endl;
    cout << "  Total Constant Memory (KB): " << prop.totalConstMem / kilo
         << endl;
    cout << "  Total Shared Memory per Block (KB): "
         << prop.sharedMemPerBlock / kilo << endl;
    cout << "  Total Registers per Block: " << prop.regsPerBlock << endl;
    cout << "  Warp Size: " << prop.warpSize << endl;
    cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "  Max Threads Dimension: " << prop.maxThreadsDim[0] << " x "
         << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
    cout << "  Max Grid Size: " << prop.maxGridSize[0] << " x "
         << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;

    cout << "  MultiProcessor Count: " << prop.multiProcessorCount << endl;
    cout << "  Max Threads per MultiProcessor: "
         << prop.maxThreadsPerMultiProcessor << endl;
    cout << "  Max Threads per SM: " << prop.maxThreadsPerMultiProcessor
         << endl;
    cout << "  Max Blocks per SM: "
         << prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock << endl;
    cout << "  Max Blocks per MP: "
         << prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock << endl;
    cout << "  Max Blocks per Grid: " << prop.maxGridSize[0] << " x "
         << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
    cout << "  Max Threads per Grid: "
         << prop.maxGridSize[0] * prop.maxGridSize[1] * prop.maxGridSize[2] *
                prop.maxThreadsPerBlock
         << endl;

    cout << "  L2 Cache Size (KB): " << prop.l2CacheSize / kilo << endl;
    cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
  }
}