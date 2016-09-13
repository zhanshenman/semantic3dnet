#include "point_cloud_util.h"

int main(int argc, char **argv) {
  std::string path = "";
  if(argc > 1) path = std::string(argv[1]);
  std::cout << "preparing " << path << " for training" << std::endl;
  DataLoaderArray data_loader_array(path);
  data_loader_array.aggregate_in_volume(10000);   
  return 0;
}
