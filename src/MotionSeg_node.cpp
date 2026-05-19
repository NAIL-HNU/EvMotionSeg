#include "motionseg/MotionSeg.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "MotionSeg");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  MotionSeg ms(nh, nh_private);

  ros::spin();

  return 0;
}
