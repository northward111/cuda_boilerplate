#include <iostream>
#include "kernel.cuh"


using namespace std;
int main()
{
    std::vector<Eigen::Vector3d> v1(10, Eigen::Vector3d{ 1.0, 1.0, 1.0 });
    std::vector<Eigen::Vector3d> v2(10, Eigen::Vector3d{ -1.0, 1.0, 1.0 });
    double x = kernel::dot(v1,v2);
    cout<<x<<endl;
    return 0;
}
