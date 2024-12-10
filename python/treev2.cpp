#include <vector>
#include <stdexcept>
#include "F:\CPP_install\eigen-master\Eigen\Dense"
#include "F:\CPP_install\eigen-master\Eigen\Sparse"
#include "F:\CPP_install\nanoflann-1.6.2\include\nanoflann.hpp"

#include <cmath>
#include <omp.h>
#include <thread>

#include <iostream>
#include <fstream>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace Eigen;
using namespace nanoflann;
using namespace std;


// 定义 KD 树适配器
struct PointCloud {
    Eigen::MatrixXd points; // 存储点云的矩阵，行表示点，列表示坐标

    inline size_t kdtree_get_point_count() const 
    {
        return points.rows(); 
    }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const 
    {
        return points(idx, dim); 
    }

    template <class BBOX> 
    bool kdtree_get_bbox(BBOX& /* bb */) const 
    { 
        return false; 
    }
};

typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor <double, PointCloud>, 
    PointCloud, 3 > KDTree;


Eigen::MatrixXd readDataToMatrixParallel(const std::string& filename) {
    // 打开文件
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件：" + filename);
    }

    std::vector<std::string> lines;
    std::string line;

    // 读取文件的所有行
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    file.close();

    // 检查是否为空
    if (lines.empty()) {
        throw std::runtime_error("文件内容为空！");
    }

    // 解析第一行，确定列数
    std::istringstream first_line(lines[0]);
    std::vector<double> temp_row;
    double value;
    while (first_line >> value) {
        temp_row.push_back(value);
    }
    size_t cols = temp_row.size();

    // 检查行的长度是否一致
    size_t rows = lines.size();
    Eigen::MatrixXd matrix(rows, cols);

    // 使用 OpenMP 并行解析每一行数据
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        std::istringstream iss(lines[i]);
        std::vector<double> row;
        while (iss >> value) {
            row.push_back(value);
        }

        // 检查当前行是否符合列数
        if (row.size() != cols) {
            #pragma omp critical
            {
                throw std::runtime_error("行长度不一致！");
            }
        }

        // 填充矩阵
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = row[j];
        }
    }

    return matrix;
}

std::vector<Eigen::Triplet<float>> 
get_triplets(Eigen::MatrixXd & centers, double rmin) 
{
    if (rmin < 0) {
        throw std::invalid_argument("rmin 不能为负数");
    }
    if (centers.rows() == 0) {
        throw std::invalid_argument("CENTERS 数组不能为空");
    }

    size_t length = centers.rows();
    // 提取 xyz 坐标
    Eigen::MatrixXd centers_xyz = centers.block(0, 1, length, centers.cols() - 1); 
    cout << "Centers xyz rows: \n" << centers_xyz.rows() << endl;

    PointCloud cloud;
    cloud.points = centers_xyz;

    KDTree tree(centers_xyz.cols(), cloud, KDTreeSingleIndexAdaptorParams(50));
    tree.buildIndex();
    cout << "Init kdtree ok" << endl;

    std::vector<Eigen::Triplet<float>> triplets;
    #pragma omp parallel
    {
        std::vector<Eigen::Triplet<float>> local_triplets; 

        #pragma omp for schedule(static)
        for (size_t i = 0; i < length; ++i) {
            std::vector<nanoflann::ResultItem<uint32_t, double>> indices_dists;
            std::vector<double> query_point(centers_xyz.cols());

            for (int d = 0; d < centers_xyz.cols(); ++d) {
                query_point[d] = centers_xyz(i, d);
            }

            size_t len = tree.radiusSearch(query_point.data(), rmin * rmin, indices_dists, nanoflann::SearchParameters(0.0));

            for (const auto& pair : indices_dists) {
                size_t j = pair.first;
                // if (i <= j) { // 避免重复存储
                    double dist = rmin - std::sqrt(pair.second);
                    local_triplets.emplace_back(i, j, static_cast<float>(dist));
                // }
            }
        }

        #pragma omp critical
        {
            triplets.insert(triplets.end(), local_triplets.begin(), local_triplets.end());
        }
    }

    cout << "Radius search done. sparse matrix length: " << length << endl;
    return triplets;
}

Eigen::SparseMatrix<float> 
get_distance_spare(size_t length, std::vector<Eigen::Triplet<float>> triplets) 
{
    Eigen::SparseMatrix<float> distance_sparse(length, length);

    distance_sparse.setFromTriplets(triplets.begin(), triplets.end());
    // 对称化矩阵
    // distance_sparse = distance_sparse + Eigen::SparseMatrix<float>(distance_sparse.transpose());
    // 移除对角线的重复值
    // distance_sparse.prune([](float value, int, int) { return value != 0; }); 

    return distance_sparse;
}

Eigen::SparseMatrix<float>
get_distance_table(double rmin, const std::string &name) {
    // 调用函数读取矩阵
    Eigen::MatrixXd matrix = readDataToMatrixParallel(name);
    size_t length = matrix.rows();

    cout << "convert data to matrix" << endl;
    cout << "length: " << length << endl;

    std::vector<Eigen::Triplet<float>> triplets = get_triplets(matrix, rmin);
    Eigen::SparseMatrix<float> distance_sparse = get_distance_spare(length, triplets);
    
    return distance_sparse;
}

void set_syscore_para() {
    // 获取系统支持的线程数
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        std::cerr << "Unable to detect hardware concurrency. Defaulting to 1 thread." << std::endl;
        num_threads = 1; // 默认至少1个线程
    }

    // 设置 OMP_NUM_THREADS 环境变量
    std::string omp_env = "OMP_NUM_THREADS=" + std::to_string(num_threads);
    if (_putenv(omp_env.c_str()) != 0) {
        std::cerr << "Failed to set OMP_NUM_THREADS environment variable." << std::endl;
        return;
    }

    // 输出设置信息
    std::cout << "OMP_NUM_THREADS set to " << num_threads << std::endl;
}


// int main() {
//     double rmin = 5;
//     // 读取 n*4 的数据文件
//     Eigen::MatrixXd data = readDataToMatrixParallel("E:\\work\\topo_secondraydevelop\\PythonC_Ansys_dev-1\\MBB120_40\\temp\\elements_centers.txt");
//     // 假设第一列是单元编号，后三列是 x,y,z 坐标
//     // 提取坐标部分
//     Eigen::MatrixXd coordinates = data.block(0, 1, data.rows(), 3);
//     // 假设我们要搜索的特定坐标
//     std::vector<double> targetPoint = {19.5,14.5,0.0};
//     // 存储所有点的点云
//     PointCloud allPoints;
//     allPoints.points = coordinates;
//     // 构建 KDTree 进行邻近搜索
//     KDTree allPointsTree(3, allPoints, KDTreeSingleIndexAdaptorParams(50));
//     allPointsTree.buildIndex();
//     std::vector<nanoflann::ResultItem<uint32_t, double>> indices_dists;
//     // 搜索邻近节点
//     size_t len = allPointsTree.radiusSearch(targetPoint.data(), rmin * rmin, indices_dists, nanoflann::SearchParameters(0.0));
    
//     // 存储索引和距离的结构体
//     struct IndexDistance {
//         size_t index;
//         double distance;
//     };
//     std::vector<IndexDistance> indexDistances;
//     for (const auto& pair : indices_dists) {
//         indexDistances.push_back({pair.first, sqrt(pair.second)});
//     }

//     // 对索引进行排序
//     std::sort(indexDistances.begin(), indexDistances.end(), [](const IndexDistance& a, const IndexDistance& b) {
//         return a.index < b.index;
//     });

//     // 输出排序后的结果
//     for (const auto& id : indexDistances) {
//         std::cout << "index: " << id.index << " dis: " << id.distance << std::endl;
//     }

//     return 0;
// }



// 包装函数
PYBIND11_MODULE(gdt, m) {
    m.doc() = "pybind11 gdt plugin";
    m.def("set_syscore_para", &set_syscore_para, "A function which set smp num");
    m.def("get_distance_table", &get_distance_table, "A function which get distance table");
}