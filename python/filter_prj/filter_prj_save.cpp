#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "F:\CPP_install\eigen-master\Eigen/Dense"
#include "F:\CPP_install\eigen-master\Eigen/Sparse"
#include <random>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/eigen.h>

// namespace py = pybind11;
using namespace std;
pair<vector<double>, vector<double>> 
de_checkboard1(const vector<double>& dc, const vector<double>& dv, 
                const Eigen::SparseMatrix<double>& Hf, const vector<double>& sum_Hf) 
{
    Eigen::Map<const Eigen::VectorXd> dc_map(dc.data(), dc.size());
    Eigen::Map<const Eigen::VectorXd> dv_map(dv.data(), dv.size());
    Eigen::Map<const Eigen::VectorXd> sum_Hf_map(sum_Hf.data(), sum_Hf.size());
    if (dc_map.size() == Hf.cols()) {
        // 对sum_Hf和计算结果进行逐元素除法
        Eigen::VectorXd corrected_dc = (dc_map.transpose() * Hf.transpose()).array() / sum_Hf_map.transpose().array();
        Eigen::VectorXd corrected_dv = (dv_map.transpose() * Hf.transpose()).array() / sum_Hf_map.transpose().array();
        // 将 Eigen::VectorXd 转换为 std::vector<double>
        return make_pair(vector<double>(corrected_dc.data(), corrected_dc.data() + corrected_dc.size()),
                         vector<double>(corrected_dv.data(), corrected_dv.data() + corrected_dv.size()));
    } else {
        return make_pair(vector<double>(), vector<double>());
    }
}


vector<double> 
de_checkboard2(const vector<double>& x, const Eigen::SparseMatrix<double>& Hf, const vector<double>& sum_Hf) 
{
    Eigen::Map<const Eigen::VectorXd> x_map(x.data(), x.size());
    Eigen::Map<const Eigen::VectorXd> sum_Hf_map(sum_Hf.data(), sum_Hf.size());
    if (x_map.size() == Hf.cols()) {
        // 对sum_Hf和计算结果进行逐元素除法
        Eigen::VectorXd corrected_x = (x_map.transpose() * Hf.transpose()).array() / sum_Hf_map.transpose().array();

        // 将 Eigen::VectorXd 转换为 std::vector<double>
        return vector<double>(corrected_x.data(), corrected_x.data() + corrected_x.size());
    } else {
        // 如果维度不匹配，返回空 vector
        return vector<double>();
    }
}


vector<double> 
prj(const vector<double>& v, double eta, double beta) 
{
    vector<double> result;
    for (double val : v) {
        double q = (tanh(beta * eta) + tanh(beta * (val - eta))) /
             (tanh(beta * eta) + tanh(beta * (1 - eta)));
        result.push_back(q);
    }
    return result;
}

vector<double> 
dprj(const vector<double>& v, double eta, double beta) {
    vector<double> result;
    double tanh_beta_eta = tanh(beta * eta);
    double tanh_beta_one_minus_eta = tanh(beta * (1 - eta));

    for (double val : v) {
        double diff = beta * (val - eta);
        double w = beta * (1 - tanh(diff) * tanh(diff)) / 
                   (tanh_beta_eta + tanh_beta_one_minus_eta);
        result.push_back(w);
    }
    return result;
}





void generate_test_data(std::vector<double>& dc, std::vector<double>& dv, Eigen::SparseMatrix<double>& sparse_Hf, std::vector<double>& sum_Hf, std::vector<double>& x, std::vector<double>& v, double& eta, double& beta) {
    // 初始化 dc 向量
    dc = {2.0, 3.0, 5.0, 1.0, 5.0};
    // 初始化 dv 向量
    dv = {6.0, 8.0, 1.0, 2.0, 6.0};
    // 初始化 sum_Hf 向量
    sum_Hf = {5.0, 9.0, 6.0, 8.0, 4.0};
    // 初始化 x 向量
    x = {9.0, 8.0, 4.0, 2.0, 5.0};
    // 初始化 v 向量
    v = {9.0, 5.0, 8.0, 6.0, 3.0};


    // 初始化 Hf 矩阵
    Eigen::MatrixXd Hf(5, 5);
    Hf << 1.0, 2.0, 3.0, 4.0, 5.0,
          6.0, 7.0, 8.0, 9.0, 10.0,
          11.0, 12.0, 13.0, 14.0, 15.0,
          16.0, 17.0, 18.0, 19.0, 20.0,
          21.0, 22.0, 23.0, 24.0, 25.0;


    // 将 Hf 转换为稀疏矩阵
    sparse_Hf = Hf.sparseView();


    // 初始化 eta 和 beta
    eta = 0.5;
    beta = 4.0;
}


int main() {
    double eta, beta;

    std::vector<double> dc, dv, sum_Hf, x, v;
    Eigen::SparseMatrix<double> Hf;


    // 生成测试数据
    generate_test_data(dc, dv, Hf, sum_Hf, x, v, eta, beta);


    // 调用 de_checkboard1 函数
    std::pair<std::vector<double>, std::vector<double>> result_pair = de_checkboard1(dc, dv, Hf, sum_Hf);
    std::vector<double> dc_result = result_pair.first;
    std::vector<double> dv_result = result_pair.second;
    std::cout << "de_checkboard1 result for dc: ";
    for (size_t i = 0; i < dc_result.size(); ++i) {
        std::cout << dc_result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "de_checkboard1 result for dv: ";
    for (size_t i = 0; i < dv_result.size(); ++i) {
        std::cout << dv_result[i] << " ";
    }
    std::cout << std::endl;


    // 调用 de_checkboard2 函数
    std::vector<double> x_result = de_checkboard2(x, Hf, sum_Hf);
    std::cout << "de_checkboard2 result: ";
    for (size_t i = 0; i < x_result.size(); ++i) {
        std::cout << x_result[i] << " ";
    }
    std::cout << std::endl;


    // 调用 prj 函数
    std::vector<double> prj_result = prj(v, eta, beta);
    std::cout << "prj result: ";
    for (size_t i = 0; i < prj_result.size(); ++i) {
        std::cout << prj_result[i] << " ";
    }
    std::cout << std::endl;


    // 调用 dprj 函数
    std::vector<double> dprj_result = dprj(v, eta, beta);
    std::cout << "dprj result: ";
    for (size_t i = 0; i < dprj_result.size(); ++i) {
        std::cout << dprj_result[i] << " ";
    }
    std::cout << std::endl;


    return 0;
}

// int main() {
//     int size = 5;  // 选择单元数

//     // 定义变量
//     std::vector<double> dc, dv, x, v;
//     Eigen::SparseMatrix<double> Hf(size, size);
//     Eigen::VectorXd sum_Hf(size);
//     double eta, beta;

//     // 生成测试数据
//     generate_test_data(size, dc, dv, Hf, sum_Hf, x, v, eta, beta);

//     // 将 dc 转换为 Eigen::VectorXd 类型
//     Eigen::VectorXd dc_eigen = Eigen::VectorXd::Map(dc.data(), dc.size());

//     // 执行 dc.T * Hf，得到一个 n x 1 的结果
//     Eigen::VectorXd result = Hf * dc_eigen;

//     // 打印结果
//     std::cout << "Result (Hf * dc): \n" << result << std::endl;

//     // 逐元素除法：result / sum_Hf
//     Eigen::VectorXd result2 = result.array() / sum_Hf.array();

//     // 打印逐元素除法的结果
//     std::cout << "Result after element-wise division (result / sum_Hf): \n" << result2 << std::endl;

//     return 0;
// }