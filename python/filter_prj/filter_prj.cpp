#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace py = pybind11;


// 原有的 de_checkboard1 函数
std::pair<std::vector<double>, std::vector<double>> 
de_checkboard1(const std::vector<double>& dc, const std::vector<double>& dv, 
                const Eigen::SparseMatrix<double>& Hf, const std::vector<double>& sum_Hf) 
{
    Eigen::Map<const Eigen::VectorXd> dc_map(dc.data(), dc.size());
    Eigen::Map<const Eigen::VectorXd> dv_map(dv.data(), dv.size());
    Eigen::Map<const Eigen::VectorXd> sum_Hf_map(sum_Hf.data(), sum_Hf.size());
    if (dc_map.size() == Hf.cols()) {
        // 对 sum_Hf 和计算结果进行逐元素除法
        Eigen::VectorXd corrected_dc = (dc_map.transpose() * Hf.transpose()).array() / sum_Hf_map.transpose().array();
        Eigen::VectorXd corrected_dv = (dv_map.transpose() * Hf.transpose()).array() / sum_Hf_map.transpose().array();
        // 将 Eigen::VectorXd 转换为 std::vector<double>
        return std::make_pair(std::vector<double>(corrected_dc.data(), corrected_dc.data() + corrected_dc.size()),
                         std::vector<double>(corrected_dv.data(), corrected_dv.data() + corrected_dv.size()));
    } else {
        return std::make_pair(std::vector<double>(), std::vector<double>());
    }
}


// 原有的 de_checkboard2 函数
std::vector<double> 
de_checkboard2(const std::vector<double>& x, const Eigen::SparseMatrix<double>& Hf, const std::vector<double>& sum_Hf) 
{
    Eigen::Map<const Eigen::VectorXd> x_map(x.data(), x.size());
    Eigen::Map<const Eigen::VectorXd> sum_Hf_map(sum_Hf.data(), sum_Hf.size());
    if (x_map.size() == Hf.cols()) {
        // 对 sum_Hf 和计算结果进行逐元素除法
        Eigen::VectorXd corrected_x = (x_map.transpose() * Hf.transpose()).array() / sum_Hf_map.transpose().array();


        // 将 Eigen::VectorXd 转换为 std::vector<double>
        return std::vector<double>(corrected_x.data(), corrected_x.data() + corrected_x.size());
    } else {
        // 如果维度不匹配，返回空 vector
        return std::vector<double>();
    }
}


// 原有的 prj 函数
std::vector<double> 
prj(const std::vector<double>& v, double eta, double beta) 
{
    std::vector<double> result;
    for (double val : v) {
        double q = (std::tanh(beta * eta) + std::tanh(beta * (val - eta))) /
             (std::tanh(beta * eta) + std::tanh(beta * (1 - eta)));
        result.push_back(q);
    }
    return result;
}


// 原有的 dprj 函数
std::vector<double> 
dprj(const std::vector<double>& v, double eta, double beta) {
    std::vector<double> result;
    double tanh_beta_eta = std::tanh(beta * eta);
    double tanh_beta_one_minus_eta = std::tanh(beta * (1 - eta));


    for (double val : v) {
        double diff = beta * (val - eta);
        double w = beta * (1 - std::tanh(diff) * std::tanh(diff)) / 
                   (tanh_beta_eta + tanh_beta_one_minus_eta);
        result.push_back(w);
    }
    return result;
}


// 绑定函数到 Python
PYBIND11_MODULE(filter_prj, m) {
    m.def("de_checkboard1", &de_checkboard1, "A function that does something");
    m.def("de_checkboard2", &de_checkboard2, "Another function that does something");
    m.def("prj", &prj, "A function that does something");
    m.def("dprj", &dprj, "A function that does something");
}
