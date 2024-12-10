import numpy as np
from scipy.sparse import csr_matrix
import filter_prj
import time




def generate_test_data(size):  # size 是单元数
    dc = [2,3,5,1,5]
    dc = np.array(dc)
    dv = [6,8,1,2,6]
    dv = np.array(dv)
    sum_Hf = [5,9,6,8,4]
    sum_Hf = np.array(sum_Hf)
    x = [9,8,4,2,5]
    x = np.array(x)
    v = [9,5,8,6,3]
    v = np.array(v)
    Hf = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
              [6.0, 7.0, 8.0, 9.0, 10.0],
              [11.0, 12.0, 13.0, 14.0, 15.0],
              [16.0, 17.0, 18.0, 19.0, 20.0],
              [21.0, 22.0, 23.0, 24.0, 25.0]], dtype=np.float32)

    sparse_Hf = csr_matrix(Hf)
    eta = 0.5
    beta = 4
    return dc, dv, sparse_Hf, sum_Hf, x, v, eta, beta




# 测试不同的数据大小
sizes = [5]


for size in sizes:
    dc, dv, Hf, sum_Hf, x, v, eta, beta = generate_test_data(size)


    # 测试 C++ 库中的函数
    start_time = time.time()
    result_de_checkboard1_cpp = filter_prj.de_checkboard1(dc, dv, Hf, sum_Hf)
    # dc,dv = filter_prj.de_checkboard1(dc, dv, Hf, sum_Hf)
    result_de_checkboard2_cpp = filter_prj.de_checkboard2(x, Hf, sum_Hf)
    result_prj_cpp = filter_prj.prj(v, eta, beta)
    result_dprj_cpp = filter_prj.dprj(v, eta, beta)
    end_time = time.time()
    cpp_time = end_time - start_time
    print(dc)
    print(f"C++ library execution time for size {size}: {cpp_time} seconds")
    print("result_de_checkboard1_cpp",result_de_checkboard1_cpp)
    print(result_de_checkboard2_cpp)
    print(result_prj_cpp)
    print(result_dprj_cpp)

    # 以下是 Python 代码中原始的函数定义，用于对比
    def de_checkboard1_py(dc, dv, Hf, sum_Hf):#矩阵相乘
        """
        本函数作用是进行过滤，
        输入：敏度、体积、过滤权重因子、过滤权重因子之和
        return：过滤后的 dc 和 dv
        """
        corrected_dc = (dc @ Hf.T/sum_Hf ).flatten()
        corrected_dv = (dv @ Hf.T/sum_Hf ).flatten()
        return corrected_dc, corrected_dv


    def de_checkboard2_py(x, Hf, sum_Hf):#矩阵相乘
        """
        本函数作用是进行过滤，
        输入：设计变量、过滤权重因子、过滤权重因子之和
        return：过滤后的 x
        """
        corrected_x = (x @ Hf.T / sum_Hf).flatten()
        return corrected_x


    def prj_py(v, eta, beta):#数组的加减乘除
        """
        本函数作用是进行投影
        输入：xTilde 初始化的投影密度
        return：投影后的密度
        """
        q = (np.tanh(beta * eta) + np.tanh(beta * (v - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
        return q


    def dprj_py(v, eta, beta):#数组的加减乘除
        """
        本函数作用是进行投影
        输入：xTilde 投影密度
        return：投影后的密度
        """
        w = beta * (1 - np.tanh(beta * (v - eta)) ** 2) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
        return w


    # 测试 Python 代码中的函数
    start_time = time.time()
    result_de_checkboard1_py = de_checkboard1_py(dc, dv, Hf, sum_Hf)
    result_de_checkboard2_py = de_checkboard2_py(x, Hf, sum_Hf)
    result_prj_py = prj_py(v, eta, beta)
    result_dprj_py = dprj_py(v, eta, beta)
    print("以下是python结果")
    print(dc)
    print(result_de_checkboard1_py)
    print(result_de_checkboard2_py)
    print(result_prj_py)
    print(result_dprj_py)



    end_time = time.time()
    py_time = end_time - start_time
    print(f"Python code execution time for size {size}: {py_time} seconds")


    # 对比结果
    def compare_results(result_cpp, result_py):
        """
        比较 C++ 库和 Python 代码的结果
        """
        if isinstance(result_cpp, tuple):
            for i in range(len(result_cpp[0])):
                if not np.isclose(result_cpp[0][i], result_py[0][i]):
                    print(f"de_checkboard1 result mismatch at index {i}: C++ {result_cpp[0][i]} vs Python {result_py[0][i]}")
            for i in range(len(result_cpp[1])):
                if not np.isclose(result_cpp[1][i], result_py[1][i]):
                    print(f"de_checkboard1 result mismatch at index {i}: C++ {result_cpp[1][i]} vs Python {result_py[1][i]}")
        else:
            if isinstance(result_cpp, list):
                result_cpp = np.array(result_cpp)
            if isinstance(result_py, np.ndarray):
                if result_cpp.shape == result_py.shape:
                    # 元素级别的比较
                    mismatch = ~np.isclose(result_cpp, result_py)
                    if mismatch.any():
                        print(f"Result mismatch at indices: {np.where(mismatch)}")
                else:
                    print("Shape mismatch between result_cpp and result_py")
            else:
                if isinstance(result_cpp, np.ndarray):
                    # 元素级别的比较
                    mismatch = ~np.isclose(result_cpp, result_py)
                    if mismatch.any():
                        print(f"Result mismatch at indices: {np.where(mismatch)}")
                else:
                    if isinstance(result_cpp, float):
                        if not np.isclose(result_cpp, result_py):
                            print(f"Result mismatch: C++ {result_cpp} vs Python {result_py}")
                    else:
                        for i in range(len(result_cpp)):
                            if not np.isclose(result_cpp[i], result_py[i]):
                                print(f"Result mismatch at index {i}: C++ {result_cpp[i]} vs Python {result_py[i]}")


    # compare_results(result_de_checkboard1_cpp, result_de_checkboard1_py)
    compare_results(result_de_checkboard2_cpp, result_de_checkboard2_py)
    compare_results(result_prj_cpp, [result_prj_py])
    compare_results(result_dprj_cpp, [result_dprj_py])
