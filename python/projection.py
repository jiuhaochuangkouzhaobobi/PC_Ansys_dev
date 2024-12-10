from numpy import *
import numpy as np
from finite_element_analysis import *
import mma
import gdt
import filter_prj
import os
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from scipy.sparse import diags
from ansys.mapdl.core import launch_mapdl,Mapdl
import vtk
import time
import subprocess
import pandas as pd


class SIMP(object):
    """
    本文件是主程序，调用finite_element_analysis.py有限元计算模块，获取模型数据、应变能等数据

    参数说明：

    element_counts：设计域的单元总数
    node_counts：设计域节点数
    CENTERS、V：设计域单元的中心坐标、体积
    flag：0：无对称；1：关于z轴120°对称；2：关于单平面对称;3：关于xoz、yoz两个平面对称
    """
    def __init__(self,flag):
        gdt.set_syscore_para()
        self.ansys_solver = FEA()
        self.mapdl = Mapdl("127.0.0.1", port=50052)
        self.mapdl.input(self.ansys_solver.target_dir+self.ansys_solver.get_meshmodel_file)
        counts = loadtxt(self.ansys_solver.workdir+'elements_nodes_counts.txt',dtype = int)
        self.element_counts,self.node_counts = counts[0],counts[1]
        self.CENTERS_path = self.ansys_solver.workdir+'elements_centers.txt'
        self.V = loadtxt(self.ansys_solver.workdir+'elements_volumn.txt')
        self.CENTERS = loadtxt(self.ansys_solver.workdir+'elements_centers.txt')
        self.flag = flag
        if flag==1:
            self.symmetric_pairs = self.find_symmetric_elements01(tolerance=0.01)
        elif flag==2:
            self.symmetric_pairs = self.find_symmetric_elements02(tolerance=0.2)
        elif flag==3:
            self.symmetric_quadruples = self.find_symmetric_elements03(tolerance=0.01)


    def get_distance_table(self, rmin):  
        """
        本函数作用是找到邻近单元并输出距离矩阵（稀疏）
        return：距离矩阵
        """
        length = self.CENTERS.shape[0]
        centers = self.CENTERS[:, 1:]
        tree = cKDTree(centers)
        row = []
        col = []
        data = []
        for i in range(length):
            indices = tree.query_ball_point(centers[i], rmin)
            for j in indices:
                if j >= i:
                    filtered_dist = rmin - np.linalg.norm(centers[i] - centers[j])
                    row.append(i)
                    col.append(j)
                    data.append(np.float32(filtered_dist))
        distance_sparse = csr_matrix((data, (row, col)), shape=(length, length), dtype=np.float32)
        distance_sparse = distance_sparse + distance_sparse.T - diags(distance_sparse.diagonal())
        return distance_sparse



    def de_checkboard1(self, dc, dv, Hf, sum_Hf):
        """
        本函数作用是进行过滤，
        输入：敏度、体积、过滤权重因子、过滤权重因子之和
        return：过滤后的dc和dv
        """
        corrected_dc = (dc @ Hf.T / sum_Hf).flatten()
        corrected_dv = (dv @ Hf.T / sum_Hf).flatten()
        return corrected_dc, corrected_dv

    def de_checkboard2(self, x, Hf, sum_Hf):
        """
        本函数作用是进行过滤，
        输入：设计变量、过滤权重因子、过滤权重因子之和
        return：过滤后的x
        """
        corrected_x = np.array(x @ Hf.T / sum_Hf).flatten()
        return corrected_x

    
    def prj(self,v,eta,beta):
        """
        本函数作用是进行投影
        输入：xTilde初始化的投影密度
        return：投影后的密度
        """
        q=(np.tanh(beta*eta)+np.tanh(beta*(v-eta)))/(np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
        return q

    def dprj(self,v, eta, beta):
    #     """
    #     本函数作用是进行投影
    #     输入：xTilde投影密度
    #     return：投影后的密度
    #     """
        w = beta * (1 - np.tanh(beta * (v - eta)) ** 2)/(np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
        return w

    def polar_to_cartesian(self, polar_centers):
        """
        本函数作用是将极坐标转换为笛卡尔直角坐标
        输入：极坐标
        return：直角坐标
        """
        r = polar_centers[:, 0]
        theta = polar_centers[:, 2]
        phi = polar_centers[:, 1]
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        z = r * np.cos(theta_rad)
        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        cartesian_centers = np.column_stack((x, y, z))
        return cartesian_centers
    
    def cartesian_to_polar(self, centers):
        """
        本函数作用是将笛卡尔直角坐标转换为极坐标
        输入：直角坐标
        return：极坐标
        """
        r = np.sqrt(np.sum(centers**2, axis=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.arccos(np.clip(centers[:, 2] / r, -1.0, 1.0))
        phi = np.arctan2(centers[:, 1], centers[:, 0])
        theta_degrees = np.degrees(theta)
        phi_degrees = np.degrees(phi)
        phi_degrees = np.mod(phi_degrees, 360)
        polar_centers = np.column_stack((r, phi_degrees, theta_degrees))
        return polar_centers

    def find_symmetric_elements01(self, tolerance):
        """
        本函数作用是找到关于z轴120度对称的单元对
        输入：容差
        return：对称单元对
        """
        centers = self.CENTERS[:, 1:]
        ids = self.CENTERS[:, 0].astype(int) 
        element_centers = np.column_stack((centers[:, 0], centers[:, 1], centers[:, 2]))
        polar_centers = self.cartesian_to_polar(element_centers)
        rotated_centers = np.copy(polar_centers)
        index_120_240 = (120 <= rotated_centers[:, 1]) & (rotated_centers[:, 1] < 240)
        index_240_360 = (240 <= rotated_centers[:, 1]) & (rotated_centers[:, 1] < 360)
        rotated_centers[index_120_240, 1] -= 120
        rotated_centers[index_240_360, 1] -= 240
        cartesian_centers = self.polar_to_cartesian(rotated_centers)
        tree = cKDTree(cartesian_centers)
        neighborhood_dict = {}
        for idx, point in enumerate(cartesian_centers):
            neighbors_idx = tree.query_ball_point(point, tolerance)
            neighbors_ids = sorted(ids[neighbors_idx])
            neighborhood_dict[ids[idx]] = tuple(neighbors_ids)
        unique_neighborhoods = []
        seen = set()
        for point_id, neighbor_ids in neighborhood_dict.items():
            unique_neighbors = sorted(set((point_id,) + neighbor_ids))
            full_tuple = tuple(unique_neighbors)
            if full_tuple not in seen:
                seen.add(full_tuple)
                unique_neighborhoods.append(full_tuple)
        print(unique_neighborhoods)
        return unique_neighborhoods


    def find_symmetric_elements02(self, tolerance):
        """
        本函数作用是找到关于平面对称的单元对
        输入：容差
        return：对称单元对
        """
        centers = self.CENTERS[:, 1:]
        ids = self.CENTERS[:, 0].astype(int) 
        processed_centers = np.column_stack((np.abs(centers[:, 0]), centers[:, 1], centers[:, 2]))
        
        tree = cKDTree(processed_centers)
        symmetric_pairs = []
        checked = set()
        for i in range(len(processed_centers)):
            indices = tree.query_ball_point(processed_centers[i], tolerance)
            for j in indices:
                if j > i:
                    distance = np.linalg.norm(processed_centers[i] - processed_centers[j])
                    if distance < tolerance and (i, j) not in checked:
                        symmetric_pairs.append((ids[i], ids[j])) 
                        checked.add((i, j))
        print("symmetric_pairs:",symmetric_pairs)
        print("Number of symmetric pairs:", len(symmetric_pairs))
        return symmetric_pairs
    

    def find_symmetric_elements03(self, tolerance):
        """
        本函数作用是xoz、yoz双平面对称；y=x
        输入：敏度
        return：对称单元对
        """
        centers = self.CENTERS[:, 1:]
        ids = self.CENTERS[:, 0].astype(int)
        processed_centers = np.column_stack((np.abs(centers[:, 0]), np.abs(centers[:, 1]), centers[:, 2]))
        condition = processed_centers[:, 1] > processed_centers[:, 0]
        indices_to_swap = np.where(condition)[0]
        if indices_to_swap.size > 0:  
            processed_centers[indices_to_swap, 0], processed_centers[indices_to_swap, 1] = \
                processed_centers[indices_to_swap, 1], processed_centers[indices_to_swap, 0]
        tree = cKDTree(processed_centers)
        symmetric_groups = []
        checked = set()
        for i in range(len(processed_centers)):
            if i not in checked:
                indices = tree.query_ball_point(processed_centers[i], tolerance)
                group = [ids[j] for j in indices]
                symmetric_groups.append(group)
                checked.update(indices)
        return symmetric_groups




    def enforce_symmetry01(self, dc):
        """
        本函数作用是将120对称单元对的敏度强制相等
        输入：敏度
        return：强制相等后的敏度
        """
        new_dc = np.copy(dc)
        pairs = np.array(self.symmetric_pairs)
        for i, j,k in pairs:
            ids = self.CENTERS[:, 0].astype(int)
            indices_i = np.where(ids == i)
            indices_j = np.where(ids == j)
            indices_k = np.where(ids == k)
            avg_value = average(new_dc[indices_i] + new_dc[indices_j]+new_dc[indices_k]) 
            new_dc[indices_i] = avg_value
            new_dc[indices_j] = avg_value
            new_dc[indices_k] = avg_value
            if abs(new_dc[indices_i]-new_dc[indices_j])>0:
                print("error in pairs:",new_dc[indices_i]-new_dc[indices_j])
        return new_dc

    def enforce_symmetry02(self, dc):
        """
        本函数作用是将xoz平面单元对的敏度强制相等
        输入：敏度
        return：强制相等后的敏度
        """
        new_dc = np.copy(dc)
        pairs = np.array(self.symmetric_pairs)
        for i, j in pairs:
            ids = self.CENTERS[:, 0].astype(int)
            indices_i = np.where(ids == i)
            indices_j = np.where(ids == j)
            avg_value = average(new_dc[indices_i] + new_dc[indices_j]) 
            new_dc[indices_i] = avg_value
            new_dc[indices_j] = avg_value
            if abs(new_dc[indices_i]-new_dc[indices_j])>0:
                print("error in pairs:",new_dc[indices_i]-new_dc[indices_j])
        return new_dc
 
    def enforce_symmetry03(self, dc):
        """
        本函数作用是将yoz/xoz平面单元对的敏度强制相等
        输入：敏度
        return：强制相等后的敏度
        """
        new_corrected_dc = np.copy(dc)
        groups = self.find_symmetric_elements03()
        groups_zero_based = [np.array(group) - 1 for group in groups]  
        for group in groups_zero_based:
            avg_value = np.average(new_corrected_dc[group])
            new_corrected_dc[group] = avg_value
        return new_corrected_dc
    

    def export_vtu(self):
        """
        本函数作用是进行初步后处理，将后处理后的模型保存为vtu文件，方便后续光滑
        """
        # mapdl = launch_mapdl() 
        self.mapdl.cwd(self.ansys_solver.workdir)
        self.mapdl.resume(self.ansys_solver.result + self.ansys_solver.model_name, "db", 0, 0)
        self.mapdl.run("/FILENAME,post,1")
        self.mapdl.title("post")
        self.mapdl.esel("S", "TYPE", "", 1,2)
        self.mapdl.eplot(vtk=True)
        mesh = self.mapdl.mesh
        mesh.save(self.ansys_solver.result + self.ansys_solver.model_name + '.vtu')  
        self.mapdl.exit(force=True)



    def smooth_model(self,vtu_file_path, stl_file_path, iterations, pass_band):
        """
        本函数作用是进行进一步后处理，对vtu文件进行光滑操作，并保存为stl格式文件
        输入：输入vtu路径、输出stl路径、迭代步数（越大越光滑）、通带宽度（0~1），越小越光滑
        """
        if not os.path.exists(vtu_file_path+self.ansys_solver.model_name + '.vtu'):
            print(f"Error: The file {vtu_file_path} does not exist.")
            return
        try:
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(vtu_file_path+self.ansys_solver.model_name + '.vtu')
            reader.Update()
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputConnection(reader.GetOutputPort())
            surface_filter.Update()
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputConnection(surface_filter.GetOutputPort())
            cleaner.Update()
            triangle_filter = vtk.vtkTriangleFilter()
            triangle_filter.SetInputConnection(cleaner.GetOutputPort())
            triangle_filter.Update()
            windowed_sinc_filter = vtk.vtkWindowedSincPolyDataFilter()
            windowed_sinc_filter.SetInputConnection(triangle_filter.GetOutputPort())
            windowed_sinc_filter.SetNumberOfIterations(iterations)
            windowed_sinc_filter.SetPassBand(pass_band)
            windowed_sinc_filter.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(windowed_sinc_filter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            renderer = vtk.vtkRenderer()
            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            render_window_interactor = vtk.vtkRenderWindowInteractor()
            render_window_interactor.SetRenderWindow(render_window)
            renderer.AddActor(actor)
            renderer.SetBackground(0.1, 0.2, 0.4)  
            renderer.ResetCamera()
            render_window.Render()
            render_window_interactor.Start()
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(stl_file_path+self.ansys_solver.model_name + '.stl')
            writer.SetInputConnection(windowed_sinc_filter.GetOutputPort())
            writer.Write()
            print(f"Successfully smoothed {vtu_file_path} and saved to {stl_file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def write_sparse_matrix_to_file(self,sparse_matrix, filename):
        with open(filename, 'w') as f:
            for row, col in zip(*sparse_matrix.nonzero()):
                value = sparse_matrix[row, col]
                f.write(f"{row} {col} {value}\n")

    def simp(self,penal = 3, volfrac = 0.5, rmin = 100, maxloop = 100):
        """
        SIMP优化算法主程序
        MMA参数请参见mma求解器
        rmin：过滤半径（绝对单元大小）
        dc/dv：敏度/体积数组
        Hf/sum_Hf：过滤权重因子/过滤权重因子之和
        x：设计变量
        beta/eta:投影参数（越大投影效果越明显)/投影阈值
        beta_max：最大beta值
        SE：应变能
        con：体积比
        """
        time01 = time.time()
        Hf01= gdt.get_distance_table(rmin,self.CENTERS_path)
        time02 = time.time()
        Hf = self.get_distance_table(rmin)
        time03 = time.time()
        print("gdt_time:",time02-time01)
        print("python:",time03-time02)
        # nonzero_count = Hf.nnz
        # print(f"Non-zero elements count: {nonzero_count}")

        # if nonzero_count > 0:
        #     self.write_sparse_matrix_to_file(Hf, r'E:\work\topo_secondraydevelop\PythonC_Ansys_dev-1/Hf02.txt')
        #     self.write_sparse_matrix_to_file(Hf01, r'E:\work\topo_secondraydevelop\PythonC_Ansys_dev-1/Hf01.txt')
        # else:
        #     print("稀疏矩阵没有非零元素，无法写入文件。")
        # print(Hf01)
        # print(Hf)
        # are_equal = all(hf01 == hf for hf01, hf in zip(Hf01, Hf))
        # print(f"Are Hf01 and Hf equal? {are_equal}")
        #... existing code...
        # return None
        # return None
        # print(Hf01!=Hf)
        # print(Hf)
        if Hf.shape!= Hf01.shape:
            print("Hf 和 Hf01 的形状不相等，因此它们不相等")
        else:
            # 提取稀疏矩阵的非零元素2
            Hf_data = Hf.data
            Hf01_data = Hf01.data
            # 提取矩阵元素的整数部分
            Hf_int_data = np.trunc(Hf_data)
            Hf01_int_data = np.trunc(Hf01_data)
            # 重新构建稀疏矩阵
            Hf_int = csr_matrix((Hf_int_data, Hf.indices, Hf.indptr), shape=Hf.shape)
            Hf01_int = csr_matrix((Hf01_int_data, Hf01.indices, Hf01.indptr), shape=Hf01.shape)
            # 计算差矩阵
            diff_matrix = (Hf_int - Hf01_int).multiply(Hf_int!= Hf01_int)
            unequal_count = diff_matrix.sum()
            total_count = Hf.shape[0] * Hf.shape[1]
            equal_count = total_count - unequal_count
            unequal_ratio = unequal_count / total_count
            print(f"Hf 和 Hf01 中整数部分相等元素的数量为: {equal_count}")
            print(f"Hf 和 Hf01 中整数部分不相等元素的数量为: {unequal_count}")
            print(f"Hf 和 Hf01 中整数部分不相等元素的比例为: {unequal_ratio}")
            self.save_sparse_matrix_to_excel(Hf_int, self.ansys_solver.awd + 'Hf_int.xlsx')
            self.save_sparse_matrix_to_excel(Hf01_int, self.ansys_solver.awd + 'Hf01_int.xlsx')
        return None
        sum_Hf = np.array(Hf.sum(axis=1)).flatten()
        x = volfrac*np.ones(self.element_counts, dtype = float)
        # print(x)
        xTilde = x.copy() 
        m = 1
        n = self.element_counts
        loop = 0
        xmin = np.zeros((n,1))
        xmax = np.ones((n,1))
        xval = x[np.newaxis].T 
        move = 0.05
        eta=0.5 
        beta=1
        beta_max=32
        xPhys = x.copy()
        scale_con = 1000   
        change = 1
        mma_solver = mma.MMASolver(n, m)
        time_sum = []
        # return None
        while change > 1e-4 and loop < maxloop:
            loop = loop+1
            xold = x                         
            time02 = time.time()
            self.ansys_solver.generate_material_properties(xPhys, penal)
            self.mapdl.input(self.ansys_solver.target_dir+self.ansys_solver.get_result_file)
            SE = loadtxt(self.ansys_solver.workdir+'SENE.TXT')
            # SE = self.ansys_solver.get_SENE_data(xPhys,penal)
            time03 = time.time()
            dc = -2*penal*SE/[(x + 1e-9) for x in xPhys]
        
            V = self.V[:,1:].T.flatten()
            c = 2*np.sum(SE)
            con = (xPhys*V).sum()/V.sum()


            dc01 = dc*filter_prj.dprj(xTilde, eta, beta)
            dv01 = V*filter_prj.dprj(xTilde, eta, beta)/V.sum()

            dc = dc*self.dprj(xTilde,eta,beta)
            dv = V*self.dprj(xTilde,eta,beta)/V.sum()

            dc01,dv01 = filter_prj.de_checkboard1(dc, dv, Hf, sum_Hf)
            dc, dv = self.de_checkboard1(dc, dv, Hf, sum_Hf)
            if self.flag==1:
                dc = self.enforce_symmetry01(dc)
            elif self.flag==2:
                dc = self.enforce_symmetry02(dc)
            elif self.flag==3:
                dc = self.enforce_symmetry03(dc)
            if loop == 1:
                scale_obj = 1000/abs(c)
            xval = x.copy()[np.newaxis].T            
            df0dx = scale_obj*array(dc, dtype = float)[np.newaxis].T
            fval = scale_con*np.array([con/volfrac-1])
            dfdx = scale_con*array(dv, dtype = float)/volfrac
            xmax = np.minimum(1,xval+move)
            xmin = np.maximum(0,xval-move)
            xmma = mma_solver.Update(xval, df0dx, fval, dfdx, xmin, xmax)
            time_sum.append(time03-time02)
            x = xmma.copy().flatten()
            xTilde01 = filter_prj.de_checkboard2(x, Hf, sum_Hf)
            xTilde = self.de_checkboard2(x, Hf, sum_Hf)
            xTilde = array(xTilde, dtype = float)
            xPhys = self.prj(xTilde,eta,beta).flatten()
            xPhys01 = filter_prj.prj(xTilde, eta, beta)
            change = max(abs(x-xold))
            obj_file_path = self.ansys_solver.result + 'Obj.txt'
            con_file_path = self.ansys_solver.result + 'Con.txt'
            if loop == 1:
                for file_path in [obj_file_path, con_file_path]:
                    if os.path.exists(file_path):
                        os.remove(file_path)  
            with open(obj_file_path, 'a') as obj_file:
                obj_file.write(f"{c}\n")
            with open(con_file_path, 'a') as con_file:
                con_file.write(f"{con}\n")
            print(f"    loop: {loop}    obj: {scale_obj*c:.3f}  con: {con:.4f}  change: {change:.4f}    bata: {beta}"  )
            if (loop) % 50 == 0:
                beta = np.minimum(2 * beta, beta_max)
                print(f'Parameter beta increased to {beta}.') 
            if loop==maxloop:
                break
        time_avg = sum(time_sum)/len(time_sum)
        print("time_avg",time_avg)
        time04 = time.time()
        # print("runtime",time04-time01)
        self.mapdl.input(self.ansys_solver.target_dir+self.ansys_solver.new_post_file)
        # self.ansys_solver.post()
        self.export_vtu()
        self.smooth_model(self.ansys_solver.result, self.ansys_solver.result, iterations=300, pass_band=0.01)
        return x
#单元测试
if __name__=='__main__':
    simp_solver = SIMP(flag=0)
    x = simp_solver.simp()
