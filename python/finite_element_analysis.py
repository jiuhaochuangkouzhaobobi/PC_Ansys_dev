import subprocess
import numpy as np
from numpy import *
import re
import os
import tkinter as tk
from tkinter import filedialog
import shutil
class FEA(object):
    """
    ANSYS 求解器

    Parameters
    ----------
    各路径的说明:
    cwd:存储ANSYS_APDL脚本，与ANSYS日志文件
    awd: 此目录为ANSYS的工作目录，所有需要在APDL中读入或输入的文件都在此目录下
    ANSYS_APDL读取文件与写入文件命名规范:

    读取文件:
    material.txt:材料文件,对于各向同性材料：Nx2，第一列杨氏模量，第二列泊松比，N为单元数
    写入文件:
    elements_nodes_counts.txt:单元数与节点数，1x2第一行单元数，第二行节点数
    elements_stiffness.out:未经处理的单元刚度矩阵文件
    elements_nodes.txt: 各单元对应的节点编号,Nx(k+1),N为单元数，第一列单元编号，剩下k列为节点编号
    elements_centers.txt: 各单元的中心坐标，Nx4，N为单元数，第一列为单元编号，其余三列为中心坐标值，（x,y,z)
    elements_volumn.txt: 各单元的体积，Nx2，N为单元数，第一列单元编号，第二列体积
    nodal_solution_u.txt: 节点位移，3列分别表示X,Y,Z方向位移
    nodal_solution_stress.txt: 节点应力，Vonmiss应力，一列，行数等于节点数
    nodal_solution_strain.txt: 节点应变，一列，行数等于节点数
    """
    def __init__(self):
        # 输入文件(APDL)和输出文件都将在cwd目录中，而ANSYS需要的其他输入数据或输出数据的路径，将由ANSYS的APDL指定
        #---------------PC路径--------------------
        self.awd = r'E:\work\topo_secondraydevelop\PythonC_Ansys_dev-1/'
        self.elemnt_type = "SOLID185"#平面PLANE182 实体SOLID185 壳单元SHELL181
        self.E = 1
        self.ansys_path = r"D:\Program Files\ANSYS Inc\v241\ansys\bin\winx64\MAPDL.exe"
        self.model_name,self.workdir=self.select_file_and_copy(self.awd)
        self.result = self.awd + self.model_name+ "/result/"
        self.target_dir,self.get_meshmodel_file,self.get_result_file,self.new_post_file=self.change_path()
        if not os.path.exists(self.result):
            os.makedirs(self.result)
        self.meshdata_cmd = [self.ansys_path, '-b', '-i',
               self.target_dir+self.get_meshmodel_file, '-o', self.workdir + 'get_meshmodel_data.out']
        # self.result_cmd = [self.ansys_path, '-b', '-i',
        #        self.target_dir+self.get_result_file, '-o', self.workdir + 'get_result_data.out']
        self.result_cmd = [self.ansys_path, '-b', '-i',
               self.target_dir+self.get_result_file, '-o', self.workdir + 'get_result_data.out']
        self.post_cmd = [self.ansys_path, '-b', '-i',
               self.target_dir+self.new_post_file, '-o', self.workdir + 'post.out']
        # subprocess.call(self.meshdata_cmd)
        # self.element_counts,self.node_counts = self.get_counts(self.workdir+'elements_nodes_counts.txt')

            


    def select_file_and_copy(self,awd):
        """选择文件并复制到指定目录"""
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        file_path = filedialog.askopenfilename(title="选择一个文件")  # 选择文件
        
        if file_path:  # 如果用户选择了文件
            # 获取文件名
            model_name = os.path.basename(file_path)
            new_model_name = model_name[:-4]  # 去掉最后四个字符.cdb
            workdir = awd + new_model_name + "/temp/"
            if not os.path.exists(workdir):
                os.makedirs(workdir)  # 创建目标目录（如果不存在）
            # 定义目标路径
            destination_path = os.path.join(workdir, model_name)
            
            # 复制文件
            shutil.copy(file_path, destination_path)
        return new_model_name,workdir
        




    def change_path(self):
        get_meshmodel_file = self.awd+'get_meshmodel_data.txt' 
        get_result_file = self.awd+'get_result_data.txt' 
        post_file = self.awd+'post.txt' 
        
        new_content01 = r"/CWD," +"'" + self.awd +  self.model_name + "/temp'" 
        new_content02 = "/INPUT,'" + self.model_name + "','cdb'," +"'" + self.awd + self.model_name + "/temp/',, 0"
        new_content03 = r"*VREAD,X(1,1),"+ self.awd + self.model_name + r"/temp\x,txt,,JIK,1,ET1SUM"
        new_content04 = "SAVE," +"'" + self.model_name+ "'" +",'db'," + "'" + self.result + "'"
        new_content05 = "ET,10000," + self.elemnt_type
        

        work_dir = self.awd + self.model_name + '/temp' 
        new_get_meshmodel_file = 'get_meshmodel_data_' + self.model_name + '.txt'
        new_get_result_file_file = 'get_result_data_' + self.model_name + '.txt'  
        new_post_file = 'post_' + self.model_name + '.txt'  
        target_dir = self.awd + self.model_name+'/'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        #mesh
        with open(get_meshmodel_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines[1] = new_content01 + '\n'
        lines[2] = new_content02 + '\n'
        new_meshmodel_path = os.path.join(target_dir, new_get_meshmodel_file)
        with open(new_meshmodel_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        #result
        with open(get_result_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines[1] = new_content01 + '\n'
        lines[2] = new_content02 + '\n'
        new_result_path = os.path.join(target_dir, new_get_result_file_file)
        with open(new_result_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        #post
        with open(post_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines[1] = new_content01 + '\n'
        lines[2] = new_content02 + '\n'
        lines[36] = new_content03 + '\n'
        lines[95] = new_content04 + '\n'
        lines[13] = new_content05 + '\n'
        new_post_path = os.path.join(target_dir, new_post_file)
        with open(new_post_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return target_dir,new_get_meshmodel_file,new_get_result_file_file,new_post_file


    def get_counts(self,element_nodes_file):
        """
        获取单元数和节点数

        Parameters
        ----------
        element_nodes_file:存储单元数和节点数的文件

        Returns
        ----------
        返回单元数和节点数
        """
        counts = loadtxt(element_nodes_file,dtype = int)
        return counts[0],counts[1]


    def generate_material_properties(self,x,penal):
        """
        将OC方法获得的x生成新的材料文件，对于各向异性的点阵材料而言，其材料属性文件将由子类实现

        Parameters
        ----------
        x : 单元密度
        penal : 惩罚因子

        Returns
        ----------
        将生成的材料文件存入material.txt
        """
        x = array(x,dtype = float)
        nu = 0.3 * np.ones((self.element_counts))
        ex = x**penal+1e-9
        material = np.array([nu, ex*self.E]).T
        with open(self.workdir + 'x.txt', 'w') as file:
            for i in x:
                    file.write(f"{i:.{2}f}\n")
        np.savetxt(self.workdir+"material.txt", material, fmt=' %-.10f', newline='\n')

    def get_meshmodel_data(self):
        """
        获取有限元模型相关数据,这些数据在迭代计算中属于不变量，只需单独调用该函数一次

        Parameters
        ----------
        dim:单元刚度矩阵的维度

        Returns
        ----------
        K:单元刚度矩阵集合
        element_attributes:单元对应的节点编号
        CENTERS:单元的中心坐标
        V:单元体积
        """        
        CENTERS = loadtxt(self.workdir+'elements_centers.txt')
        V = loadtxt(self.workdir+'elements_volumn.txt')
        return self.workdir+'elements_centers.txt',V,CENTERS
    
    
    def get_SENE_data(self,x,penal):
        """
        更新设计变量
        更新应变能，进行有限元分析并获取结果数据文件
        """
        self.generate_material_properties(x, penal)
        subprocess.call(self.result_cmd)
        SE = loadtxt(self.workdir+'SENE.TXT')
        return SE  
    
    def post(self):
        """
        更新设计变量
        更新应变能，进行有限元分析并获取结果数据文件
        """
        subprocess.call(self.post_cmd)
        return None  
    


#单元测试
if __name__=='__main__':
    ansys_solver = FEA()




