import numpy as np
import pandas as pd
from itertools import combinations
from itertools import chain


def generate_partition(f_part):  #input a dict
    setnum = len(f_part)  #子集數
    num_set = [i for i in range(setnum)]  #子集數list
    cardlist = [len(i) for i in f_part.values()]  #各子集cardinality
    f_list = [i for i in f_part.values()]  #list of list
    all_subsets = []
    for r in range(setnum+1):
        for s in combinations(num_set, r):
            tmp = []
            for t in s:
                tmp.extend(f_list[t])
            all_subsets.append(np.array(sorted(tmp)))
    # Create a hash table to store the index of each subset
    subset_index_table = {}
    for index, value in enumerate(all_subsets):
        subset_index_table[tuple(value)] = index
    return all_subsets, subset_index_table, len(all_subsets)


class Hypercube_wpart:
    '''
    A class to create a hypercube object which stores values of vertices
    '''    
    #輸入維度
    def __init__(self, partition):   #input a dict
        self.f_part = partition
        self.n_dim = len(self.f_part)
        self.elem_count = sum([len(i) for i in self.f_part.values()])
        # vertex_values is a dictionary to store the value of each vertex.
        # Because np.array is not hashable, we use tuple to represent the vertex.
        self.vertex_values = {}
        self.vertices, self.vertex_index, self.vertex_num = generate_partition(self.f_part)  #vertex 上的value一次考慮整個subset
        self.edges, self.edge_num = self.build_edges()
        self.differential_matrix = None
        self.weight_matrix = None
        self.generate_min_l2_norm_matrix()
    
    def build_edges(self):
        num_set = [i for i in range(self.n_dim)]  #子集數list
        s_set = set(num_set)  #轉集合
        cardlist = [len(i) for i in self.f_part.values()]  #各子集cardinality
        f_list = [i for i in self.f_part.values()]  #list of list
        #print(f'Receive {f_list}')
        s_subset = set(tuple(i) for i in f_list)
        edges = []
        for r in range(self.n_dim): 
            for v in combinations(num_set, r):
                v_set = set(v)
                adjunct_v = s_set - v_set
                for new_elem in adjunct_v:
                    d_set = v_set | {new_elem}
                    outlist, inlist = [],[]                    
                    for k in v_set:
                        outlist.extend(f_list[k])
                    for l in d_set:
                        inlist.extend(f_list[l])
                    edges.append(((np.array(sorted(outlist))),np.array(sorted(inlist))))
        return edges, len(edges)
    
    def get_elements(self, index):
        return tuple(self.f_part[index])

    def set_vertex_values(self, vertex_values):         #設置點值
        for v in vertex_values:                         #用鍵值來做查找
            self.vertex_values[v] = vertex_values[v]
        
    def does_edge_exist(self, v1, v2):
        if abs(len(v1)-len(v2))==1:
            interset = np.intersect1d(v1,v2)
            smaller = v1 if len(v1)<len(v2) else v2
            return True if np.array_equal(smaller, interset) else False
        else:
            return False
    
    # Establish the matrix A in the above formula: AX-Y
    def generate_differential_matrix(self):
        if self.differential_matrix is None:
            self.differential_matrix = np.zeros((self.edge_num+1, self.vertex_num))
            for i,v_pair in enumerate(self.edges):
                j = self.vertex_index[tuple(v_pair[1])]
                k = self.vertex_index[tuple(v_pair[0])]
                self.differential_matrix[i][j] = 1
                self.differential_matrix[i][k] = -1
            # Add one more equestion that x_0 = 0 into the matrix form
            self.differential_matrix[-1][0]=1
        return self.differential_matrix

    # Pre-calcuate "W=(A^T*A)^-1*A^T" for the formula "X = ((A^T*A)^-1*A^T)*Y
    def generate_min_l2_norm_matrix(self):
        matrix_A = self.generate_differential_matrix()
        matrix_A_T = np.transpose(matrix_A)
        self.weight_matrix = np.linalg.inv(matrix_A_T @ matrix_A) @ matrix_A_T

    def get_gradient_vector(self):
        gradient_vector = np.zeros(self.edge_num)
        for i,v_pair in enumerate(self.edges):
            gradient_vector[i] = self.vertex_values[tuple(v_pair[1])]-self.vertex_values[tuple(v_pair[0])]    
        return gradient_vector      
        
    def get_partial_gradient_vector(self,subset_i):  #feature->subset->allow input int or tuple
        if isinstance(subset_i, int):
            feature_i = self.get_elements(subset_i)
        else:
            feature_i = []            
            for i in subset_i:
                feature_i.append(self.get_elements(i))
            feature_i = tuple(chain.from_iterable(feature_i))
        partial_gradient_vector = np.zeros(self.edge_num)
        for i,v_pair in enumerate(self.edges):
            if (not set(feature_i).issubset(set(v_pair[0]))) and (set(feature_i).issubset(set(v_pair[1]))):
                partial_gradient_vector[i] = self.vertex_values[tuple(v_pair[1])]-self.vertex_values[tuple(v_pair[0])]    
        return partial_gradient_vector
    
    def resolve_vi(self, subset_i, phi_0=0):  #feature->subset
        pgd = self.get_partial_gradient_vector(subset_i)
        # Append equation x_0=0 at the end of partial gradient vector.
        pgd = np.append(pgd, phi_0)
        vi = self.weight_matrix @ pgd
        # Reconstruct the vertex values
        new_vertices = {}
        for i,v in enumerate(self.vertices):
            new_vertices[tuple(v)] = vi[i]
        return vi, new_vertices


def make_cube_list(vertex_df, F_P):  #input vertex dataframe and return a list of cubes for instances
    subsets, _, _ = generate_partition(F_P)
    cubelist = []
    for idx in range(vertex_df.shape[0]):
        tp = Hypercube_wpart(F_P)
        vertices = {}
        values = vertex_df.iloc[idx].tolist()
        for v in subsets:
            vertices[tuple(v)] = values.pop(0)
            tp.set_vertex_values(vertices)
        cubelist.append(tp)
    return cubelist


def single_subset_norm(cubelist,F_P):
    #get partial gradient vector and residual for n*subsets
    rows, cols = len(cubelist), len(F_P)
    pg_m = [[None for i in range(cols)] for j in range(rows)]  #partial gradient vector matrix in 3D
    residual_m = [[None for i in range(cols)] for j in range(rows)]  #residual matrix in 3D
    for i in range(rows):
        for j in range(cols):
            pgd = cubelist[i].get_partial_gradient_vector(j)  #partial gradient of instance i of subset j 
            pg_m[i][j] = pgd
            vi, new_vs = cubelist[i].resolve_vi(j)
            h1 = Hypercube_wpart(F_P)
            h1.set_vertex_values(new_vs)
            dvi = h1.get_gradient_vector()
            residual_m[i][j] = dvi-pgd
    #l2 norm for single subset    
    pgd_l2 = [[float(np.linalg.norm(pg_m[i][j])) for j in range(cols)] for i in range(rows)]
    r_l2 = [[float(np.linalg.norm(residual_m[i][j])) for j in range(cols)] for i in range(rows)]
    scaled_norm = [[float(r_l2[i][j] / pgd_l2[i][j]) for j in range(cols)] for i in range(rows)]
    scn_df = pd.DataFrame(scaled_norm)
    pgd_df = pd.DataFrame(pgd_l2)
    r_df = pd.DataFrame(r_l2)
    l2_mean = scn_df.mean()
    print(l2_mean)
    return pgd_df, r_df, scn_df


def two_subset_norm(cubelist,F_P):
    #get partial gradient vector and residual for n*length2subsets
    length = 2
    rows = len(cubelist)
    subset_num = len(F_P)
    items = [i for i in range(subset_num)]
    combo = [i for i in combinations(items,length)]
    cols = len(combo)
    pg_m = [[None for i in range(cols)] for j in range(rows)]  #partial gradient vector matrix in 3D
    residual_m = [[None for i in range(cols)] for j in range(rows)]  #residual matrix in 3D
    for i in range(rows):
        for j in range(cols):
            pg = cubelist[i].get_partial_gradient_vector(combo[j])  #partial gradient of instance i of subset j 
            pg_m[i][j] = pg
            vi, new_vs = cubelist[i].resolve_vi(combo[j])
            h1 = Hypercube_wpart(F_P)
            h1.set_vertex_values(new_vs)
            dvi = h1.get_gradient_vector()
            residual_m[i][j] = dvi-pg    
    #l2 norm 
    pgd_l2 = [[float(np.linalg.norm(pg_m[i][j])) for j in range(cols)] for i in range(rows)]
    r_l2 = [[float(np.linalg.norm(residual_m[i][j])) for j in range(cols)] for i in range(rows)]
    scaled_norm = [[float(r_l2[i][j] / pgd_l2[i][j]) for j in range(cols)] for i in range(rows)]
    scn_df = pd.DataFrame(scaled_norm, columns=combo)
    pgd_df = pd.DataFrame(pgd_l2,columns=combo)
    r_df = pd.DataFrame(r_l2)
    l2_mean = scn_df.mean()
    print(l2_mean)
    return pgd_df, r_df, scn_df
    











