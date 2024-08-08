import numpy as np
import sympy as sp

class FEM:
    def __init__(self, nodes, elements, E=1e6, nu=0.3, thickness=1):
        ## Material Properties ##
        self.E= E
        self.nu = nu

        ## Mesh Properties ##
        self.nodes = nodes
        self.elements = elements

        ## Matrices ##
        self.D = None
        self.B = None
        self.K_e = None
        self.K_global = None
        self.N = None
        self.N_evaluated = None
        self.J = None
        self.J_det = None
        self.G = None

        ## thickness of element ##
        self.thickness = thickness
    
    def D_matrix(self): # plane stress
        """
        Creates the material property D matrix

        Parameters:
        self

        out:
        D matrix --> self.D
        """
        coef = self.E/(1-self.nu**2)
        self.D = np.array([[self.E*coef, self.nu*self.E*coef, 0.0]
                           ,[self.nu*self.E*coef, self.E*coef, 0.0]
                           ,[0.0, 0.0, self.E/(2*(1+self.nu))]])
    
    def N_matrix_evaluation(self,xi_val,eta_val):

        # Define symbolic variables
        xi, eta = sp.symbols('xi eta')
        
        # Define shape functions
        N1 = 0.25 * (1 - xi) * (1 - eta)
        N2 = 0.25 * (1 + xi) * (1 - eta)
        N3 = 0.25 * (1 + xi) * (1 + eta)
        N4 = 0.25 * (1 - xi) * (1 + eta)

        # Correctly define the N matrix with symbolic expressions
        self.N_evaluated = np.array([
            [N1, sp.sympify(0), N2, sp.sympify(0), N3, sp.sympify(0), N4, sp.sympify(0)],
            [sp.sympify(0), N1, sp.sympify(0), N2, sp.sympify(0), N3, sp.sympify(0), N4]
        ])
        
        # Substitute values into the N matrix
        self.N_evaluated = np.array([
            [n.subs({'xi': xi_val, 'eta': eta_val}) for n in row]
            for row in self.N
        ])
        
        # Convert the resulting matrix to float values
        self.N_evaluated = self.N.astype(np.float64)
    
    def N_matrix(self):

        # Define symbolic variables
        xi, eta = sp.symbols('xi eta')
        
        # Define shape functions
        N1 = 0.25 * (1 - xi) * (1 - eta)
        N2 = 0.25 * (1 + xi) * (1 - eta)
        N3 = 0.25 * (1 + xi) * (1 + eta)
        N4 = 0.25 * (1 - xi) * (1 + eta)

        # Correctly define the N matrix with symbolic expressions
        self.N = np.array([
            [N1, sp.sympify(0), N2, sp.sympify(0), N3, sp.sympify(0), N4, sp.sympify(0)],
            [sp.sympify(0), N1, sp.sympify(0), N2, sp.sympify(0), N3, sp.sympify(0), N4]
        ])

        print(f'N: {self.N}')
    
    def derivative_matrix(self):
        xi, eta = sp.symbols('xi eta')
        dN1dxi = -(1-eta)*0.25
        dN2dxi= (1-eta)*0.25
        dN3dxi = (1+eta)*0.25
        dN4dxi = -(1+eta)*0.25
        
        dN1deta = -(1-xi)*0.25
        dN2deta = -(1+xi)*0.25
        dN3deta = (1+xi)*0.25
        dN4deta = (1-xi)*0.25

        return np.array([[dN1dxi,dN2dxi,dN3dxi,dN4dxi],
                        [dN1deta,dN2deta,dN3deta,dN4deta]])
        
    
    def J_matrix(self,element):
        # get nodes
        nodes = self.nodes[self.elements[element],:]
        nodes[[2,3]] = nodes[[3,2]]

        # calculate the jacobian matrix
        self.J = self.derivative_matrix() @ nodes
        print(f'J: {self.J}')
        
    def determinant_J(self,element):
        self.J_matrix(element)
        self.J_det = self.J[0,0]*self.J[1,1]-self.J[0,1]*self.J[1,0]

        print(f'det J: {self.J_det}')

    def G_matrix(self,element):
        self.determinant_J(element)
        self.G  = (1/self.J_det)*np.array([[self.J[1,1],-self.J[0,1]]
                            ,[-self.J[1,0],self.J[0,0]]])
        print(f'G: {self.G}')

    def B_matrix(self,element):
        d = self.derivative_matrix() # get derivative matrix
        # augment d
        d_aug = np.array([[d[0,0],0,d[0,1],0,d[0,2],0,d[0,3],0],[0,d[1,0],0,d[1,1],0,d[1,2],0,d[1,3]]])
        self.G_matrix(element) # calculate G

        d_trans_flat = (self.G @ d).T.flatten()

        self.B = np.array([[1,0],[0,1],[0,0]]) @ self.G @ d_aug # find B without the last row
        self.B[2,:] = d_trans_flat # 
    
    def K_local_matrix(self,element):
        self.B_matrix(element)
        self.D_matrix()
        self.determinant_J(element)
        integrand = self.B.T @ self.D @ self.B * self.J_det * self.thickness
        
        print(integrand.shape)



        
    







