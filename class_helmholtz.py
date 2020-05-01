#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:14:07 2020

@author: chihin
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from math import cos

class Helmholtz:
    # Intialises class with No. of elements (N_el) and polynomial order (p)
    def __init__(self, N_el, p):
        
        # Initialises no. of elements
        self.N_el = N_el
        
        # Initialises Domain length
        self.L = 1
        
        # Initialises element size h
        self.dx = self.L/self.N_el
        print("dx: " + str(self.dx))
        
        # Total number of elemental DOF
        self.n_eof = N_el*(p+1)
        
        # Total number of DOFs
        self.n_dof = N_el*p + 1
        print("Total number of DOFs: " + str(self.n_dof))
        
        # Initialises lambda and sigma
        self.lmda = 1.
        self.sigma = 1.
        self.alpha = 1.
        self.beta = 0.
        
        # Initialises Polynomial Order
        # p = 1 -> Linear, p = 2 -> Quadratic
        if (p == 1):
            print("Linear Shape Functions Selected")
        elif (p == 2):
            print("Quadratic Shape Functions Selected")
        else:
            raise Exception("Invalid polynomial chose. Please select p=1, 2 for linear, quadratic shape functions")
        self.p = p
        
        # Define Shape Functions
        
        
    def init_gauss_lobatto(self):
        """
        Implements Numerical Integration using Gauss Lobatto
        Stores the value of phi at quadrature points -1, 0, 1
        i -> phi_i, j -> phi_i evaluated at quadrature point j
        """
        phi = np.zeros((self.p+1, 3))
        dphi = np.zeros((self.p+1, 3))
        if (self.p == 1):
            # phi values evaluated at quadrature points
            phi[0,0] = 1.0
            phi[0,1] = 0.5
            phi[0,2] = 0.0
            phi[1,0] = 0.0
            phi[1,1] = 0.5
            phi[1,2] = 1.0
            # d(phi)/d(psi) values evaluted at quadrature points
            dphi[0,0] = -0.5
            dphi[0,1] = -0.5
            dphi[0,2] = -0.5
            dphi[1,0] = 0.5
            dphi[1,1] = 0.5
            dphi[1,2] = 0.5
            
        elif (self.p == 2):
            # phi values evaluated at quadrature points
            phi[0,0] = 1.0
            phi[0,1] = 0.0
            phi[0,2] = 0.0
            phi[1,0] = 0.0
            phi[1,1] = 1.0
            phi[1,2] = 0.0
            phi[2,0] = 0.0
            phi[2,1] = 0.0
            phi[2,2] = 1.0
            # d(phi)/d(psi) values evaluted at quadrature points
            dphi[0,0] = -3/2
            dphi[0,1] = -1/2
            dphi[0,2] = 1/2
            dphi[1,0] = 2.0
            dphi[1,1] = 0.0
            dphi[1,2] = -2.0
            dphi[2,0] = -1/2
            dphi[2,1] = 1/2
            dphi[2,2] = 3/2
            
        self.phi = phi
        self.dphi = dphi
     
    # Constructs Elemental Matrices and return them as output
    # [MElem, LElem, FElem]  = case.ConstructElem()
    
    def ConstructElem(self):
        """
        Constructing elemental matrices
        """
        
        MElem = np.zeros((self.p+1,self.p+1))
        LElem = np.zeros((self.p+1,self.p+1))
        FElem = np.zeros((self.N_el,self.p+1))

        # Assigning Quadrature Weights
        weights = np.zeros((3,1))
        weights[0] = 1/3
        weights[1] = 4/3
        weights[2] = 1/3
        
        
        # Defining source function
        source = lambda x : -(4*self.sigma*pi**2 + self.lmda)*cos(2*pi*x)
        
        for i in range(self.p+1):
            for j in range(self.p+1):
                for k in range(3):
                    MElem[i,j] += self.phi[i,k] * self.phi[j,k] * self.dx/2.0 * weights[k]
                    LElem[i,j] += self.dphi[i,k] * self.dphi[j,k] * self.dx/2.0 * (2.0/self.dx)**2 * weights[k]
        
        
        for e in range(self.N_el):
            for j in range(self.p+1):
                for k in range(3):
                    FElem[e,j] += self.phi[j,k] * weights[k] * self.dx / 2.0 * source(e*self.dx + j*self.dx/self.p)
        
        return [MElem, LElem, FElem] 

    # Builds Global Matrices with elemental matrices as input and return them as output
    # [MG, LG, FG] = case.ConstructGlob(MElem, LElem, FElem):
    def ConstructGlob(self, MElem, LElem, FElem):
        
        MG = np.zeros((self.n_dof, self.n_dof))
        LG = np.zeros((self.n_dof, self.n_dof))
        FG = np.zeros(self.n_dof)

        # Defines connectivity map
        connect = np.zeros(self.p+1, dtype = int)
        
        # Intialising Connectivity map
        for i in range(self.p+1):
            connect[i] = i
            
        # Constructing Global Mass/Laplacian Matrices    
        for e in range(self.N_el):
            for i in range(self.p+1):
                for j in range(self.p+1):
                    MG[connect[i], connect[j]] = MG[connect[i],connect[j]] + MElem[i,j]
                    LG[connect[i], connect[j]] = LG[connect[i],connect[j]] + LElem[i,j]
            # Updating connectivity map
            connect = connect + self.p
        
        # Intialising Connectivity map
        for i in range(self.p+1):
            connect[i] = i
        
        # Constructing Global Source Vector
        for e in range(self.N_el):
            for i in range(self.p + 1):
                FG[connect[i]] = FG[connect[i]] + FElem[e,i]
            connect = connect + self.p
        
        return [MG, LG, -FG]
    
    # Solve the system of linear equations given LG, MG and MG
    # Method of large numbers is used here for implementing B.Cs 
    def Solve(self, MG, LG, FG):
        
        # Setting up Stiff Matrix and RHS for the homogeneous problem
        stiff = LG + self.lmda*MG
        rhs = FG
        
        # Using Method of Large Number
        large = 10e15
        stiff[0,0] = large
        rhs[0] = large*self.alpha
        
        # Solving with matrix inversion
        u = np.zeros(self.n_dof)
        u = np.linalg.inv(stiff)@rhs
        
        # Setup grid        
        x = np.linspace(0,1,self.n_dof)
   
        # Computing Exact Solution
        u_ex = np.zeros(len(x))
        for i in range(len(x)):
            u_ex[i] = cos(2*pi*x[i])
        
        # Computing the L2-norm
        l2 = 0.0
        for i in range(len(x)):
            l2 += (u_ex[i] - u[i])**2
        l2 = math.sqrt(l2/(len(x)))
        print("L2 Norm: " + str(l2) + '\n')
        
        # Computing error distribution
        err = np.zeros(self.n_dof)
        err = np.abs(u_ex-u)
        
        self.l2 = l2
        self.err = err
        self.x = x
        self.u = u
        self.u_ex = u_ex
    
    # Plotting the solution for single case
    def PlotOnce(self):
        
        plt.close()
        if (self.p == 1):
            title_text = ("Linear Elements, $N_{el} = $" 
                          + str(self.N_el) 
                          + '\n' + 'dx: ' + str(self.dx) 
                          + '   L2-$norm$: ' +str("{:.3f}".format(self.l2)))
        elif(self.p == 2):
            title_text = ("Quadratic Elements, $N_{el} = $" 
                          + str(self.N_el) + '\n' 
                          + 'dx: ' + str(self.dx) 
                          + '   L2-$norm$: ' +str("{:.3f}".format(self.l2)))
            
        plt.figure()
        plt.plot(self.x, self.u, '-o', label = "FEM Solution")
        plt.plot(self.x, self.u_ex, 'r--', linewidth = '1.6', label = "Exact Solution")
        plt.title(title_text)
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.grid()
        
    
    # Error Analysis for N_el = [5, 10, 20, 50, 100] in q4b
    # Note that the single case have to be setup first
    def ErrorAnalysis(self):
        
        # Declaring an array of elements to be solved
        NN_el = np.array([5, 10, 20, 50, 100])
        lin_err = np.zeros(len(NN_el))
        quad_err = np.zeros(len(NN_el))
        
        # Computing L2-norms for linear elements
        for i in range(len(NN_el)):
            linear = Helmholtz(NN_el[i],1)
    
            # Initialising phi/dphi values at Gauss-Lobatto quadrature points
            linear.init_gauss_lobatto()
            
            # Building Elemental Mass, Laplacian and Source matrices/vectors
            [MElem, LElem, FElem] = linear.ConstructElem()
            
            # Constructing Global Mass, Laplacian and Source matrices/vectors
            [MG, LG, FG] = linear.ConstructGlob(MElem, LElem, FElem)
            
            # Assemble Stiffness matrix and solving Ax = b
            # Dirichlet B.Cs are solved using "Method of Large Number"
            linear.Solve(MG, LG, FG)
            
            lin_err[i] = linear.l2
            
        # Computing L2-norms for quadratic elements
        for i in range(len(NN_el)):
            quad = Helmholtz(NN_el[i],2)
    
            # Initialising phi/dphi values at Gauss-Lobatto quadrature points
            quad.init_gauss_lobatto()
            
            # Building Elemental Mass, Laplacian and Source matrices/vectors
            [MElem, LElem, FElem] = quad.ConstructElem()
            
            # Constructing Global Mass, Laplacian and Source matrices/vectors
            [MG, LG, FG] = quad.ConstructGlob(MElem, LElem, FElem)
            
            # Assemble Stiffness matrix and solving Ax = b
            # Dirichlet B.Cs are solved using "Method of Large Number"
            quad.Solve(MG, LG, FG)
            
            quad_err[i] = quad.l2
            
        
        # Getting slope and y-intercept of loglog graph
        m_lin, c_lin = np.polyfit(np.log(1/NN_el), np.log(lin_err), 1)   
        m_quad, c_quad = np.polyfit(np.log(1/NN_el), np.log(quad_err), 1)
        
        # Plotting Error plots for both linear and quadratic
        plt.figure()
        plt.loglog(1/NN_el, lin_err, '-o', label = 'Linear')
        plt.loglog(1/NN_el, quad_err, '-o', label = 'Quadratic')
        plt.text(0.05, 0.0025, "Gradient: " + str("{:.2f}".format(m_lin)), fontsize = 12)
        plt.text(0.05, 0.0000025, "Gradient: " + str("{:.2f}".format(m_quad)), fontsize = 12)
        plt.title("Q4b) L2-$norm$ vs Grid Size")
        plt.legend()
        plt.xlabel('Grid Size: log' + r'$\frac{1}{N_{el}}$ ', fontsize = 12)
        plt.ylabel('log(L2-$norm$) ', fontsize = 12)
        plt.grid(which='minor')
        plt.show()
        
    # Alternative Error Analysis for specified no. of elements in q4c
    # Compares the error distribution between linear and quad elements
    # Note that the single case have to be setup first
    def AltErrorAnalysis(self, N_el):
            
        # Initialising problem for both linear and quad
        lin = Helmholtz(N_el,1)
        quad = Helmholtz(N_el,2)
        
        # Initialising phi/dphi values at Gauss-Lobatto quadrature points
        lin.init_gauss_lobatto()
        quad.init_gauss_lobatto()

        
        # Building Elemental Mass, Laplacian and Source matrices/vectors
        [MElem_L, LElem_L, FElem_L] = lin.ConstructElem()
        [MElem_Q, LElem_Q, FElem_Q] = quad.ConstructElem()
       
        # Constructing Global Mass, Laplacian and Source matrices/vectors
        [MG_L, LG_L, FG_L] = lin.ConstructGlob(MElem_L, LElem_L, FElem_L)
        [MG_Q, LG_Q, FG_Q] = quad.ConstructGlob(MElem_Q, LElem_Q, FElem_Q)
       
        # Assemble Stiffness matrix and solving Ax = b
        # Dirichlet B.Cs are solved using "Method of Large Number"
        lin.Solve(MG_L, LG_L, FG_L)
        quad.Solve(MG_Q, LG_Q, FG_Q)
        
        # Plotting Error distribution
        plt.figure()
        plt.plot(lin.x, lin.err, '-ob', label = 'Linear')
        plt.plot(quad.x, quad.err, '-or', label = 'Quadratic')
        plt.title("Q4c) Error distribution, |$u_{exact}$ - $u_{fem}$|" + '\n'+ "$N_el$ = " + str(N_el), fontsize = 16)
        plt.legend()
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('|$u_{exact}$ - $u_{fem}$|', fontsize = 16)
        plt.grid()
        plt.show()
        
        # Plotting Error distribution
        plt.figure()
        plt.plot(quad.x, quad.u_ex, '--g', label = 'Exact', linewidth = '4.5')
        plt.plot(lin.x, lin.u, '-ob', label = 'Linear', linewidth = '2')
        plt.plot(quad.x, quad.u, '-or', label = 'Quadratic', linewidth = '2')
        plt.title("Q4c) Solution comparison between Linear/Quadratic" + '\n'+ "$N_el$ = " + str(N_el), fontsize = 16)
        plt.legend()
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('u', fontsize = 16)
        plt.grid()
        plt.show()
            
            
        

        
        

        
        
                
            
            
        