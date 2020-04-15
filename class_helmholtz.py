#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:14:07 2020

@author: chihin
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import cos

class Helmholtz:
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
            raise Exception("Invalid polynomial chose. Please select p=1, 2 for linea, quadratic shape functions")
        self.p = p
        
        # Define Shape Functions
        
        
    def gauss_lobatto_mass(self):
        """
        Implements Numerical Integration using Gauss Lobatto
        Stores the value of phi at quadrature points -1, 0, 1
        Used for construct Mass Matrix
        """
        phi = np.zeros((self.p+1, 3))
        if (self.p == 1):
            phi[0,0] = 1.0
            phi[0,1] = 0.5
            phi[0,2] = 0.0
            phi[1,0] = 0.0
            phi[1,1] = 0.5
            phi[1,2] = 1.0
            
        elif (self.p == 2):
            phi[0,0] = 1.0
            phi[0,1] = 0.0
            phi[0,2] = 0.0
            phi[1,0] = 0.0
            phi[1,1] = 1.0
            phi[1,2] = 0.0
            phi[2,0] = 0.0
            phi[2,1] = 0.0
            phi[2,2] = 1.0
            
        self.phi = phi
     
    def gauss_lobatto_laplacian(self):
        """
        Implements Gauss Lobatto Quadrature
        Stores the value of dphi at quadrature points -1, 0, 1
        Used to construct Laplacian Matrix
        """
        dphi = np.zeros((self.p+1, 3))
        if (self.p == 1):
            dphi[0,0] = -0.5
            dphi[0,1] = -0.5
            dphi[0,2] = -0.5
            dphi[1,0] = 0.5
            dphi[1,1] = 0.5
            dphi[1,2] = 0.5
            
        elif (self.p == 2):
            dphi[0,0] = -3/2
            dphi[0,1] = -1/2
            dphi[0,2] = 1/2
            dphi[1,0] = 2.0
            dphi[1,1] = 0.0
            dphi[1,2] = -2.0
            dphi[2,0] = -1/2
            dphi[2,1] = 1/2
            dphi[2,2] = 3/2
            
        self.dphi = dphi
        
    def MatElem(self):
        """
        Constructing elemental matrices
        """
        
        self.MElem = np.zeros((self.p+1,self.p+1))
        self.LElem = np.zeros((self.p+1,self.p+1))
        self.FElem = np.zeros((self.N_el,self.p+1))

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
                    self.MElem[i,j] += self.phi[i,k] * self.phi[j,k] * self.dx/2.0 * weights[k]
                    self.LElem[i,j] += self.dphi[i,k] * self.dphi[j,k] * self.dx/2.0 * (2.0/self.dx)**2 * weights[k]
        
        
        for e in range(self.N_el):
            for j in range(self.p+1):
                for k in range(3):
                    self.FElem[e,j] += self.phi[j,k] * weights[k] * self.dx / 2.0 * source(e*self.dx + j*self.dx/self.p)
        
        # Checking
#        print("Elemental F Vector")
#        print(self.FElem)
#        print("Elemental Mass Matrix:")
#        print(self.MElem)
#        print("Elemental Laplacian Matrix: ")
#        print(self.LElem)
        
    def MatGlob(self):
        MG = np.zeros((self.n_dof, self.n_dof))
        LG = np.zeros((self.n_dof, self.n_dof))
        
        # Defines connectivity map
        connect = np.zeros(self.p+1, dtype = int)
        
        # Intialising Connectivity map
        for i in range(self.p+1):
            connect[i] = i
            
        for e in range(self.N_el):
            for i in range(self.p+1):
                for j in range(self.p+1):
                    MG[connect[i], connect[j]] = MG[connect[i],connect[j]] + self.MElem[i,j]
                    LG[connect[i], connect[j]] = LG[connect[i],connect[j]] + self.LElem[i,j]
            # Updating connectivity map
            connect = connect + self.p
        
        # Prints Global Matrices for Checking
        self.MG = MG
        self.LG = LG
        
    def SourceGlob(self):
        
        self.FGlob = np.zeros(self.n_dof)
        
        # Defines connectivity map
        connect = np.zeros(self.p+1, dtype = int)
        
        # Intialising Connectivity map
        for i in range(self.p+1):
            connect[i] = i
            
        for e in range(self.N_el):
            for i in range(self.p + 1):
                self.FGlob[connect[i]] = self.FGlob[connect[i]] + self.FElem[e,i]
            connect = connect + self.p
            
        self.FGlob = -self.FGlob
    
    def Solve(self):
        
        # Setting up Stiff Matrix and RHS for the homogeneous problem
        stiff = self.LG + self.lmda*self.MG
        stiff = stiff[1:-1,1:-1]
        rhs = self.FGlob[1:-1]
        
        # Solving for the homogenous Solution
        uH = np.zeros(self.n_dof)
        uH = np.linalg.inv(stiff)@rhs
        
        # Solution by adding Dirchlet Condition
        u = uH + self.alpha
        x = np.linspace(0,1,self.n_dof)
   
        # Computing Exact Solution
        u_ex = np.zeros(len(x))
        for i in range(len(x)):
            u_ex[i] = cos(2*pi*x[i])
        
        
        plt.plot(x[1:-1], u)
        plt.plot(x, u_ex)
        plt.grid()
        
        
                
            
            
        