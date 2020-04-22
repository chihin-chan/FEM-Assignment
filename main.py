#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:07:54 2020

@author: chihin
"""

from class_helmholtz import Helmholtz

# Initialising Class with Helmholtz(N_el, p)
# N_el -> No. of elements
# P -> Polynomial order, p=1/2 for linear/quad
case = Helmholtz(3, 2)

# Initialising phi/dphi values at Gauss-Lobatto quadrature points
case.init_gauss_lobatto()

# Building Elemental Mass, Laplacian and Source matrices/vectors
[MElem, LElem, FElem] = case.ConstructElem()

# Constructing Global Mass, Laplacian and Source matrices/vectors
[MG, LG, FG] = case.ConstructGlob(MElem, LElem, FElem)

# Assemble Stiffness matrix and solving Ax = b
# Dirichlet B.Cs are solved using "Method of Large Number"
case.Solve(MG, LG, FG)

# Plotting Solution Once
case.PlotOnce()

case.ErrorAnalysis()

case.AltErrorAnalysis(6)