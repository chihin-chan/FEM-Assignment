#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:07:54 2020

@author: chihin

main.py
Main script for running the code that solves the Helmholtz equation

class_helmholtz.py
Class with functions that aid solving of Helmholtz equation
"""

from class_helmholtz import Helmholtz


"""
Running for just a single case
Elemental, Global matrices can be checked here
"""

# Initialising Class with Helmholtz(N_el, p)
# N_el -> No. of elements
# P -> Polynomial order, p=1/2 for linear/quad
case = Helmholtz(5, 1)

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


"""
Running Grid study for Q4b)

Note that the single case above have to setup because
the ErrorAnalysis() is a class function
"""
# Plots/Displays Error Analysis for Q4b)
case.ErrorAnalysis()


"""
Alternative Error Analysis
for Q4c) which plots compares the error distribution
between linear and quad elements given no. of elements as input

case.AltErrorAnalysis(N),   where N is the no. of elements

Note that the single case have to be setup as well
"""
# Plots/Displays Error Analysis for Q4c)
case.AltErrorAnalysis(6)