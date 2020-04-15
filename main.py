#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:07:54 2020

@author: chihin
"""

from class_helmholtz import Helmholtz

test = Helmholtz(10,2)
test.gauss_lobatto_laplacian()
test.gauss_lobatto_mass()
test.MatElem()
test.MatGlob()
test.SourceGlob()
test.Solve()
