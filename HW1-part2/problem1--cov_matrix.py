# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:32:49 2021

@author: AllenPC
"""
import numpy as np
A = np.array([89.7, -50.5, -39.2])
B = np.array([-10.1, 48.5, -38.4])
C = np.array([-9.8, -48, 57.8])
# A = np.array([2.4, -2, -0.4])
# B = np.array([-0.4, 0, 0.4])
# C = np.array([-0.1, 0.5, -0.4])

def cross_product(M):
    M = np.expand_dims(M,axis=1)
    transp_M = np.transpose(M)
    matrix = np.dot(M,transp_M)
    # matrix = np.dot(transp_M,M)
    return matrix

def covarince_matrix(A, B, C):
    pi_a = 0.1
    pi_b = 0.5
    pi_c = 0.4
    covariance  = pi_a*cross_product(A) + pi_b*cross_product(B) + pi_c*cross_product(C)
    return covariance

cov_mat = covarince_matrix(A, B, C)