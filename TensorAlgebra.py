# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:37:49 2016

@author: kmakantasis
"""

import numpy as np


def vectorize_tensor(X):
    """      
    .. function:: vectorize_tensor(X)

       Function for vectorizing the mode-1 matricizatin of a tensor. The input can be any multidimensional array. 
       Let's say that the inpyut is a K dimensional array, then its shape should be :math:`(N^K, N^{k-1},...,N^1)`, 
       where :math:`N^i` is the number of elements along the :math:`i^{th}` dimension. The first dimension corresponds 
       to rows, the socond to columns, the thord to tubes and so on.

       :param X: the tensor to be vectorized
       :type X: Nd-Array
       :returns: the vectorized tensor
       :rtype: 1d-Array  
    """
    shape_len = len(X.shape)
    roll = np.rollaxis(X, shape_len-1, shape_len-2)
    return roll.flatten()
    
    
def khatri_rao_product(A,B):
    """      
    .. function:: khatri_rao_product(A,B)

       Function for computing the Khatri-Rao product of two matrices :math:`A` and :math:`B`. The matrices must
       have the same number of columns, otherwise the output of the fuction will be wrong. If the :math:`A` has 
       :math:`I` rows and :math:`K` columns and :math:`B` has :math:`J` rows and :math:`K` columns then the 
       Khatri-Rao product will have :math:`IJ` rows and :math:`K` columns.

       :param A: the first of the two matrices
       :type A: 2d-Array
       :param B: the second of the two matrices
       :type B: 2d-Array
       :returns: the Khatri-Rao product of :math:`A` and :math:`B`
       :rtype: 2d-Array  
    """
    cols = np.shape(A)[1]
    
    C = np.zeros((np.shape(A)[0]*np.shape(B)[0], np.shape(A)[1]), dtype = np.float)
    
    for i in range(cols):
        a = A[:,i]
        b = B[:,i]
        c = np.kron(a,b)
        
        C[:,i] = c
        
    return C        


def tensor_3d_matricization(X, mode=1): 
    """      
    .. function:: 3d_tensor_matricization(X, mode=1)

       Function for matricizing a 3D tensor. 

       :param X: a three order tensor
       :type X: 3d-Array
       :param mode: mode of matricization. default = 1
       :type mode: integer (1,2,3)
       :returns: the n-mode matricization of a tensor
       :rtype: 2d-Array  
    """       
    order = len(X.shape)
    X = np.rollaxis(X, 2, 1)
    inter = order-mode
    X = np.rollaxis(X, inter, 0)
    matricization = np.reshape(X, (X.shape[0], -1))
    
    return matricization
    
    

if __name__=="__main__":
    X = np.array((((1,4,7,10), (2,5,8,11),(3,6,9,12)), ((13,16,19,22),(14,17,20,23),(15,18,21,24))), dtype=np.float)
    vec = vectorize_tensor(X)
    
    A = np.array(((1,1,1), (2,2,2)))
    B = np.array(((1,2,3), (4,5,6), (7,8,9))) 
    C = khatri_rao_product(A,B)
    
    print tensor_3d_matricization(X, mode=1)
    
    
    
    
    