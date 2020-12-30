# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np


# +
def desc_array(array,symboltable=None):
    '''describe array (shape,name)'''
    if symboltable==None:
        print('No symboltable',end=', ')
    else:
        print(getVarName(array,symboltable),end=', ')
    print('shape:'+str(array.shape))

def getVarsNames( _vars, symboltable ) :
    """
    This is wrapper of getVarName() for a list references.
    """
    return [ getVarName( var, symboltable ) for var in _vars ]

def getVarName( var, symboltable, error=None ) :
    """
    Return a var's name as a string.\nThis funciton require a symboltable(returned value of globals() or locals()) in the name space where you search the var's name.\nIf you set error='exception', this raise a ValueError when the searching failed.
    """
    if var is None:
        if error == "exception" :
            raise ValueError("Varieble is None")
        else:
            return error
    for k,v in symboltable.items() :
        if id(v) == id(var) :
            return k
    else :
        if error == "exception" :
            raise ValueError("Undefined function is mixed in subspace?")
        else:
            return error
