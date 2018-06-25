from numpy import *

def importdata(filepath):
    """
    Function to load txt data to matrix
    """
    with open(filepath) as fileid:
        content = fileid.readlines()
    listdata = [item.strip('\n').split('\t') for item in content]
    arraydata = array(listdata)
    returnMat = arraydata[:,:3]
    labelVector = arraydata[:,3] 
    return returnMat, labelVector

#