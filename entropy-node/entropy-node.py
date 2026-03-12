import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    if len(y) == 0:
        return float(0)
    count = dict()
    for i in y:
        if i not in count:
            count[i] = 1
        else:
            count[i]+=1
    res = float(0)
    for i in count:
        p=count[i]/len(y)
        res -= p*np.log2(p)
    return res