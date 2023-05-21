import numpy as np
import sys

DEBUGGING = True

def setdebugging(boolarg=True):
    global DEBUGGING
    DEBUGGING = boolarg
    print(f"DEBUGGING : {DEBUGGING}")

def is_debugging_true():
    global DEBUGGING
    return DEBUGGING


def debug(*expressions):
    """Call with expression string to return evaluated values
    e.g. debug('a+1','b') will print  
    >>> a+1 : 5 
    >>> b : 3 
    """
    if DEBUGGING == False:
        return

    frame = sys._getframe(1)    
    for expression in expressions:
        # print(f'{arg=}'.split('=')[0], arg)
        print(expression, ':', eval(expression, frame.f_globals, frame.f_locals))

def debugprint(*expressions):
    """Call with expression string to return evaluated values
    e.g. debugprint('a+1','b') will print  
    >>> 5 
    >>> 3 
    """
    if DEBUGGING == False:
        return

    frame = sys._getframe(1)    
    for expression in expressions:
        print(repr(eval(expression, frame.f_globals, frame.f_locals)))


def quickscatter3d(pts, title=None, darkmode=False, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np

    if darkmode:
        plt.style.use('dark_background')

    fig = plt.figure()#figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    pts=np.array(pts)
    # print(pts)
    ax.set_box_aspect(aspect = (1,1,1))
    ax.scatter(*np.hsplit(pts,3), **kwargs)
    if title!=None:
        plt.title(title)
        
    plt.show()


def quickscatter3d_multi(l_pts, title=None, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()#figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    for pts in l_pts:
        pts=np.array(pts)
        ax.scatter(*np.hsplit(pts,3), **kwargs)
        # print(pts)
    if title!=None:
        plt.title(title)
        
    ax.set_box_aspect(aspect = (1,1,1))
    plt.show()


def quickheatmap(arr):
    import matplotlib.pyplot as plt
    plt.imshow(arr)
    plt.show()
    return 


def test_repeat(func,params, repeats=1):
    results={}
    for iter in range(repeats):
        for param in params:
            str_param = str(*param)
            if str_param not in results:
                results[str_param]=[]
            results[str_param].append(func(*param))
    return results

# def pprint_df(arr):


def pprint_arr(arr, level=[], rounding=0):
    if rounding != 0:
        arr = np.around(arr,decimals=rounding)
    ndims = len(arr.shape)
    if ndims==0:
        return
    elif ndims==1:
        print(level, end="\t")
        str_arr = list(map(str,arr))
        # print(str_arr)
        nchars = max(map(len,str_arr))
        # print(nchars, '{0: <'+str(nchars)+'}')
        # print(('{0: <'+str(nchars)+'}').format("1.2"),"a", sep="")
        for num in str_arr:
            # print(num)
            print(('{0: >'+str(nchars)+'}').format(num), end=", ")
        print("")
        # print(" ".join([('{0: <'+str(nchars)+'}').format(num) for num in str_arr]), sep="\t")
    else:
        for i in range(len(arr)):
            pprint_arr(arr[i], level+[i])



def pprint_arr_2d(arr, rounding=0, real=False):
    """Print a 2D array like a near table, with optional rounding and real conversion"""
    if rounding != 0:
        temparr = np.around(arr,decimals=rounding)
    else:
        temparr = arr

    if real:
        temparr = np.real_if_close(temparr)

    for row in temparr:
        for ele in row:
            print("{:>{totalspace}.{digits}f}".format(ele, totalspace=rounding+5, digits=rounding), end=" ")
        print()



def read_txt(filename,col_range=None,row_range=None, colnum=-1):
    """Read data in txt format, separated by whitespace and select the given rows"""
    
    with open(filename,'r') as fin:
        lines = [float(line.strip().split()[colnum]) for line in fin]
    
    return lines


def print_comments(filename,symbol="#",line_no_size=5):
    """Read lines from filename and 
    print all lines starting with symbol. 
    (Remove leading whitespace, ofcourse)"""
    with open(filename, "r") as f:
        for i,line in enumerate(f.readlines()):
            lstripped = line.strip()
            if lstripped:
                if lstripped[0]==symbol:
                    print(f"{i}".ljust(line_no_size), lstripped[1:])
                    printed=True
                elif printed:
                    print()
                    printed=False
                    
    


if __name__ == "__main__":
    # print_comments("/home/agrim/copyBerkeleyGW/Common/vcoul_generator.f90","!")
    # print_comments("/home/agrim/copyBerkeleyGW/Sigma/sigma_main.f90","!")
    filename = "/home/agrim/Documents/copyBerkeleyGW/Common/vcoul_generator.f90"
    print_comments(filename,"!")

    # pprint_arr_2d(20*np.random.uniform(size=120).reshape(10,12), rounding=3)

    # pprint_arr(np.random.uniform(size=60).reshape(3,4,5), rounding=3)




    # a = 5
    # b = "string"
    # debug('a','b')
    # quickscatter3d([[0,0,0],[1,0,1],[2.3,0,1.2]], "Test 1")
    # lines = read_txt("./QE_data/control_scripts/vcoul")
