import multiprocessing as mp
from typing import Callable, Any, List, Dict


def mp_run(args:List[Dict[Any,Any]], func:Callable[[Any],Any]) -> Dict[Any,Any]:
    r'''
    Run multiple instances of function sumultaneously by using a list of argument dictionaries
    '''

    procs = []
    out_q = mp.Queue()
    for i in range(len(args)):
        p = mp.Process(target=func, args=(args[i], out_q))
        procs.append(p)
        p.start() 
    result_dict = {}
    for i in range(len(args)): result_dict.update(out_q.get()) 
    for p in procs: p.join()  
    return result_dict
    