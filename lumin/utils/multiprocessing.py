import multiprocessing as mp
from typing import Callable, Any, List, Dict

__all__ = ['mp_run']


def mp_run(args:List[Dict[Any,Any]], func:Callable[[Any],Any]) -> Dict[Any,Any]:
    r'''
    Run multiple instances of function simultaneously by using a list of argument dictionaries
    Runs given function once per entry in args list.

    .. Important:: Function should put a dictionary of results into the `mp.Queue` and each result key should be unique otherwise they will overwrite one another.

    Arguments:
        args: list of dictionaries of arguments
        func: function to which to pass dictionary arguments

    Returns:
        DIctionary of results
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
    