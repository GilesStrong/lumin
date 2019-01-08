import multiprocessing as mp


def mp_run(args, func):
    procs = []
    out_q = mp.Queue()
    for i in range(len(args)):
        p = mp.Process(target=func, args=(args[i], out_q))
        procs.append(p)
        p.start() 
    result_dict = {}
    for i in range(len(args)):
        result_dict.update(out_q.get()) 
    for p in procs:
        p.join()  
    return result_dict
    