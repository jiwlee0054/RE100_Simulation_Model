import os
import threading
import time
import pickle
import copy
import json

import lib.parameters as para

def read_pkl(loc, name):
    with open(f"{loc}/{name}", 'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data

def save_pkl(loc, name, item):
    with open(f'{loc}/{name}', 'wb') as f:
        pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)

def worker(file):
    os.system('python.exe ' + file)


if __name__ == '__main__':
    options = para.ProbOptions()
    pp_num = options.pp_num
    loc_pp = f"{options.loc_result}/{options.date_result}"
    file_pp = 'result_profile.pkl'
    file_main = 'main.py'

    if os.path.isfile(f"{loc_pp}/{file_pp}"):
        result = read_pkl(loc_pp, file_pp)
        for s in result.keys():
            if result[s] == 1 and os.path.isfile(f"{options.loc_pp_result}/{s}.json"):
                pass
            elif result[s] == 1 and os.path.isfile(f"{options.loc_pp_result}/{s}.json") == False:
                result[s] = 0
            elif result[s] == 0 and os.path.isfile(f"{options.loc_pp_result}/{s}.json"):
                result[s] = 1
            else:
                pass
        save_pkl(loc_pp, file_pp, result)
    else:
        result = dict()
        for s in range(1, pp_num + 1):
            result[s] = 0
        save_pkl(loc_pp, file_pp, result)

    complete_num = copy.deepcopy(pp_num)

    while complete_num != 0:
        try:
            thr_count = threading.active_count() - 4

            if thr_count >= 60:
                time.sleep(5)
                continue

            else:
                if read_pkl(loc_pp, file_pp)[complete_num] == 1:
                    pass
                else:
                    p = threading.Thread(target=worker, args=(file_main,))
                    p.start()
                    time.sleep(2)

                if read_pkl(loc_pp, file_pp)[complete_num] == 1:
                    complete_num -= 1
                else:
                    pass

        except EOFError:
            pass