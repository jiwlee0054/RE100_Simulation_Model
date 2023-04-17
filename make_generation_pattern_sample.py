import os
import pandas as pd
import json


if __name__ == "__main__":
    Item = 'WT_onshore'
    loc_data = f"data/{Item}"
    hist_list = os.listdir(f"{loc_data}")

    sample_dict = dict()
    for i in range(365):
        for j in range(24):
            sample_dict[f"{i},{j}"] = []
    for hist in hist_list:
        f = pd.read_excel(f"{loc_data}/{hist}", sheet_name='Sheet1', index_col=0)
        cap_f = pd.read_excel(f"{loc_data}/{hist}", sheet_name='Sheet2').iloc[0, 0]
        f = f.values / cap_f
        for i in range(365):
            for j in range(24):
                sample_dict[f"{i},{j}"].append(f[i, j])

    with open(f"data/generation_sample_{Item}.json", 'w') as f:
        json.dump(sample_dict, f)