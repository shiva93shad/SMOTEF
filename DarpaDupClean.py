import csv
from random import random

import pandas as pd

# duplicate cleaner
"""
header_list = ["from", "to", "time",'label']
header_list2 = ["from", "to", "time"]
df = pd.read_csv("./data/darpa_processed.csv", names=header_list)
df.drop('label', inplace=True, axis=1)
df = df.groupby(header_list2).size().reset_index(name='count')

#dx.to_csv("./data/darpa_NonDup.csv",index=False)
df.sort_values('time').to_csv("./data/darpa_NonDup.csv",index=False)
exit()
"""


edgelist = []
smurfEdges = []

newedgeList = []
count = 0
with open(".\data\darpa_processed.csv", newline="") as f:
    reader = csv.reader(f)
    for item in list(reader):
        if count == 0:
            count += 1
            continue
        if item[3] == "smurf" and int(item[1]) == 8117:
            edgelist.append([item[0], item[1], int(item[2]), 1])
            edgelist.append(["238", item[0], int(item[2]) - 10, 1])
            edgelist.append(["3138", item[0], 21000, 1])
            edgelist.append(["277", item[0], 22000, 1])
            smurfEdges.append("238" + str(item[0]))
            smurfEdges.append(item[0] + item[1])
            continue
        if item[3] == "smurf" and int(item[1]) == 7607:
            edgelist.append([item[0], item[1], int(item[2]), 1])
            edgelist.append(["248", item[0], int(item[2]) - 10, 1])
            edgelist.append(["31", item[0], 21000, 1])
            edgelist.append(["279", item[0], 22000, 1])
            smurfEdges.append("248" + str(item[0]))
            smurfEdges.append(item[0] + item[1])
            continue
        edgelist.append([item[0], item[1], int(item[2]), 1])


# add smurf fanIn into groundtruth dataset
df_smurf = pd.DataFrame(edgelist, columns=["from", "to", "time", "label"])
df_smurf.sort_values("time")
df_smurf.to_csv("./smurf_add_darpa/3darpa_processed_GT.csv", index=False, header=None)


with open("./smurf_add_darpa/3darpa_smurf_GT.csv", newline="", mode="w") as f:
    writer = csv.writer(f)
    for items in smurfEdges:
        writer.writerow([items])

# delete duplicates
header_list = ["from", "to", "time", "label"]
header_list2 = ["from", "to", "time"]
df = pd.read_csv(
    "./smurf_add_darpa/3darpa_processed_GT.csv", names=header_list, header=None
)
# df.drop('label', inplace=True, axis=1)
print(df)
dz = df.groupby(header_list2).size().reset_index(name="count")
dz.sort_values("time").to_csv("./smurf_add_darpa/3darpa_NonDup.csv", index=False)
