import argparse
import csv
from datetime import datetime
import pandas as pd
import random
import time


def EpochConv(date):
    date_t = "19" + str(date)
    date_time = str(date_t[0:4]) + "." + str(date_t[4:6]) + "." + str(date_t[6:8])
    pattern = "%Y.%m.%d"
    epoch = int(time.mktime(time.strptime(date_time, pattern)))
    return epoch


def RandomValGen(min, max):
    n = random.randint(min, max)
    return n


def flowGen(flowbit):

    if flowbit == 0:
        n = random.randint(0, 100)
    else:
        n = random.randint(1000, 10000)
    return n



def TrueCaseGen(smurfingList, dateList, flowbit, NumSmurf):
    count = 0
    # intermediaries included only
    injected = []

    mintime = dateList[0]
    maxtime = dateList[-1]
    maxmin = maxtime - mintime
    for items in smurfingList[1:-1]:
        count += 1
        if count <= (NumSmurf / 2):
            # identifying t1 in fanIn
            injected.append([smurfingList[0], items, dateList[0], flowGen(flowbit)])
            # identifying t1' in fanIn in 1/5 of data
            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[1], dateList[1] + int(maxmin / 5)),
                    flowGen(flowbit),
                ]
            )
            # identifying t2 in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(dateList[1], dateList[1] + int(maxmin / 4)),
                    flowGen(flowbit),
                ]
            )
            # identifying t2' in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(
                        dateList[1] + int(maxmin / 4), dateList[1] + int(maxmin / 2)
                    ),
                    flowGen(flowbit),
                ]
            )

        else:
            injected.append([smurfingList[0], items, dateList[10], flowGen(flowbit)])

            # identifying t1' in fanIn in 1/5 of data
            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[10], dateList[10] + int(maxmin / 5)),
                    flowGen(flowbit),
                ]
            )
            # identifying t2 in fanOut
            injected.append([items, smurfingList[-1], dateList[2], flowGen(flowbit)])
            # identifying t2' in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(dateList[10] + int(maxmin / 4), dateList[10] + maxmin),
                    flowGen(flowbit),
                ]
            )
    return injected


def TruePeriodGen(smurfingList, dateList, flowbit):
    flowbit = 1
    # intermediaries included only
    injected = []

    for items in smurfingList[1:-1]:

        # FAN IIIIIIIIINNN
        tmpFlow = flowGen(flowbit)
        injected.append([smurfingList[0], items, dateList[10], tmpFlow])
        # 1 month period
        injected.append([smurfingList[0], items, dateList[10] + 2630000, tmpFlow])
        del dateList[10]

        # identifying t1' in fanIn in 1/5 of data
        tmpFlow = flowGen(flowbit)
        tmp = RandomValGen(dateList[10], dateList[10] + (2630000 / 10))
        injected.append([smurfingList[0], items, tmp, tmpFlow])
        # 1 month period
        injected.append([smurfingList[0], items, tmp + 2630000, tmpFlow])

        # FAN OOOUTTTTTTTTTTT
        # identifying t2 in fanOut
        tmpFlow = flowGen(flowbit)
        injected.append([items, smurfingList[-1], dateList[2], tmpFlow])
        del dateList[2]
        # 1 month period
        injected.append([items, smurfingList[-1], dateList[2] + 2630000, tmpFlow])

        # identifying t2' in fanOut
        tmpFlow = flowGen(flowbit)
        tmp = RandomValGen(dateList[10] + (2630000 / 10), dateList[10] + 2630000)
        injected.append([items, smurfingList[-1], tmp, tmpFlow])

        # 1 month period
        injected.append([items, smurfingList[-1], tmp + 2630000, tmpFlow])
    return injected

def FalseCaseGen(smurfingList, dateList, NumSmurf):
    count = 0
    # intermediaries included only
    injected = []
    mintime = dateList[0]
    maxtime = dateList[-1]
    maxmin = maxtime - mintime
    for items in smurfingList[1:-1]:
        count += 1
        if count <= (NumSmurf / 2):
            # identifying t1 in fanIn
            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[50], dateList[50] + int(maxmin / 10)),
                    RandomValGen(100, 1000),
                ]
            )
            # identifying t1' in fanIn in 1/5 of data
            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(
                        dateList[50] + int(maxmin / 10), dateList[50] + int(maxmin / 5)
                    ),
                    RandomValGen(100, 10000),
                ]
            )
            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[50] + int(maxmin / 5), dateList[50] + maxmin),
                    RandomValGen(0, 1000),
                ]
            )

            # identifying t2 in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(dateList[0], dateList[49]),
                    RandomValGen(100, 10000),
                ]
            )

        else:

            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[1] + int(maxmin / 5), dateList[1] + maxmin),
                    RandomValGen(100, 1000),
                ]
            )

            # identifying t2 in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(
                        dateList[1] + int(maxmin / 10), dateList[1] + int(maxmin / 5)
                    ),
                    RandomValGen(1000, 10000),
                ]
            )
            # identifying t2' in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(dateList[1], dateList[1] + int(maxmin / 10)),
                    RandomValGen(600, 1000),
                ]
            )
    return injected


def Mix1CaseGen(smurfingList, dateList, flowbit, NumSmurf):
    # 7 true cases and 3 false cases
    count = 0
    # intermediaries included only
    injected = []
    mintime = dateList[0]
    maxtime = dateList[-1]
    maxmin = maxtime - mintime
    for items in smurfingList[1:-1]:
        count += 1
        # 80 percent of true number of smurfs
        if count <= (0.8 * NumSmurf):
            injected.append([smurfingList[0], items, dateList[10], flowGen(flowbit)])
            # identifying t1' in fanIn in 1/5 of data
            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[10], dateList[10] + int(maxmin / 5)),
                    flowGen(flowbit),
                ]
            )
            # identifying t2 in fanOut
            injected.append([items, smurfingList[-1], dateList[2], flowGen(flowbit)])
            # identifying t2' in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(dateList[10] + int(maxmin / 4), dateList[10] + maxmin),
                    flowGen(flowbit),
                ]
            )

        else:

            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[1] + int(maxmin / 5), dateList[1] + maxmin),
                    flowGen(flowbit),
                ]
            )
            # identifying t2 in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(
                        dateList[1] + int(maxmin / 10), dateList[1] + int(maxmin / 5)
                    ),
                    flowGen(flowbit),
                ]
            )
            # identifying t2' in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(dateList[1], dateList[1] + int(maxmin / 10)),
                    flowGen(flowbit),
                ]
            )
    return injected


def Mix2CaseGen(smurfingList, dateList, NumSmurf):
    # 3 true cases and 7 false cases
    count = 0
    # intermediaries included only
    injected = []
    mintime = dateList[0]
    maxtime = dateList[-1]
    maxmin = maxtime - mintime
    for items in smurfingList[1:-1]:
        count += 1
        if count <= (0.3 * NumSmurf):
            injected.append(
                [smurfingList[0], items, dateList[0], RandomValGen(500, 5000)]
            )
            # identifying t1' in fanIn in 1/5 of data
            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[1], dateList[1] + int(maxmin / 5)),
                    RandomValGen(500, 5000),
                ]
            )

            # identifying t2 in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(dateList[1], dateList[1] + int(maxmin / 4)),
                    RandomValGen(500, 5000),
                ]
            )
            # identifying t2' in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(
                        dateList[1] + int(maxmin / 4), dateList[1] + int(maxmin / 2)
                    ),
                    RandomValGen(500, 5000),
                ]
            )

        else:

            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[50], dateList[50] + int(maxmin / 10)),
                    RandomValGen(500, 5000),
                ]
            )
            # identifying t1' in fanIn in 1/5 of data
            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(
                        dateList[1] + int(maxmin / 10), dateList[1] + int(maxmin / 5)
                    ),
                    RandomValGen(500, 5000),
                ]
            )
            injected.append(
                [
                    smurfingList[0],
                    items,
                    RandomValGen(dateList[1] + int(maxmin / 5), dateList[1] + maxmin),
                    RandomValGen(500, 5000),
                ]
            )

            # identifying t2 in fanOut
            injected.append(
                [
                    items,
                    smurfingList[-1],
                    RandomValGen(dateList[0], dateList[49]),
                    RandomValGen(500, 5000),
                ]
            )
    return injected


def RandomSmurfGen(from_acc, to_acc, NumSmurf, NumAttack):
    randSmurf = []
    fromList = []
    toList = to_acc.tolist()

    for item in range(NumAttack):
        fromList = from_acc.tolist()
        smurfing = []
        src = random.choices(fromList, k=1)
        # fromList.remove(src[0])
        dst = random.choices(toList, k=1)
        # toList.remove(dst[0])
        # find random smurfs
        smurfs = random.choices(toList, k=NumSmurf)

        smurfing = src + smurfs + dst
        randSmurf.append(smurfing)

    return randSmurf

if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser(description="Parameters for making the attacks")
    # parser.add_argument('--f', default='data2/smurfGraphSmall.txt', type=str, help='Input Path')
    parser.add_argument("--s", default=None, type=int, help="Number of smurfs")
    parser.add_argument("--a", default=None, type=int, help="Number of attacks")
    args = parser.parse_args()
    args.s = 10
    args.a = 100

    # smurf_order = AA_Smurf(ajm, args.i, args.o)
    old_df = pd.read_csv("data/B4Inject_cfd_trans.csv", sep=",")
    from_acc = old_df["from"].unique().astype(int)
    to_acc = old_df["to"].unique().astype(int)
    # old_df['date'] = old_df['date'].apply(EpochConv)
    date = old_df["date"].unique().astype(int)
    dateList = date.tolist()

    # making random non duplicate sets of smurf injection
    smurfingList = RandomSmurfGen(from_acc, to_acc, args.s, args.a)
    allInjected = []
    trueInjections = []
    counter = 1
    for i in smurfingList:
        tmp = []
        # if counter%7==0:
        # tmp=TruePeriodGen(i,dateList,1)
        # trueInjections.append(i)
        if counter % 7 == 1:
            tmp = TrueCaseGen(i, dateList, 1, args.s)
            trueInjections.append(i)
        elif counter % 7 == 2:
            tmp = FalseCaseGen(i, dateList, args.s)
        elif counter % 7 == 3:
            tmp = Mix1CaseGen(i, dateList, 1, args.s)
            trueInjections.append(i)
        elif counter % 7 == 4:
            tmp = Mix2CaseGen(i, dateList, args.s)
        elif counter % 7 == 5:
            tmp = TrueCaseGen(i, dateList, 0, args.s)
        elif counter % 7 == 6:
            tmp = Mix1CaseGen(i, dateList, 0, args.s)
        allInjected = allInjected + tmp
        counter += 1

    # Create the pandas DataFrame
    df = pd.DataFrame(allInjected, columns=["from", "to", "date", "amount"])
    result_df = old_df.append(df)
    rslt_df = result_df.sort_values(by=["date"])
    rslt_df.to_csv(
        "./analysis/revision/"
        + str(args.s)
        + "_"
        + str(args.a)
        + "_SmurfCfd_trans.csv",
        index=False,
    )

