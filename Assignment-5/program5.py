import matplotlib.pyplot as plt
# matplotlib.use("gtk")
import pandas as pd

"""
5.a : Recall Precision graph for the following relevant documents and documents retrieved

Rq= {d3,d5, d9,d25,d39,d44,d56,d71,d89,d94,d105,d119,d124,d136, d144}
Aq ={d123,d84,d56,d6,d8,d9,d511,d129,d187,d25,d38,d48,d250,d113 , d44,d99,d95,d214,d136,d39,d128,d71,d14,d5}

"""
# Relevant documents set
# rq = [3,5,9,25,39,44,56,71,89,123]
rq = [3, 5, 9, 25, 39, 44, 56, 71, 89, 94, 105, 119, 124, 136, 144]

# Answer set

aq = [123, 84, 56, 6, 8, 9, 511, 129, 187, 25, 38, 48, 250, 113, 44, 99, 95, 214, 136, 39, 128, 71, 14, 5]
# aq = [123,84,56,6,8,9,511,129,187,25,38,48,250,113,3]

# Recall list initialization
recall = []

# Precision list initialization
precision = []

rlen = len(rq)
alen = len(aq)

recallCount = 0

# to keep track of the retrieved documents
retrievedDocumentCount = 0
# pc = 0

rr = 0
pr = 0

for i in aq:
    retrievedDocumentCount += 1
    if i in rq:
        recallCount += 1
        rr = recallCount / rlen

    pr = recallCount / retrievedDocumentCount

    # print(rr,pp,recall_count,precision_count)
    recall.append(rr * 100)
    precision.append(pr * 100)

print("\n\nThe R-precision value is :", rr)
dashline = "\n\n---------------------------------------------------"
print(dashline)

plt.plot(recall, precision, color='orange')
plt.title('Recall Precision curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0, 115)
plt.ylim(0, 115)
plt.show()

# 5.b: r-precision comparison for to different algorithms for 5 different queries

algo_a = []
algo_b = []


def r_precision(a, b, r):
    c1 = 0
    c2 = 0
    for i in a:
        if i in r:
            c1 += 1
    for i in b:
        if i in r:
            c2 += 1

    algo_a.append(c1 / len(a))
    algo_b.append(c2 / len(b))


# list of the 5 queries, relevant documents, documents retrieved by algorithm a and b respectively
rel1 = [3, 5, 9, 25, 39, 44, 56, 71, 89, 123]
algoA1 = [123, 84, 56, 6, 8, 9, 511, 129, 187, 25, 38, 48, 250, 113, 3]
algoB1 = [12, 39, 13, 123, 8, 9, 19, 89, 87, 25, 70, 71, 29, 44, 3]

rel2 = [3, 20, 5, 68, 51, 21, 27, 64, 6, 93]
algoA2 = [3, 13, 5, 68, 51, 67, 32, 64, 45, 6, 94, 95, 93]
algoB2 = [20, 30, 7, 78, 21, 27, 14, 15, 16, 6, 54, 4, 6]

rel3 = [38, 65, 73, 88, 93, 74, 36, 4, 28, 30]
algoA3 = [66, 88, 45, 43, 23, 12, 188, 200, 34, 4]
algoB3 = [56, 73, 65, 3, 2, 99, 146, 93, 76, 74, 4]

rel4 = [85, 95, 25, 64, 52, 12, 43, 18, 6, 66]
algoA4 = [52, 62, 64, 77, 12, 45, 18, 43, 6]
algoB4 = [95, 85, 25, 77, 123, 3213, 78, 18, 6]

rel5 = [9, 76, 78, 31, 7, 47, 30, 8, 43, 51]
algoA5 = [76, 75, 31, 7, 30, 44, 56, 50, 94, 223]
algoB5 = [78, 9, 48, 47, 4, 31, 43, 56, 55, 99, 123, 222]


r_precision(algoA1, algoB1, rel1)
r_precision(algoA2, algoB2, rel2)
r_precision(algoA3, algoB3, rel3)
r_precision(algoA4, algoB4, rel4)
r_precision(algoA5, algoB5, rel5)

print('\n\n')
print('algorithm a is better' if sum(algo_a) > sum(algo_b) else 'algorithms b is better')
print(dashline)

# algo_a = [0.3,0.6,0.3,0.5,1,0.78,0.24]
# algo_b = [0.1,0.3,0.6,0.4,0,0.7,0.01]

x = map(lambda a, b: a - b, algo_a, algo_b)
x = list(x)

fig = plt.figure(figsize=(10, 10))
langs = [i for i in range(1, len(x) + 1)]

plt.xlabel("Query Number")
plt.ylabel("R Precision A/B")
plt.title("Precision Histogram")

plt.bar(langs, list(x), color='purple', width=0.5)

plt.show()

# 5.c : Harmonic Mean and E-Measure

Rq = ['d3', 'd5', 'd9', 'd25', 'd39', 'd44', 'd56', 'd71', 'd89', 'd123']
A1 = ['d123', 'd84', 'd56', 'd6', 'd8', 'd9', 'd511', 'd129', 'd187', 'd25', 'd38', 'd48', 'd250', 'd113', 'd3']


def calhme(Rq, Aq):
    rel_doc_count = 0
    rn = len(Rq)
    recall, precision, harmonic_mean, em1, em2, em0 = {}, {}, {}, {}, {}, {}

    for i in range(len(Aq)):
        if Aq[i] in Rq:
            rel_doc_count += 1
            recall[Aq[i]] = (round(rel_doc_count / rn, 2))
            precision[Aq[i]] = (round(rel_doc_count / (i + 1), 2))
            harmonic_mean[Aq[i]] = round(2 / ((1 / recall[Aq[i]]) + (1 / precision[Aq[i]])), 2)
            em0[Aq[i]] = round(1 - harmonic_mean[Aq[i]], 2)
            # Set b=2 for E-Measure
            b = 2
            em1[Aq[i]] = round(1 - ((1 + (b ** 2)) / (((b ** 2) / recall[Aq[i]]) + (1 / precision[Aq[i]]))), 2)

            b = 0.2
            em2[Aq[i]] = round(1 - ((1 + (b ** 2)) / (((b ** 2) / recall[Aq[i]]) + (1 / precision[Aq[i]]))), 2)

        else:
            pass


    return pd.DataFrame({'Recall': pd.Series(recall), 'Precision': pd.Series(precision), 'Harmonic mean': pd.Series(harmonic_mean),
                         'E-Measure (b=1)': pd.Series(em0), 'E-Measure (b>1)': pd.Series(em1),
                         'E-Measure (b<1)': pd.Series(em2)})


# Harmonic Mean and E-Measure
resultDataframe = calhme(Rq, A1)
print()
print()
print(resultDataframe)
# resultDataframe.to_csv('5(c).csv')
