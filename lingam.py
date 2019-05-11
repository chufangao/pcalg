import pandas as pd
from sklearn.decomposition import FastICA

def lingam(data, verbose=False):
    pass


if __name__ == '__main__':
    file = 'datasets/gmD.csv'
    # file = 'datasets/BD Cont.csv'
    # file = 'datasets/BD Disc.csv'
    # file = 'datasets/BD5 Cluster X Disc Y Outcome (2).csv'
    # file = 'datasets/BD5 Cluster X2 Cont X1 Outcome (1).csv'
    # file = 'datasets/BD5 Cluster X2 Disc X1 Outcome (1).csv'
    # file = 'datasets/ID1 Disc (1).csv'
    # file = 'datasets/ID1 Disc (2).csv'
    # file = 'datasets/mdata.csv'
    # file = 'datasets/mdata2.csv'
    # file = 'datasets/dataset1-continuous.csv'
    # file = 'C:/Users/gaoan/Desktop/dataset/dataset1 (0-5).csv'
    data = pd.read_csv(file)
    print(data.columns)
    p = lingam(data, True)
