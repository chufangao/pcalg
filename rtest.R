#source("https://bioconductor.org/biocLite.R")
#biocLite("graph")
#biocLite("RBGL")
#biocLite("Rgraphviz")
#install.packages("pcalg")
#library("Rgraphviz")
library(pcalg)

data(gmD); d1=gmD$x
d1 = read.csv('~/pcalg/datasets/BD Disc.csv', header = TRUE, sep=",")
d1 = read.csv('~/pcalg/datasets/BD Cont.csv', header = TRUE, sep=",")
d1 = read.csv('~/pcalg/datasets/BD5 Cluster X Disc Y Outcome (2).csv', header = TRUE, sep=",")
d1 = read.csv('~/pcalg/datasets/BD5 Cluster X2 Cont X1 Outcome (1).csv', header = TRUE, sep=",")
d1 = read.csv('~/pcalg/datasets/BD5 Cluster X2 Disc X1 Outcome (1).csv', header = TRUE, sep=",")
d1 = read.csv('~/pcalg/datasets/ID1 Disc (1).csv', header = TRUE, sep=",")
d1 = read.csv('~/pcalg/datasets/ID1 Disc (2).csv', header = TRUE, sep=",")
d1 = read.csv('~/pcalg/datasets/mdata.csv', header = TRUE, sep=",")
d1 = read.csv('~/pcalg/datasets/mdata2.csv', header = TRUE, sep=",")
d1 = read.csv('~/pcalg/datasets/dataset1-continuous.csv', header = TRUE, sep=",")
d1 = read.csv('~/2019 Spring/CS590AML/hw1-data/dataset.csv')
d1 = read.csv('C:/Users/gaoan/Downloads/Microsoft.SkypeApp_kzf8qxf38zg5c!App/All/Learn Model Test/datasets/kaggle/admission 1.1.csv')

p = fci(list(C=cor(d1), n=nrow(d1)),pcalg::gaussCItest,alpha=.05,colnames(d1), type="adaptive")
p = fci(list(dm = gmD$x, adaptDF = FALSE),pcalg::disCItest,alpha=.05,c('X1','X2','X3','X4','X5'), type="adaptive")
p = pc(list(C=cor(d1), n=nrow(d1)),pcalg::gaussCItest,alpha=.05,colnames(d1))

score = new("GaussL0penObsScore", d1)
p = ges(new("GaussL0penObsScore", d1, phase = c("forward", "backward"), iterate=FALSE),verbose = TRUE); as(p$essgraph,"matrix")+0
plot((p = ges(new("GaussL0penObsScore", d1)))$essgraph)

#suffstat = list(C=cor(d1), n=nrow(d1))
#skel = skeleton(suffstat,pcalg::gaussCItest,alpha=.05,colnames(d1))
#pc = pc.cons.intern(skel, suffstat, pcalg::gaussCItest,alpha=.05, version.unf = c(1,1))

#p = udag2pdagRelaxed(pc$sk, rules = c(TRUE, TRUE, TRUE))
#pc$unfTripl
#summary(p)