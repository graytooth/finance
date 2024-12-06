#財務工程5.模擬binomial tree

#binomial tree的參數設定#####

s<-100 #期初資產價格
k<-105 #履約價格
r<-0.07
u<-1.2 #上升比例
d<-0.8 #下降比例
n<-5 #模擬次數

#binomial tree的模擬#####
p<-(1+r-d)/(u-d)
p #風險中立機率
#先建立組合數的函數
comb = function(n, x) {
  factorial(n) / factorial(n-x) / factorial(x)
}
#預留用於s和c/p儲存的向量
m<-(n+2)*(n+1)*0.5 #當做n期時，恰有m個結果
asset<-c(rep(0,m))
call<-c(rep(0,m))
put<-c(rep(0,m))
for (i in 1:n){
  for (j in 1:i+1){
  }
}
