#財務工程6.模擬Monte Carol

#Monte Carol的參數設定#####

s<-100 #標的資產期初價格
k<-105 #履約價格
r<-0.07 #利率
sigma<-0.05 #波動程度
t<-150/365 #到期時間
n<-50 #模擬次數

#一次Monte Carol的模擬#####

asset<-c(s,rep(0,n-1))  #用於資產價格隨時間變化儲存
epislon<-c(0,rnorm(n-1,mean=0,sd=1)) #標準常態分佈中隨機抽取n-1個亂數
dt<-t/n #t的變化單位

for (i in 2:n){
  asset[i]<-asset[i-1]*exp((r-0.5*sigma**2)*dt+sigma*epislon[i]*dt**0.5)
}

#歐式買權賣權價格

eu_call<-exp(-r*t)*max(asset[n]-k,0)
eu_put<-exp(-r*t)*max(k-asset[n],0)
eu_call
eu_put

#畫出折線圖####
#需要讀的套件#####
#ggplot
#install.packages("ggplot2")
library(ggplot2)
#好看的顏色的套件
#install.packages("RColorBrewer")
library("RColorBrewer")
brewer.pal(n=7,name="Blues")
#畫圖####

t<-c(1:n)
data<-data.frame(t,asset)
ggplot(data,aes(x=t, y=asset)) +
  geom_area( fill="#084594", alpha=0.4) +
  geom_line(color="#084594", size=0.1) +
  geom_point(size=1, color="#084594") +
  ggtitle("Monte Carol")+
  scale_y_continuous(limits = c(min(asset),max(asset)))

#用Monte Carol模擬多次####

m<-10 #模擬次數
dt<-t/n #t的變化單位
asset_1<-c(rep(0.1,m*n))
type<-c(rep(0,m*n)) #將每次模擬結果進行區分
asset_t<-c(rep(0.1,m)) #儲存s_t

for (j in 1:m){
  asset_0<-c(s,rep(0.1,n-1))  #用於資產價格隨時間變化儲存
  epislon<-c(0,rnorm(n-1,mean=0,sd=1)) #標準常態分佈中隨機抽取n-1個亂數
  for (i in 1:n){
    k<-(j-1)*n+i  #用於計算在最終儲存結果中的位置
    type[k]<-j   #用於對每一次模擬進行區分
    if (i==1){
      asset_1[k]<-s
    }else{
    asset_0[i]<-asset_0[i-1]*exp((r-0.5*sigma**2)*dt+sigma*epislon[i]*dt**0.5)
    asset_1[k]<-asset_0[i]
    }
    if (i==n){
      asset_t[j]<-asset_0[i]
    }
  }
}
#計算期望值
s_t<-mean(asset_t)
#歐式買權賣權價格
eu_call<-exp(-r*t)*max(s_t-k,0)
eu_put<-exp(-r*t)*max(k-s_t,0)
eu_call
eu_put

#畫圖####

t<-c(rep(1:n,m))
data<-data.frame(t,asset_1,type)
ggplot(data,aes(x=t, y=asset_1 ,group=type,color=type)) +
  geom_line(size=0.1) +
  ggtitle("Monte Carol")+
  scale_y_continuous(limits = c(min(asset_1),max(asset_1)))

