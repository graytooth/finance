#財務工程1.模擬random walk

#random walk的模擬#####
#先設定需要模擬的次數(n)
n<-50
#預留一個變數，後續用來儲存標的資產的價格
s<-100 #t=0時的價格
asset<-c(s,rep(0,n-1))
#使用合適的機率分布產生亂數，這裡使用的是uniform，
#用normal也可以，只要確保後續用來判斷的兩事件互斥且發生機率皆為0.5即可。
epislon<-c(0,runif(n-1,min=0,max=1))
for (i in 2:n){
  if (epislon[i]>0.5)
    epislon[i]<-1
  else epislon[i]<-(-1)
  asset[i]<-asset[i-1]+epislon[i]
}
#畫出折線圖
#需要讀的套件#####
#ggplot
install.packages("ggplot2")
library(ggplot2)
#好看的顏色的套件
install.packages("RColorBrewer")
library("RColorBrewer")
brewer.pal(n=7,name="Blues")
#畫圖#####
t<-c(1:n)
data<-data.frame(t,asset)
ggplot(data,aes(x=t, y=asset)) +
  geom_area( fill="#084594", alpha=0.4) +
  geom_line(color="#084594", size=0.1) +
  geom_point(size=1, color="#084594") +
  ggtitle("random walk")+
  scale_y_continuous(limits = c(90,110))
