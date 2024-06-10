#財務工程2-1.模擬Arithmetic Brownian Motion

######Arithmetic Brownian Motion的模擬#####
#先設定需要模擬的次數(n)
n<-200
#預留一個變數，後續用來儲存標的資產的價格
s<-0#t=0時的價格
asset<-c(s,rep(0,n-1))
#設定每單位時間平均成長與變異程度以及時間的切割大小
u<-0
sigma<-1 #標準差，不是變異數
dt<-0.01
#使用normal來產生Brownian Motion
dWt<-c(0,rnorm(n-1,0,dt))
for (i in 2:n){
  asset[i]<-asset[i-1]+(u*dt+sigma*dWt[i])
}
######畫出折線圖#####
#需要讀的套件#####
#ggplot
install.packages("ggplot2")
library(ggplot2)
#好看的顏色的套件
install.packages("RColorBrewer")
library("RColorBrewer")
brewer.pal(n=7,name="Blues") #選擇其中需要的套組
#畫圖#####
t<-c(1:n)
data<-data.frame(t,asset)
png("Brownian Motion R.png", width = 640, height = 360) # 設定輸出圖檔
ggplot(data,aes(x=t, y=asset)) +
  geom_area( fill="#084594", alpha=0.4) +
  geom_line(color="#084594", size=0.1) +
  geom_point(size=0.1, color="#084594") +
  ggtitle("Brownian Motion")+
  scale_y_continuous(limits = c(-0.5,0.5))
dev.off() # 關閉輸出圖檔
