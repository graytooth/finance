#�]�Ȥu�{2-1.����Arithmetic Brownian Motion

######Arithmetic Brownian Motion������#####
#���]�w�ݭn����������(n)
n<-200
#�w�d�@���ܼơA����Ψ��x�s�Ъ��겣������
s<-0#t=0�ɪ�����
asset<-c(s,rep(0,n-1))
#�]�w�C���ɶ����������P�ܲ��{�ץH�ήɶ������Τj�p
u<-0
sigma<-1 #�зǮt�A���O�ܲ���
dt<-0.01
#�ϥ�normal�Ӳ���Brownian Motion
dWt<-c(0,rnorm(n-1,0,dt))
for (i in 2:n){
  asset[i]<-asset[i-1]+(u*dt+sigma*dWt[i])
}
######�e�X��u��#####
#�ݭnŪ���M��#####
#ggplot
install.packages("ggplot2")
library(ggplot2)
#�n�ݪ��C�⪺�M��
install.packages("RColorBrewer")
library("RColorBrewer")
brewer.pal(n=7,name="Blues") #��ܨ䤤�ݭn���M��
#�e��#####
t<-c(1:n)
data<-data.frame(t,asset)
png("Brownian Motion R.png", width = 640, height = 360) # �]�w��X����
ggplot(data,aes(x=t, y=asset)) +
  geom_area( fill="#084594", alpha=0.4) +
  geom_line(color="#084594", size=0.1) +
  geom_point(size=0.1, color="#084594") +
  ggtitle("Brownian Motion")+
  scale_y_continuous(limits = c(-0.5,0.5))
dev.off() # ������X����