#�]�Ȥu�{2-2.����Geometric Brownian Motion

######Geometric Brownian Motion������#####
#���]�w�ݭn����������(n)
n<-200
t<-c(1:n)
#�w�d�@���ܼơA����Ψ��x�s�Ъ��겣������
s<-100 #t=0�ɪ�����
asset<-c(s,rep(0,n-1))
#�]�w�C���ɶ����������P�ܲ��{�ץH�ήɶ������Τj�p
u<-0
sigma<-1 #�зǮt�A���O�ܲ���
dt<-0.01
#�ϥ�normal�Ӳ���Brownian Motion
for (i in 2:n){
  asset[i]<-s*exp(rnorm(1,u*t*dt,(sigma^2)*dt*t))
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

data<-data.frame(t,asset)
png("Geometric Brownian Motion R.png", width = 640, height = 360) # �]�w��X����
ggplot(data,aes(x=t, y=asset)) +
  geom_area( fill="#084594", alpha=0.4) +
  geom_line(color="#084594", size=0.1) +
  geom_point(size=0.1, color="#084594") +
  ggtitle("Geometric Brownian Motion")+
  scale_y_continuous(limits = c(95,105))
dev.off() # ������X����