library(factoextra)
library(NbClust)

bb <- read_excel("C:/Users/Julian/Desktop/Universidad/ciberseguridad/ciberseguridad/codigos/resultados2.xlsx")

x11()
fviz_nbclust(bb, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")

# Silhouette method
x11()
fviz_nbclust(bb, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")

x11()
set.seed(123)
fviz_nbclust(bb, kmeans, nstart = 25,  method = "gap_stat", nboot = 50)+
  labs(subtitle = "Gap statistic method")