#Demo code for prediction tree taken from: http://www.statmethods.net/advstats/cart.html

library(rpart)
fit <- rpart(t ~.-start_sample, method="class", data=minter)
printcp(fit)
plotcp(fit)
plot(fit, uniform=TRUE, main="Classification Tree")
text(fit, use.n=TRUE, all=TRUE, cex=.8)
predict(fit)