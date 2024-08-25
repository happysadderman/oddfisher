# README for OddFisher

Clone the repo
> git clone git@github.com:happysadderman/oddfisher.git
> cd oddfisher
> poetry install .

To run, check out the usage
> oddfisher fisherexact -h

Example command
> oddfisher fisherexact --odd-ratio 10 1 2 3 4
Results
-------------------
p-value: {'two-sided': np.float64(0.07542579075425782), 'less': 0.07542579075425782, 'greater': 0.997566909975669}
confidence interval at 0.95: (0.008503581019485222, 20.296323344994953)
odds ratio: 0.6937896639529924

Above is equivalent to R fisher.test
> d <- matrix(c(1, 2,, 3, 4), nrow=2)
> fisher.test(d, or=10)


        Fisher's Exact Test for Count Data

data:  d
p-value = 0.07543
alternative hypothesis: true odds ratio is not equal to 10
95 percent confidence interval:
  0.008512238 20.296715040
sample estimates:
odds ratio
  0.693793
