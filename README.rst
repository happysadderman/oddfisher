 OddFisher
##########

oddfisher fisher exact test includes the odd ratio parameter, which is equivalent to *or* parameter from fisher.test in R, where **or** is hypothesized odds ratio.
This parameter is currently not available in fisher exact test in scipy.stats package, and I hope to benefit the python users, looking for python-based fisher exact test with with hypothesized odds ratio parameter.

Please note that this code is based on `R fisher-test implementation <https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/fisher.test.R>`_, relying on the scipy.stats package to achieve equivalency around the negative hypergeometric distribution functions.


Instruction
===========

1. Clone the repo

.. code-block::

        > git clone git@github.com:happysadderman/oddfisher.git
        > cd oddfisher
        > poetry install .

2. To run, check out the usage with help menu

.. code-block::

        > oddfisher fisherexact -h


Altermatively, if you want to use the fisher exact test as stand-alone script, you can run the code using

.. code-block::

        > python fisher.py fisherexact 1 2 3 4 --odd-ratio 10
        > python fisher.py fisherexact 1 2 3 4  # default fisher-exact with no hypothesized odd-ratio

Example command
---------------

Example code shows how parameter **--odd-ratio** can be used during the fisher exact test.
When **--odd-ratio** parameter is not used, it defaults to odd-ratio of 1.

.. code-block::

        > oddfisher fisherexact --odd-ratio 10 1 2 3 4
        Results
        -------------------
        p-value: {'two-sided': np.float64(0.07542579075425782), 'less': 0.07542579075425782, 'greater': 0.997566909975669}
        confidence interval at 0.95: (0.008503581019485222, 20.296323344994953)
        odds ratio: 0.6937896639529924

Above is equivalent to R fisher.test, with **or=10**

.. code-block::

        > d <- matrix(c(1, 2, 3, 4), nrow=2)
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
