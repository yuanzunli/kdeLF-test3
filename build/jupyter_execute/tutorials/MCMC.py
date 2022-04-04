#!/usr/bin/env python
# coding: utf-8

# (MCMC)=
# 
# # MCMC
# 
# One important feature of our KDE method is involving the Bayesian inference. \texttt{run\_mcmc} implements a fully Bayesian Markov Chain Monte Carlo (MCMC) method to determine the posterior distributions of bandwidths and other parameters (e.g., $\beta$ for adaptive estimators). The MCMC core embedded in kdeLF is the Python package emcee (Foreman-Mackey et al.2013). The Bayesian method allows us to recover the parameters with a complete description of their uncertainties and degeneracies via calculating their probability density functions (PDFs). Then, by running the chain analysis subroutine, one can probe the shape of these PDFs, and the correlations among bandwidth parameters, giving more information than just the best-fit and the marginalized values for the parameters.

# In[1]:


get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')

from matplotlib import rcParams

rcParams["savefig.dpi"] = 100
rcParams["figure.dpi"] = 100
rcParams["font.size"] = 20


# coming soon ...

# In[ ]:




