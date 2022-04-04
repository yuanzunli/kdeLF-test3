#!/usr/bin/env python
# coding: utf-8

# (parallel)=
# 
# # Parallelization

# In[1]:


get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')

from matplotlib import rcParams

rcParams["savefig.dpi"] = 100
rcParams["figure.dpi"] = 100
rcParams["font.size"] = 20

import multiprocessing

multiprocessing.set_start_method("fork")


# coming soon...

# In[ ]:




