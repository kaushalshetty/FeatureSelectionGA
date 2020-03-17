from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored 
from sksurv.metrics import concordance_index_ipcw

import numpy as np

"""
def concordance_index_censored(event_indicator, event_time, estimate, tied_tol = 1e-08):
	Concordance index for right-censored data
	Parameters
	----------
	event_indicator, array-like, shape = (n_samples,)
		Array containing the time of an event or time of censoring
	event_time, array-like, shape = (n_samples, )
		Array containing the time of an event or time of censoring
	estimate, array-like, shape = (n_samples, )
		Estimated risk of experiencing an event
	tied_tol, float, optional, default = 1e-08

	Returns
	-------
	cindex, float
		concordance index
	concordant, int
		number of concordant pairs
	disconcordant, int
		number of disconcordant pairs
	tied_risk, int
		number of pairs having tied estimated risks
	tied_time, int
		number of comparable pairs sharing the same time

"""

class SurvivalFitness:

	def __init__(self, n_splits=5, )