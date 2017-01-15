import numpy as np
import matplotlib.pylab as plt
from stats import stats
from d3_state_machine import *

'''
	Three-nsition 3D time series :
	Predict three-nsition Past => Present => Future
	The number of states has is limited by the ability to handle
	3D numpy array. Normally UP TO 300 states is good rule of thumb.

	You have to call predict() before update(), for .sig, .yhat sake.

'''

class TDTTS: #3D threensition time series

	def __init__(self,
		ctx_len=20, zero_freqs=False, nstates=100, vmin=0, vmax=100
	):
		self.cnt = 0 # count of the items processed so far
		self.ctx_len = ctx_len
		self.ctx = np.zeros(self.ctx_len, dtype=np.int)

		#range properties
		self.vmin = vmin
		self.vmax = vmax
		self.vrange = vmax - vmin
		self.nstates = nstates

		#3D state machine
		self.tdsm = TDSM(nstates)
		self.last_pred = 0
		#if prediction is 0-state or max-state substitute the last TS value instead. 
		#Fixes spikes, but we a cheating a little, don't tell anyone :)
		self.zero_freqs = zero_freqs

		#we collect the whole signal for plotting&stats purposes,
		# .. but just ctx_len buffer will be enough for functioning model
		self.orig = [0]
		self.sig = [0]
		self.nope = [ ] #collect assumed/copied but not predicted : see. zero_freqs
		self.yhat = [ ]

	@property
	def last(self): return self.sig[-1]

	def np_shift_ctx(self, data) : #roll context and the data item at the end
		self.ctx = np.roll(self.ctx, -1)
		self.ctx[-1] = data

	def shift_ctx(self, data) :
		self.ctx.append(data)
		del(self.ctx[0])

	def learn(self, data):
#		assert self.vmin <= data <= self.vmax
		#cut values to min-max range
		if data <= self.vmin : data = self.vmin
		if data >= self.vmax : data = self.vmax

		#smoothing : limit the number of states, so we can do predictions, otherwize we need fuzzy calculations
		data = self.encode(data)

		# << roll ctx
		self.np_shift_ctx(data)
		#skip/dont-learn initial ctx, too many zeros
		if self.cnt > self.ctx_len/2 : self.tdsm.learn(self.ctx)

		self.cnt += 1
		return data

	def predict(self):
		now = self.ctx[-1]
		rv = self.tdsm.predict(now, self.ctx)

		#gives better prediction, corrects for min/max state spikes
		if self.zero_freqs :
			if rv == 1 or rv == self.nstates :
				rv = self.last_pred
				self.nope.append(rv)
			else :
				rv = self.decode(rv)
				self.nope.append(0)

			self.last_pred = self.sig[-1] #or we can use rv, which is the last predicted
			return rv
		else :
			self.nope.append(0)
			return self.decode(rv)

	#have to run predict() before learn(), because the way ctx-fifo is handled and signal-stats accumulated
	def train(self, data, log=False):
		pred = self.predict()
		learned = self.learn(data)

		#collect signal and prediction so we can calculate stats
		self.sig.append(self.decode( learned ))
		self.yhat.append(float(pred))
		if log and len(self.yhat) > 0 : print "%s : %s => %s : %s" % (self.cnt, self.sig[-1] , self.yhat[-1], self.sig[-1] - self.yhat[-1])

	def batch_train(self,sig, log=False):
		self.orig = np.concatenate([[0], sig]) #keep original signal around
		for s in sig : self.train(s, log)

	def encode(self, values):
		return int( np.floor(self.nstates * ((values - self.vmin)/float(self.vrange)) ) )

	def decode(self, values):
		rv = np.floor( (values * self.vrange) / float(self.nstates) ) + self.vmin
		return rv

	def plot(self, skip=0, nope=False, original=True):
		fig = plt.figure()
		ax = fig.add_subplot(111)

		if original : plt.plot(self.orig[skip:], color='yellow')
		plt.plot(self.sig[skip:], color='blue')
		plt.plot(self.yhat[skip:], color='green')

		#plot non predicted
		if nope and self.zero_freqs :
	 		nope = np.array(self.nope, dtype=np.int)[skip:]
			nidxs = (np.where(nope > 0))[0]
			plt.plot(nidxs, nope[nidxs], 'r.')

		metrics = self.stats(skip=skip, original=original)
		fig.suptitle("s:%s, ctx:%s, min:%s, max:%s" % (self.nstates, self.ctx_len, self.vmin, self.vmax ))
		y = 0.998
		for m in ['mape','nll', 'mae', 'rmse', 'r2', 'sparsity', 'mem'] :
			metric = "%s: %s" % (str(m).upper(), metrics[m])
			plt.text(0.998,y, metric, horizontalalignment='right',  verticalalignment='top', transform=ax.transAxes)
			y -= 0.03

		plt.grid()
		plt.tight_layout()


	def stats(self, skip=None, original=False):
		if skip is None or skip == 0 : skip = self.ctx_len
		data = None
		if original :
			data = np.array(self.orig)[skip:]
		else :
			data = np.array(self.sig)[skip:]

		yhat = np.array(self.yhat + [0])[skip:]

		m = {}
		m['mape'] = "%.3f%%" % (stats.mape(data,yhat) * 100)
		m['mae'] =  "%.3f" % stats.mae(data,yhat)
		m['rmse'] = "%.3f" % stats.rmse(data,yhat)
		m['r2'] = "%.3f%%" % (stats.r2(data,yhat) * 100)
		m['nll'] = "%.3f" % stats.nll(data,yhat)

		print "==== MODEL stats =============="
		print "mape: %s " % m['mape']
		print "mae : %s" % m['mae']
		print "rmse: %s" % m["rmse"]
		print "r2: %s" % m['r2']
		print "nll: ", m['nll']
		print "resolution: %s" % (self.vrange/float(self.nstates))

		info = self.tdsm.info()
		m['sparsity'] = "%.3f%%" % (info['sparsity'] * 100)
		m['mem'] = "%.2f MB" % info['mem']
		return m




