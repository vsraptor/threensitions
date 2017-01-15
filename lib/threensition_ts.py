import numpy as np
import matplotlib.pylab as plt
from stats import stats

'''
	Three-nsition time series :
	Buffer big chunk of the signal. Then search for three-nsition Past => Present => Future to
	predict the nth step in the furure.

	You have to call predict() before learn().

'''

class TTS: #Threensition time series

	def __init__(self,
		 buffer_len=1000, ctx_len=20, predict_steps=1, include_ctx=False,
		 resolution=10, smooth=True
	):
		self.buffer_len = buffer_len
		self.buf = np.zeros(self.buffer_len)
		self.cnt = 0 # count of the items processed so far
		#FIFO buffer rolls right to left, the last element is the NOW element
		self.ctx_end_ix = self.buf.size - 1 #last element is NOW|PRESENT
		self.ctx_len = ctx_len
		self.scores = {} #holds the prediction scores to decide future state
		self.predict_steps = predict_steps #how many steps in the future to predict
		#should the PAST include PRESENT context when matching i.e. ctx
		self.include_ctx = include_ctx

		self.resolution = resolution
		self.smooth = smooth #should I adjust the original data

		#stats
		self.orig = [0]
		self.sig = [0]
		self.yhat =  [ ]

	@property
	def last(self): return self.buf[-1]

	#manage buffer, FIFO.. right to left
	def learn(self, data):
		#limit the number of states, so we can do predictions, otherwize we need fuzzy calculations
		if self.smooth : data = self.encode(data)

		self.buf = np.roll(self.buf, -1)
		self.buf[-1] = data
		self.cnt += 1

		return data

	def predict(self, predict_steps=None):
		if predict_steps is None : predict_steps = self.predict_steps
		else: self.predict_steps = predict_steps

		ctx_start = self.ctx_end_ix - self.ctx_len
		if ctx_start < 0 : ctx_start = 0 #cant go below zero
		ctx = self.buf[ ctx_start : self.ctx_end_ix ]

		#first prediction is ZERO
		if ctx.size == 0 : return 0

		self.scores = {} #clean up

		#where to look for match
		if self.include_ctx :
			past = self.buf[ : self.ctx_end_ix ]
		else:
			past = self.buf[ : ctx_start ]

		#states which match PRESENT
		match_idxs = np.where(past == self.last)[0]

		for i in match_idxs : #for all matches check for matching ctx three-nsitions in the past

			start = i - self.ctx_len
			match = self.buf[ start : i ]
#			print match, ctx
			if match.size == ctx.size :
				score = (match == ctx).sum() - 1 #by default ith position match already, so -1
			else : score = 0

			future_pred_ix = i + predict_steps
			if future_pred_ix < self.buffer_len : future = self.buf[future_pred_ix]
			else :
				break #out of bounds of buffer, no score increase

			#stupid Python, doesn't support autovivification
			if future in self.scores :
				self.scores[future] += score #increase STATE prediction counter
			else:
				self.scores[future] = 1

		#find the future state with the highest score
#		print self.scores
		pred = 0
		if len(self.scores.keys()) :
			pred = max(self.scores, key=self.scores.get)
		#little cheating here ;)
		if pred == '' : pred = self.sig[-1] #if no prediction use the last signal-value
		return pred


	def train(self, data, predict_steps=None, log=False):
		pred = self.predict(predict_steps=predict_steps)
		data = self.learn(data)

		self.sig.append(data)
		self.yhat.append(float(pred))
		if log and len(self.yhat) > 0 : print "predicted : %s => %s : %s" % (self.sig[-1] , self.yhat[-1], self.sig[-1] - self.yhat[-1])


	#Predict n-steps in the future
	def forward(self, steps=1):
		rv = []
		for f in xrange(1, steps):
			rv.append( self.predict(predict_steps=f) )

		return rv

	#Predict on predict mode
	def ppredict(self, steps=1):
		rv = []
		for p in xrange(1,steps):
			pred = self.predict(predict_steps=1)
			rv.append(pred)
			self.learn(pred)

		return rv

	#combine prediction methods, doesnt work so far
	def pf_predict(self, steps=10):
		rv = []
		avg = 0
		forward = self.forward(steps)
		for i in xrange(1,steps):
			pred = self.predict(predict_steps=1)
			delta = (forward[i-1] - pred)
			avg = (avg + delta) / 2.0
			res = forward[i-1] + avg
			rv.append(res)
			self.learn(res)

		return rv


	def batch_train(self,sig):
		self.orig = np.concatenate([[0], sig]) #keep original signal around
		for s in sig : self.train(s)

	def plot(self, shift_it=False, original=True):
		plt.figure()
		shift = shift = [0] * (self.predict_steps - 1) if shift_it else []
		if original : plt.plot(self.orig, color='yellow')
		plt.plot(self.sig)
		plt.plot( shift + self.yhat )
		plt.tight_layout()

	def stats(self, skip=None, original=False):
		if skip is None : skip = self.ctx_len
		data = None
		if original :
			data = np.array(self.orig)[skip:]
		else :
			data = np.array(self.sig)[skip:]

		yhat = np.array(self.yhat + [0])[skip:]

		print "==== MODEL stats =============="
		print "mape: ",  stats.mape(data,yhat)
		print "mae : ", stats.mae(data,yhat)
		print "rmse: ",  stats.rmse(data,yhat)
		print "r2: ",  stats.r2(data,yhat)
		print "nll: ",  stats.nll(data,yhat)



	def encode(self, values): #to limit the number of states
#		if isinstance(values,list) : values = np.array(values)
		#the values are centered in the middle of the resolution
		#  f.e. res=5 will generate : 2.5, 7.5, 12.5, 17.5 ...
		return ((values / self.resolution ) * self.resolution) + (self.resolution/2.0)

	def decode(self, values) :
		if isinstance(values,list) : values = np.array(values)
		pass

