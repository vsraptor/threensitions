import numpy as np
import matplotlib.pylab as plt

'''
	3D-Threensition State machine i.e. the transition array supports 3 variables.
	This is a research project, because 3 dimention array grows too fast.
	Also predicting more than one n-step in the future will need adding more dimentions.

	Numpy/scipy does not support 3D sparse arrays, so to make this to work
	for more states I will need to rely on other structure, 3 level deep tree may be.

'''

class TDSM :


	def __init__(self, nstates=10, zero_based=True):
		self.nstates = nstates
		self.zero_based = zero_based
		self.smap = np.zeros((nstates, nstates, nstates), dtype=np.uint16)

	#sequence format : past, now, future i.e.  pn, pn-1, ...., p3, p2, p1, n, f
	def learn(self, seq, log=False):
		assert len(seq) >= 3
		if log : print seq
		now = seq[-2] - 1 if self.zero_based else seq[-2]
		future = seq[-1] - 1 if self.zero_based else seq[-1]
		for past in seq[:-2] :
			if self.zero_based : past -= 1
			if log : print past+1, now+1, future+1
			self.smap[ past, now, future ] += 1

	#batch process ...
	def roll_seq(self,seq, window=3):
		assert len(seq) >= 3
		for start in xrange(len(seq)-2) : #at least 3 elems required
			#print seq[start : start + window ]
			self.learn(seq[ start : start+window ])

	#slice across the NOW|PRESENT dimention and pick the max, depending on the context
	def predict(self, now, ctx):
		#[:,N,:]  =2d=> rows: Past, cols: Future
		# Sum by cols will give us the winner i.e. best prediction
		if self.zero_based :
			future = self.smap[ctx-1,now-1,:].sum(axis=0).argmax() #!fixme : random if multiple winners
			return future + 1
		else :
			future = self.smap[ctx,now,:].sum(axis=0).argmax() #!fixme : random if multiple winners
			return future


	#don't use it
	def to_dict(self):
		self.ary = {}
		ixs = np.where(self.smap > 0)
		for i in xrange(len(ixs[0])) :
			self.ary[ (ixs[0][i], ixs[1][i], ixs[2][i]) ] = 1


	def info(self):
		print "======== TDSM ================="
		print "MLL: %s" % self.smap.max()
		print "MLL ix: (past, present, future) %s" % ( np.unravel_index( self.smap.argmax(), self.smap.shape), )
		sparsity = (self.smap > 0).sum() / float(np.power(self.nstates,3))
		print "sparsity: %.2f%%" % (sparsity * 100)
		mem = self.smap.nbytes / (1024*1024.)
		print "mem: %.2f MB" % mem
		return {'sparsity' : sparsity, 'mem' : mem }



