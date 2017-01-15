import numpy as np
import matplotlib.pylab as plt
from stats import stats

'''
	Self Organized Map (SOM)

'''

class SOM:

	def __init__(self, nstates=20, vmin=0, vmax=100, initial_states=None, radiusp=0.1,  learn_steps=500, learn_rate=0.1):
		self.vmin = vmin
		self.vmax = vmax
		self.nstates = nstates

		if initial_states is not None :
			assert len(initial_states) == nstates
			self.states = np.array(initial_states)
			self.map_states = np.array(initial_states)#keep orig for plotting
		else :
			self.states = np.linspace(self.vmin, self.vmax, nstates)
			self.map_states = np.linspace(self.vmin, self.vmax, nstates)

		self.radiusp = radiusp
		#which of the neigboors of best match are adjusted too
		self.learn_radius = int(self.nstates * self.radiusp)
		assert self.learn_radius >= 2
		self.learn_steps = learn_steps #training until
		self.tick = 0

		self.learn_rate = learn_rate


	#calculate Euclidian distance
	def euclidean_distance(self, data):
		vector = np.zeros(self.nstates) + data
		dist = np.sqrt((vector - self.states)**2)
		return dist

	def rate(self, x0, current=None, total=None): #decaying rate
		if total is None : total = self.learn_steps
		if current is None : current = self.tick

		if 0 < x0 < 1 :#learning rate
			rv = x0 * np.exp(- current / float(total))
		else:#learning radius
			coef = total / float(np.log(x0))
			#print 'coef:' , coef
			rv = x0 * np.exp( - current / coef )
		return rv

	def gauss(self, dist, radius):
		return np.exp(- np.power(dist,2)/ (2*(float(radius)**2)) )

	def learn(self, data) :
		self.tick += 1
		print data

		lradius = int( self.rate(self.learn_radius) )
		ldia = 1 if lradius == 0 else lradius * 2
		lrate = self.rate(self.learn_rate)
		print "ldia: ", ldia
		print 'lrate: ', lrate

		#find the best matching value indexes
		best = np.argsort( self.euclidean_distance(data) )[: ldia]
		best_ix = best[0]
		delta = (self.states[best] - data)
		decay_rate =  self.gauss(np.arange(ldia), ldia) * lrate
		print "bix: ", best_ix
		print 'd:' , delta
		print 'g: ', decay_rate
		print 'final: ', decay_rate * delta
		amt = decay_rate * delta

		print self.states[best]
		self.states[best] -= decay_rate * delta
		print self.states[best]

		return best_ix

	def encode(self, data):
		best_ix = self.learn(data)
		return best_ix

	def decode(self, data):
		return self.states[data]

	def plot_map(self):
	#	plt.figure()
		plt.plot(self.map_states, self.states)

	# and colllect stats change over time
	def plot_trail(self, xs):
		plt.cla()
		self.mae = []
		self.rmse = []
		ys = []
		for x in xs :
			ys.append( self.decode( self.encode(x) ) )
			lys = len(ys)
			self.rmse.append( stats.rmse(xs[:lys],ys) )
			self.mae.append( stats.mae(xs[:lys],ys) )

		plt.scatter(xs,xs, color='green')
		plt.scatter(xs,ys, color='red')


