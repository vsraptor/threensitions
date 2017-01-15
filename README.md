# threensitions

What is three-nsition ? Glad you asked :)

In two words it is a way to do prediction of Time Series. Normally you would use Markov chains or Autoregressive models to do that. Three-nstion is sort of Markov chain with a twist.

It is sort of Higher order Markov chain where we have only 3 Super states : PAST => PRESENT => FUTURE.

  -  The PAST (p) super-state is a placeholder for a multiple states that happened in the past just before the present : p1, p2, p3, ..., pl where L is the number of past states we will consider for the calculation.
  -  The PRESENT|NOW (n) super-state is the fulcrum of the system. We use it as mechanism to lower the amount of information we need to store for the model.
  -  The FUTURE (f) super-state is the prediction and dependent only on state n.

You can read the documentation [Threensition](http://ifni.co/threensition.html) or the docs directory of this repository.