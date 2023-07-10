# SpaceSpanningNetwork

## The problem
We are trying to maximize the range of MLPs (tanh), at fixed topology. The range of a network is defined as the set of points it "typically" outputs given a gaussian input vector. We are interested in the cases where the input dimension is much less than the output dimension.

There are many possible measures of how good a certain range is. I opted for the following formal definition of the problem. We are looking for the set of weights and biases $\tilde{p}$ , that given :

#### $NN_p :  \mathbb{R}^{d_{in}} \rightarrow  \mathbb{R}^{d_{out}}, \quad  d_{in} \ll d_{out} \newline$<br> 
#### $Y \hookrightarrow \mathcal{N}(0,1)^{d_{out}} \quad$<br>
 
#### $` n \in \mathbb{N}^*, \quad \forall i \in [1, n] \quad X_i \hookrightarrow \mathcal{N}(0,1)^{d_{in}}, \quad (X_i)_i \quad i.i.d.\newline `$<br>
Satisfies:<br>
### $` \tilde{p} \quad  =  \quad argmin_p[ \quad E_{Y}( \quad E_{(X_1,..X_n)}[ \quad min_{i\in[1,n]}( \quad ||NN_p(X_i) - Y||_2 \quad )])]`$

<br>

We will refer to the quantity in the $argmin_p$ as $SF(p)$, the Space Fillig capacity of the networks with parameters $p$. Even though it depends on $n$, it will not appear explicitly. The same value of $n$ will be used in the code when two experiences are compared.

This expression means that for a *likely* $Y \in  \mathbb{R}^{d_{out}}$, the network $NN_{\tilde{p}}$  frequently outputs vectors close to $Y$ given *likely* inputs $X \in  \mathbb{R}^{d_{in}}$. The purpose of $n$ is to have both a "good coverage" of space with the $(X_1,..X_n)$, but also not to cover "too much" of it so that a vector $Y$ difficult to reach for the network appears as such in the measure. Taking $n = \infty$ provides good insights.  

This investigation was motivated by the fact that as the number of layers of a randomly initialized* MLP grows, the output range shrinks, and quickly collapses to near 0 (whatever the measure). This can be somewhat mitigated by zeroing the biases, but it is not sufficient. Explored methods are described in the next section.

 (*) Random initialization like pytorch's default for instance. Example in MLP_range.py 

## Results

Two machine learning techniques are used: gradient descent on a single network and evolutionnary algorithms on a population. An estimation of $SF(p)$ is needed in both cases, which is obtained as follows using the code's terminology:

- A score is initialized at 0.
- N_YS independent gaussian vectors are generated.
- For each one of those vectors Y, N_XS independent random gaussian vectors are generated. The minimum of the distances between Y and the NN(X)s is subtracted to the score.
- Once we have been through all Ys, the approximation is obtained by dividing the score by N_YS.

The consistency `(and probably the accuracy as it is unbiased (?), TODO check)` of this estimator is evaluated at each run by computing its variance over several measurements. Note that the expectation over the $X_i$, $E_{(X_1,..X_n)}$, is estimated with only one sample.

Since the MLP's activation is tanh, its range is included within $[-1, 1]^{d_{out}}$. To compensate, the components of the gaussian vectors sampled in the output space (the Ys) are scaled by .3 and then clamped to [-1, 1].

### A- Genetic algorithm

* Technicalities

Specimens are simply parameters sets. Sparse gaussian auto-regulatory mutations and sparse combinations, using RECURSIVE_NODES phylogenetic tracking. Selection is the 2 factor ranking technique used in RECURSIVE_NODES. 

A fixed dataset of $N$ couples $(X,Y)$, randomly (i.i.d.) generated with a normal distribution, is used for the entirety of the evolution process. The fitness of a specimen $p$ is:\
<br>
$f_p = -\frac{1}{N}\sum_{(X,Y) \in dataset}{||NN_p(X) - Y||_2}\newline$
<br>
* Observations

As the average fitness over the population increases, we observe a decrease of $SF(p^*)$, $p^*$ being the fittest specimen at this step ( and also probably of the average $SF$ but is is expensive to compute accurately.).


### B- Gradient descent

We generate fixed dataset of $N$ couples $(X,Y)$, i.i.d. gaussian vectors, and apply gradient descent (full batch) to a randomly initialized network. 
As for the evolutionary process, we observe that as the loss on the dataset decreases, so does $SF$. 


### C- Comparison
 
Zeroing the biases at initialization kickstarts the convergence for both methods but does not improve the end result.

The final $SF$ are similar with GD and GA, but the GA is much slower. Performances on the proxy task were much better with GD, but that is irrelevant here.

Those results are satisfactory, but in the lack of theoretical studies of $SF$ I do not know if this is close to the lower bound for $SF$, with a given set of hyperparameters and this inference structure (tanh MLP).