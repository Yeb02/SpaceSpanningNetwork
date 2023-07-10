# SpaceSpanningNetwork

## The problem
We are trying to maximize the range of MLPs (tanh), at fixed topology. The range of a network is defined as the set of points it "typically" outputs given a gaussian input vector. We are interested in the cases where the input dimension is much less than the output dimension.

There are many possible measures of how good a certain range is. I opted for the following formal definition of the problem. We are looking for the set of weights and biases $\tilde{p}$ , that given :

#### $NN_p :  \mathbb{R}^{d_{in}} \rightarrow  \mathbb{R}^{d_{out}}, \quad  d_{in} \ll d_{out} \newline$<br> 
#### $Y \hookrightarrow \mathcal{N}(0,1)^{d_{out}} \newline$<br>
#### $` n \in \mathbb{N}^*, \quad \forall i \in [1, n] \quad X_i \hookrightarrow \mathcal{N}(0,1)^{d_{in}}, \quad (X_i)_i \quad i.i.d.\newline `$<br>
Satisfies:<br>
### $` \tilde{p} \quad  =  \quad argmin_p[ \quad E_{Y}( \quad E_{(X_1,..X_n)}[ \quad min_{i\in[1,n]}( \quad ||NN_p(X_i) - Y||_2 \quad )])]`$

<br>

This means that for a likely $Y \in  \mathbb{R}^{d_{out}}$, the network $NN_{\tilde{p}}$  frequently outputs vectors close to $Y$ given likely inputs $X \in  \mathbb{R}^{d_{in}}$. The integer $n$ is used in the code to limit the computational costs when sampling to estimate expected values. 

This investigation was motivated by the fact that as the number of layers of a randomly initialized* MLP grows, the output range shrinks. This can be somewhat mitigated by zeroing the biases, but it is not sufficient.

 (*) Random initialization like pytorch's default for instance. Example in MLP_range.py 

## Results

Two machine learning techniques are used: gradient descent on a single network and evolutionnary algorithms on a population. 


