#import "paper.typ": *

#show: doc => config(
  title: [*Covariance Matrix Adaptation - Evolution Strategy: \A summary*],
  doc
)

= Introduction
The _Covariance Matrix Adaptation - Evolution Strategy_ (CMA-ES)
is a global and black-box optimization algorithm.
It is a randomized black-box search algorithm as it samples
evaluation points from a distribution conditionned with the previous parameters.
This kind of algorithm is detailed in @black-box-search.
CMA-ES uses a multivariate Gaussian as the sampling distribution $cal(N)(m, C)$. The authors made
this choice as, given the variances and covariances between components, the
normal distribution has the largest entropy in $RR^d$.
To goal is to find how update the mean and covariance matrix of this distribution
to minimize the trade-off between finding a good approximation of the optimum
and evaluate the objective function as few times as possible.
In this small paper, we will present the ideas behind CMA-ES.
For more details, see @hansen_cma_2023.
Throughout this paper, we will suppose that the objective function is
to be maximized.

#figure(
  align(left, text(style: "italic")[
  For $g$ in $1 dots k$:
    + Let $d_theta$ a distribution on $RR^d$ parametrized by $theta$;

    + Sample $lambda$ points: $(x_i)_(1 lt.eq i lt.eq lambda) dash.wave d_(theta_i)$;

    + Evaluate the points: $f((x_i)_(1 lt.eq i lt.eq lambda))$;

    + Update the parameters $theta_(i+1) = F(theta_i, (x_1, f(x_1)), dots, (x_lambda, f(x_lambda)))$.]),
  caption: [
    Black-box search algorithm.
  ],
  kind: "algorithm",
  supplement: "Algorithm"
)<black-box-search>

= Update the mean<mean>
In the CMA Evolution Strategy, the $lambda$ points are sampled from a multivariate Gaussian distribution
which writes:

$ (x^((g))_i)_(1 lt.eq i lt.eq lambda ) dash.wave m^((g)) + sigma^((g)) cal(N)(0, C^((g))), $

where $g$ is the generation number, $m^((g))$ is the mean vector, $sigma^((g))$ is the "overall" standard deviation and
$C^((g))$ is the covariance matrix. It is equivalent to say that $x^((g))_i dash.wave cal(N)(m^((g)), (sigma^((g)))^2 C^((g)) )$. \
To update the mean, we begin by selecting the $mu$ best points, i.e.:

$ f(x^((g))_1) gt.eq dots gt.eq f(x^((g))_mu) gt.eq f(x^((g))_(mu+1)) gt.eq dots f(x^((g))_lambda). $

We introduce the index notation $i : lambda$, denoting the index of the $i$-th best point.
The mean at generation $g+1$ becomes a weighted average of those points:
#let i_lam = $i : lambda$

$ m^((g+1)) = m^((g)) + c_m sum_(i=1)^mu w_i(x_#i_lam - m^((g))), $<mean-update>

where:

$ sum_(i=1)^mu w_i = 1, quad w_1 gt.eq dots gt.eq w_mu gt.eq 0, $<weights>

and $c_m$ is a learning rate, usually set to $1$.
In that case, @mean-update simply becomes:

$ m^((g+1)) = sum_(i=1)^mu w_i x_#i_lam. $

The choice of the weights is crucial in CMA-ES as they represent the trade-off between
exploration and exploitation.
To do so, we define the quantity $mu_"eff"$ as:

$ mu_"eff" = (norm(w)_1 / norm(w)_2)^2 = (sum_(i=1)^mu abs(w_i))^2 / (sum_(i=1)^mu w_i^2) = 1 / (sum_(i=1)^mu w_i^2). $

From @weights, one can easily derive $1 lt.eq mu_"eff" lt.eq mu$, the latter happens when all the weights are equal,
i.e. $forall 1 lt.eq i lt.eq mu, w_i = 1/mu$.
$mu_"eff"$ quantize the loss of variance due to the selection of the best points.
According to the author, $mu_"eff" approx lambda/4$ indicates a reasonable choice of $w_i$.
A simple and decent way to achieve that is to set $w_i prop mu - i + 1$ (see @annex_weights).
Choosing $c_m < 1$ can work well on noisy function. However, the step-size $sigma$
is roughly proportional to $1/c_m$ and thus, with a too small $c_m$, the search
would moves away from the current region of relevance.

= Update the covariance matrix
To update the covariance matrix, we need to estimate it using
the points $(x_i)_(1 lt.eq i lt.eq lambda)$. In this section, we assume $sigma = 1$ for simplicity.
If $sigma eq.not 1$, one can simply rescale the covariance matrix by $1 / sigma^2$.
If we have enough sample, one can use the empirical covariance matrix:

$ C_"emp"^((g+1)) = 1 / (lambda - 1) sum_(i=1)^lambda (x_i^((g+1)) - 1/lambda sum_(i=1)^lambda x_i^((g+1)) )
                                                      (x_i^((g+1)) - 1/lambda sum_(i=1)^lambda x_i^((g+1)) )^top. $

A different would be to use the real mean $m^((g+1))$ computed before instead of the empirical mean:

$ C_lambda^((g+1)) =  1 / lambda sum_(i=1)^lambda (x_i^((g+1)) - m^((g+1)) )
                                                        (x_i^((g+1)) - m^((g+1)) )^top. $

Both are unbiaised estimators of the covariance matrix.
However, they do not influence the search towards the direction of the $mu$ best points.
To do so, one can use the same _weighted selection_ as in @mean-update:

$ C_mu^((g+1)) =  sum_(i=1)^mu w_i (x_#i_lam^((g+1)) - m^((g)) )
                                                        (x_#i_lam^((g+1)) - m^((g)) )^top. $<c_mu>
                                                        
This last estimator tends to reproduce the current best points and thus allows a faster convergence.
However, this estimation method requires a lot of samples for $mu_"eff"$ must be large enough to be reliable.
The author suggests another method to estimate $C^((g+1))$ that tackles these two issues, the _rank_-$mu$ method.

== Rank-$mu$ method
As stated before, for @c_mu to be a reliable estimator, one need a lot of sample, as ideally $mu_"eff" approx lambda/4$~.
However, evaluate the function is often the main bottleneck of the algorithm. The author suggests to use the _rank_-$mu$ method
to estimate the covariance matrix. The idea behind this method is to use information of previous generations in the
estimation of the next one:

$ C^((g+1)) = 1 / (g+1) sum_(i=0)^(g) 1 / sigma^(i)^2 C_mu^((i+1)), $<prev_info>

where $sigma^((i))$ is the step-size at generation $i$.
In @prev_info, each generation has the same weight in the estimation of the covariance matrix of the next generation.
A natural idea would be to give more weight to the most recent generations through exponential smoothing:

$ C^((g+1)) = (1 - c_mu) C^((g)) + c_mu 1 / sigma^(g)^2 C_mu^((g+1)), $<exp_smoothing>

where $c_mu lt.eq 1$ is a learning rate. The author suggests that $c_mu approx min(1, mu_"eff"/d^2)$ is a reasonable choice.
@exp_smoothing can be written as:

$ C^((g+1)) = (1 - c_mu) C^((g)) + c_mu sum_(i=1)^mu w_i y_#i_lam^((g+1)) y_#i_lam^(g+1)^top, $<exp_smoothing2>

where $y_#i_lam^((g+1)) = ( x_#i_lam^((g+1)) - m^((g)) ) / sigma^((g))$.
This method is so called _rank_-$mu$ as the rank of the sum of the dot products is $min(mu, d)$.
Finally, the author generalizes @exp_smoothing2 with $lambda$ weights that does not requires to sum
to $1$ nor being positive:

$ C^((g+1)) = (1 - sum_(i=1)^lambda w_i c_mu) C^((g)) + c_mu sum_(i=1)^lambda w_i y_#i_lam^((g+1)) y_#i_lam^(g+1)^top. $

Usually, $sum_(i=1)^mu w_i = 1 = - sum_(i=mu+1)^lambda w_i$.
To emphasize the importance of $c_mu$, the author introduce the _backward time horizon_ $Delta g$.
It represent the number of generations used to encode roughly $63%$ of the information of
the estimation of the covariance matrix of the next generation. E.g. if $Delta g eq 10$, it means
that the $10$ last generations are used to compute $63%$ of the information of the covariance matrix
of the next generation. Indeed, @exp_smoothing can be extended to:

$ C^((g+1)) = (1 - c_mu)^((g+1)) C^((0)) + c_mu sum_(i=0)^g (1 - c_mu)^(g - i) 1/sigma^(i)^2 C_mu^((i+1)). $

Therefore, $Delta g$ is defined by:

$ c_mu sum_(i=g+1 - Delta g)^g (1 - c_mu) approx 0.63 approx 1 - 1/e. $

One can solve this equation to find $Delta g approx 1/c_mu$ (see @annex_delta_g).
It shows that, the smaller is $c_mu$ the more past generations are used to compute the covariance matrix.

== Rank-1 method
#let o_lam = $1 : lambda$
In the previous section, we defined a method that uses information of the entire population to update the covariance matrix. The author present the _rank-1_ method that uses only the best point of the population to update the covariance matrix. This method highlights the correlation between generations and the final update rule of CMA-ES uses both _rank_-$mu$ and _rank-1_ methods.
The idea of the _rank-1_ method is simply to use only the best point to update the covariance matrix in order to increase the likelihood of reproducing this point in the next generation. The update rule is just an adaption of @exp_smoothing2 where only $y_#o_lam^((g+1))$ is used:

$ 
C^((g+1)) = (1 - sum_(i=1)^lambda w_i c_1) C^((g)) + c_1 y_#o_lam^((g+1)) y_#o_lam^(g+1)^top,
 $
 
where, according to the author, $c_1 approx 2 / n^2$ is a good choice.
This method however, discards the sign information of $y_#o_lam^((g+1))$ as

$ y_#o_lam^((g+1)) y_#o_lam^(g+1)^top = -y_#o_lam^((g+1)) (-y_#o_lam^(g+1)^top). $

To reintroduce this information in the estimation of the covariance matrix, the author suggests to build a _evolution path_ $p_c^((g))$:

$ p_c^((g+1)) = (1 - c_c) p_c^((g)) + sqrt(c_c (2 - c_c) mu_"eff") (sum_(i=1)^mu w_i ( x_i^((g+1)) - m^((g)) )) / sigma^((g)), $

where $c_c lt.eq 1$.
This represent a exponential smoothing of the sum:

$ sum_(g) (sum_(i=1)^mu w_i ( x_i^((g+1)) - m^((g)) )) / sigma^((g)). $<p_c_update>

We set $p_c^((0)) = 0$ and $sqrt(c_c (2 - c_c) mu_"eff")$ is a normalization
constant in order to $p_c^((g)) dash.wave y_#o_lam^((g)) dash.wave cal(N)(0, C^((g)))$ (see @annex_p_c).
The final update rule of the covariance matrix of CMA-ES combine @exp_smoothing2 and @p_c_update:

$  
C^((g+1)) = (1 underbrace(- c_1 - c_mu sum_(i=1)^lambda w_i, approx 0) ) C^((g)) + c_1 underbrace(p_c^((g+1)) p_c^((g+1))^top, "rank-"1" update") + c_mu underbrace(sum_(i=1)^lambda w_i y_#i_lam^((g+1)) y_#i_lam^(g+1)^top, "rank-"mu" update"),
 $

where 
- $c_1 approx 2/n^2$,

- $c_mu approx min(1 - c_1, mu_"eff"/n^2)$,

- $y_#i_lam^((g+1)) = (x_#i_lam^((g+1)) -m^((g)) ) / sigma^((g))$,

- $sum_(i=1)^lambda w_i approx -c_1 / c_mu$.

= Choice of the step-size
The last parameter to choose is the step-size $sigma^((g))$. The author suggests to control
the step-size using the evolution of the direction of the steps, represented by $y_#i_lam^((g))$.
One can dinstinguish three cases:
+ the steps goes in the same direction, meaning that the optimum is likely to be in this direction and increasing the step-size allows to reach the optimum faster;
+ the steps cancel each other, meaning the the optimum is likely to be in the interior of the region "drawn" by the steps, and decreasing the step-size allows to explore that region;
+ the steps follows almost orthogonal directions, meaning that the algorithm follows the contour lines of the objective function and the step-size must be kept constant.
Theses cases are illustrated in @step-size_evolution-path.
One way to infer in which case steps are is to use the length of the evolution path $p_sigma^((g))$:

$ 
p_sigma^((g)) = sum_g (sum_(i=1)^mu w_i ( x_i^((g)) - m^((g-1)) )) / sigma^((g-1)).
 $

As before, the author used an exponential smoothing to compute $p_sigma^((g))$:

$ 
p_sigma^((g+1)) = (1 - c_sigma) p_sigma^((g)) + sqrt(c_sigma (2 - c_sigma) mu_"eff") C^(-1/2) (sum_(i=1)^mu w_i ( x_i^((g+1)) - m^((g)) )) / sigma^((g)).
 $

Using the same reasoning as in @annex_p_c, one can show that, if $p_sigma^((0)) dash.wave cal(N)(0, I)$, $p_sigma^((g)) dash.wave cal(N)(0, I)$. This way, one can compare the length of the path represented by $norm(p_sigma^((g)))$ to its expected value and state on how update $sigma^((g))$.
The author suggests to update $sigma^((g))$ as follow:

$ sigma^(g+1) = sigma^((g)) exp( c_sigma / d_sigma ( norm(p_sigma^(g+1)) / (EE [ norm(cal(N)(0, I))]) - 1 ) ), $
where $d_sigma approx 1$ and $EE[norm(cal(N)(0, I))] approx sqrt(d)$.
The author provide default values for all parameters of CMA-ES, summarized in @default_parameters.

#figure(
  image("figures/step-size_evolution-path.png"),
  caption:[_Left_: Steps cancels each other, step-size must be decreased. _Right_: Steps are correlated and goes to the same direction, step-size must be increased. _Middle_: Steps follows almost orthogonal directions, step-size must be kept constant.]
)<step-size_evolution-path>


#bibliography("refs.bib")

= Appendix

== Default parameters
#let br_d_l = $bracket.l.double$
#let br_d_r = $bracket.r.double$
#figure(
  table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon,
  [Parameters], [Default value],
  [$lambda$], [$4 + floor(3 ln d)$],
  [$mu$], [$ceil(lambda / 2)$],
  [$w_i'$], [$ln (lambda + 1) / 2 - ln i$, $forall i in #br_d_l 1 dots lambda #br_d_r$],
  [$mu^(-)_"eff"$], [$(sum_(i = mu+1)^lambda w_i')^2 / (sum_(i = mu+1)^lambda w_i'^2)$],
  [$alpha_"cov"$], [$2$],
  [$c_mu$], [$min( 1 - c_1, alpha_"cov" (1/4 + mu_"eff" + 1 / mu_"eff" - 2) / ((n+2)^2 + alpha_"cov" mu_"eff" / 2)) $],
  [$c_1$], [$alpha_"cov" / ( (n + 1.3)^2 + mu_"eff" )$],
  [$c_c$], [$(4 + mu_"eff" / n) / (n + 4 + 2mu_"eff" / n)$],
  [$alpha^(-)_mu$], [$1 + c_1/c_mu$],
  [$alpha^(-)_(mu_"eff")$], [$1 + (2 mu^(-)_"eff") / (mu_"eff" + 2)$],
  [$alpha^(-)_"pos def"$], [$(1 - c_1 - c_mu)/d c_mu$],
  [$w_i$], [$cases( 1 / (sum_j abs(w_j')^+) w_i' &"if" w_i' gt.eq 0 "(sum to 1)", (min(alpha_mu^-, alpha_(mu_"eff")^-, alpha_"pos def"^-) / (sum_j abs(w_j')^-)) &"if" w_i' < 0 ) $],
  [$c_sigma$], [$( mu_"eff" + 2 ) / ( d + mu_"eff" + 5 )$],
  [$d_sigma$], [$1 + 2 max(0, sqrt( (mu_"eff" - 1) / (d + 1) ) - 1 )$],
  [$c_m$], [$1$],
),
  caption:[Default parameters of CMA-ES.],
)<default_parameters>

== $mu_"eff"$<annex_weights>
#proof([
Let $w_i = (mu - i + 1) / (sum_(i=1)^mu mu - i + 1)$. Then:
$ 
mu_"eff" &= 1 / (sum_(i=1)^mu ((mu - i + 1) / (sum_(i=1)^mu mu - i + 1 ))^2) = 1 / (sum_(i=1)^mu (i / (sum_(i=1)^mu i))^2) \
           &= 1 / (sum_(i=1)^mu i^2 / (mu(mu+1)/2)^2) = 1 / ( (mu(mu+1)(2mu+1)) / (6 (mu(mu+1)/2)^2) ) \
           &= (6 (mu(mu+1)/2)^2) / (mu(mu+1)(2mu+1)) = (3mu (mu^3 + 2mu^2 + mu)) / (2 (2mu^3 + 3mu^2 + mu)) \
           &= (3mu (1 + mu)) / (2 (1 + 2mu)) approx ( (3 lambda)/2 (1 + lambda/2) ) / (2 (1 + lambda)) \
           &= ((3 lambda^2 + 6lambda) / 2 ) / (4 (1 + lambda)) = (3 lambda (2 + lambda)) / (8 (1 + lambda)) \
           &approx (3 lambda) / 8.
 $
])

== $Delta g$<annex_delta_g>
#proof([
$ 
c_mu sum_(i=g+1 - Delta g)^g (1 - c_mu)^(g - i) &= c_mu ( (1 - c_mu)^(Delta g - 1) + (1 - c_mu)^(Delta g - 2) + dots + (1 - c_mu)^0 ) \
& = c_mu sum_(i=0)^(Delta g - 1) (1 - c_mu)^i \
&= c_mu ((1 - c_mu)^0 - (1 - c_mu)^(Delta g)) / (1 - (1 - c_mu)) \
& = c_mu (1 - (1 - c_mu)^(Delta g)) / c_mu = 1 - (1 - c_mu)^(Delta g).
 $
Thus, the problem becomes to find $Delta g$ such that $1 - (1 - c_mu)^(Delta g) approx 0.63 approx & 1 - 1/e$:

$ 
&1 - (1 - c_mu)^(Delta g) = 1 - 1/e \
arrow.l.r.double & (1 - c_mu)^(Delta g) = e^(-1) \
arrow.l.r.double & Delta g ln(1 - c_mu) = -1 \
arrow.l.r.double & Delta g = -1 / ln(1 - c_mu) approx 1 / c_mu ("using Taylor's expansion of order 1").
 $
])

== $p_c^((g)) dash.wave cal(N)(0, C)$<annex_p_c>
#proof([
Under the assumption that $p_c^((g)) dash.wave cal(N)(0, C^((g)))$, we want to prove that $p_c^((g+1)) dash.wave cal(N)(0, C^((g)))$:
$ 
p_c^((g+1)) =& (1 - c_c) p_c^((g)) + sqrt(c_c (2 - c_c) mu_"eff") (sum_(i=1)^mu w_i ( x_i^((g+1)) - m^((g)) )) / sigma^((g)) \
dash.wave& (1 - c_c) cal(N)(0, C^((g))) + sqrt(c_c (2 - c_c) mu_"eff") (sum_(i=1)^mu w_i cal(N)(0, C^((g))) ) \
dash.wave& cal(N)(0, (1 - c_c)^2 C^((g))) + sqrt(c_c (2 - c_c) mu_"eff") cal(N)(0, sum_(i=1)^mu w_i^2 C^((g)) ) \
dash.wave& cal(N)(0, (1 - c_c)^2 C^((g))) + sqrt(c_c (2 - c_c) mu_"eff") cal(N)(0, 1/mu_"eff" C^((g)) ) \
dash.wave& cal(N)(0, (1 - c_c)^2 C^((g))) + sqrt(c_c (2 - c_c) mu_"eff") 1/sqrt(mu_"eff") cal(N)(0, C^((g)) ) \
dash.wave& cal(N)(0, (1 - c_c)^2 C^((g))) + cal(N)(0, c_c (2 - c_c) C^((g)) ) \
dash.wave& cal(N)(0, ((1 - c_c)^2 + c_c (2 - c_c)) C^((g))) \
dash.wave& cal(N)(0, (1 - 2c_c + c_c^2 + 2c_c - c_c^2) C^((g))) \
dash.wave& cal(N)(0, C^((g))).
 $
])