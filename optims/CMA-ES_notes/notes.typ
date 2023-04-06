#set page(
  paper: "a4",
  header: align(right)[
    CMA-ES
  ],
  numbering: "1",
)

#set par(justify: true)
#set heading(numbering: "1.")
#set cite(style: "chicago-author-date")
#set math.equation(numbering: "(1)")
#set ref(supplement: it => {
  let fig = it.func()
  if fig == math.equation {
    "Eq."
  } 
  
  else if fig == figure {
    let kind = it.kind
    if kind == "algorithm" {
      "Algorithm"
    } else {
      "Figure"
    }
  }
})

#align(center, text(16pt)[
  *Covariance Matrix Adaptation - Evolution Strategy: \
  A summary*
])

#align(center, text(14pt)[
  Gaëtan Serré \
  ENS Paris-Saclay - Centre Borelli \
  #link("mailto:gaetan.serre@ens-paris-saclay.fr")
])

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
    + Let $d_theta$ a distribution on $cal(X)$ parametrized by $theta$;

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
However, this estimation method requires a lot of samples and $mu_"eff"$ must be large enough to be reliable.
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
$ C^((g+1)) = (1 - c_mu) C^((g)) + c_mu sum_(i=1)^mu w_i y_#i_lam^((g+1)) y_#i_lam^(g+1)^top, $
where $y_#i_lam^((g+1)) = ( x_#i_lam^((g+1)) - m^((g)) ) / sigma^((g))$.


#bibliography("refs.bib")

#pagebreak()
= Annex
Let $w_i = (mu - i + 1) / (sum_(i=1)^mu mu - i + 1)$. Then:
$ 
mu_"eff" &= 1 / (sum_(i=1)^mu ((mu - i + 1) / (sum_(i=1)^mu mu - i + 1 ))^2) = 1 / (sum_(i=1)^mu (i / (sum_(i=1)^mu i))^2) \
           &= 1 / (sum_(i=1)^mu i^2 / (mu(mu+1)/2)^2) = 1 / ( (mu(mu+1)(2mu+1)) / (6 (mu(mu+1)/2)^2) ) \
           &= (6 (mu(mu+1)/2)^2) / (mu(mu+1)(2mu+1)) = (3mu (mu^3 + 2mu^2 + mu)) / (2 (2mu^3 + 3mu^2 + mu)) \
           &= (3mu (1 + mu)) / (2 (1 + 2mu)) approx ( 3 lambda/2 (1 + lambda/2) ) / (2 (1 + lambda)) \
           &= ((3 lambda^2 + 6lambda) / 2 ) / (4 (1 + lambda)) = (3 lambda (2 + lambda)) / (8 (1 + lambda)) \
           &approx (3 lambda) / 8.
 $<annex_weights>