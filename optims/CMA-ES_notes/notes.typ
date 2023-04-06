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

// Reference style
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

// Proof environment
#let proof(content) = {
  [_Proof._ $space$] + content + align(right, text()[$qed$])
}

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

#bibliography("refs.bib")

= Annex
== $mu_"eff"$<annex_weights>
#proof([
Let $w_i = (mu - i + 1) / (sum_(i=1)^mu mu - i + 1)$. Then:
$ 
mu_"eff" &= 1 / (sum_(i=1)^mu ((mu - i + 1) / (sum_(i=1)^mu mu - i + 1 ))^2) = 1 / (sum_(i=1)^mu (i / (sum_(i=1)^mu i))^2) \
           &= 1 / (sum_(i=1)^mu i^2 / (mu(mu+1)/2)^2) = 1 / ( (mu(mu+1)(2mu+1)) / (6 (mu(mu+1)/2)^2) ) \
           &= (6 (mu(mu+1)/2)^2) / (mu(mu+1)(2mu+1)) = (3mu (mu^3 + 2mu^2 + mu)) / (2 (2mu^3 + 3mu^2 + mu)) \
           &= (3mu (1 + mu)) / (2 (1 + 2mu)) approx ( 3 lambda/2 (1 + lambda/2) ) / (2 (1 + lambda)) \
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