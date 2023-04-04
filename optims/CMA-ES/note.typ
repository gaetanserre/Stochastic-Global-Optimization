#set page(
  paper: "a4",
  header: align(right)[
    CMA-ES
  ],
  numbering: "1",
)

#set par(justify: true)
#set heading(numbering: "1.")

#align(center, text(16pt)[
  *Covariance Matrix Adaptation - Evolution Strategy: \
  An summary*
])

#align(center, text(14pt)[
  Gaëtan Serré \
  ENS Paris-Saclay - Centre Borelli \
  #link("mailto:gaetan.serre@ens-paris-saclay.fr")
])

= Introduction
The #emph[Covariance Matrix Adaptation - Evolution Strategy] (CMA-ES)
is a global and black-box optimization algorithm.
It is a randomized black-box search algorithm as it samples
evaluation points from a distribution conditionned with the previous parameters:



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
)

CMA-ES uses a multivariate Gaussian as the sampling distribution $cal(N)(m, C)$. The authors made
this choice as, given the variances and covariances between components, the
normal distribution has the largest entropy in $RR^d$.
To goal is to find how update the mean and covariance matrix of this distribution
to minimize the trade-off between finding a good approximation of the optimum
and evaluate the objective function as few times as possible.
In this small paper, we will present the ideas behind CMA-ES.
For more details, see @hansen_cma_2023.

= Update the mean
#emph[Select the $mu$ best points and do a weighted average of their coordinates. The choice of $mu$, $w$ and $sigma$ is crucial.
The quantity $mu_("eff")$ is an indicator on the loss of variance due to the selection.]

= Update the covariance matrix
#emph[Unbiased estimator using empirical mean vs using real mean updated before and only the selected point.
The second one redirects the search towards those points and turns out to be more efficient. Rank-$mu$ update?]

#bibliography("refs.bib")