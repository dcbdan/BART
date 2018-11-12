\name{sample_modbart}
\title{Sample and update a BART object for continuous outcomes}
\alias{sample_modbart}
\description{
  Sample and update a BART object for continuous outcomes.

  BART is a Bayesian \dQuote{sum-of-trees} model.\cr
  For a numeric response \eqn{y}, we have
  \eqn{y = f(x) + \epsilon}{y = f(x) + e},
  where \eqn{\epsilon \sim N(0,\sigma^2)}{e ~ N(0,sigma^2)}.\cr

  This function samples \eqn{f|y, \sigma}.
}
\usage{
sample_modbart = function(
  object,
  y,
  sig,
  save_draw = TRUE,
  steps = 1,
  start_dart = TRUE)
}
\arguments{
  \item{object}{mod BART object as returned by \code{open_modbart}.}
  \item{y}{set of data to condition on. Must be the same length as the number of rows
    of the covariate data stored in the \code{object}.
  }
  \item{sig}{standard deviation to condition on. This can be either length 1 or
    the length of the data y.}
  \item{save_draw}{If true, then the trees contained within \code{object} will
    be saved for use by \code{convert_modbart}. Otherwise, the tree values
    will not be saved. \code{save_draw} should be false during burn in.}
  \item{steps}{The number of times the tree values are updated. Only
    one tree value will potentially be saved.}
  \item{start_dart}{Whether to start the Dirichlet Prior for variable selection.
    Sometimes it helps to get the chain started by leaving this value to false
    during the burn in. If \code{object} wasn't initialized with the Dirichlet Prior
    or if \item{start_dart} has been set to true in a previous call to this function,
    there is no effect.}
}
\details{
  This function updates the mod BART object and returns a sample from \eqn{f(x)}.
}
\value{
  Returns a sample from \eqn{f(x)}.
}
\references{
}
\author{
Daniel Bourgeois: \email{dcb10@rice.edu}.
}
\seealso{
\code{\link{open_modbart}}
\code{\link{convert_modbart}}
}
\examples{
sampler = function(x, y, n_burn, n_post)
{
  beta1_modbart = open_modbart(x, k = 2.0, numcut = 10, ntree = 5)

  s2e = var(y)/10
  beta1 = rep(0, nN)

  post_beta1 = array(dim = c(n_post, nN))
  post_s2e   = array(dim = c(n_post))

  is_good_out = function(x){ !is.nan(x) && !is.na(x) }

  for(idx in 1:(n_burn+n_post)){
    take_sample = idx > n_burn

    beta1 = sample_modbart(
      beta1_modbart, y, sqrt(s2e), save_draw = take_sample)

    stopifnot(all(is_good_out(beta1)))

    z = y - beta1
    s2e = 1/rgamma(1, 0.1 + nN/2, 0.1 + 0.5*sum(z*z))

    if(take_sample){
      mcmc_idx = idx - n_burn
      post_beta1[mcmc_idx,] = beta1
      post_s2e[mcmc_idx] = s2e
    }

    if(idx \%\% 100 == 0){
      print(paste0(idx, " out of ", n_burn + n_post))
    }
  }

  eval_beta1 = convert_modbart(beta1_modbart)

  beta_test = predict(eval_beta1, xtest)

  return(list(
    "post_beta1"=post_beta1,
    "post_s2e"=post_s2e,
    "eval_beta1"=eval_beta1,
    "beta_test"=beta_test))
}
}
\keyword{nonparametric}
\keyword{tree}
\keyword{regression}
\keyword{nonlinear}

