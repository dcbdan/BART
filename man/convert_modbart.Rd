\name{convert_modbart}
\title{Return }
\alias{convert_modbart}
\description{
  Create an object that will give posterior samples.
}
\usage{
convert_modbart = function(object)
}
\arguments{
  \item{object}{mod BART object as returned by \code{open_modbart}. It is assumed that
    \code{object} has been called at least once with \code{sample_modbart} function
    where \code{save_draw} was true. }
}
\details{
  This function creates an object that contains the posterior parameters of a BART model.
  This is the same type of object as returned by \code{wbart}
}
\value{
  Returns an object that contains the posterior parameters of a BART model.
}
\references{
}
\author{
Daniel Bourgeois: \email{dcb10@rice.edu}.
}
\seealso{
\code{\link{sample_modbart}}
\code{\link{open_modbart}}
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
