\name{open_modbart}
\title{Open a BART object for continuous outcomes}
\alias{open_modbart}
\description{
BART is a Bayesian \dQuote{sum-of-trees} model.\cr
For a numeric response \eqn{y}, we have
\eqn{y = f(x) + \epsilon}{y = f(x) + e}
where \eqn{\epsilon \sim N(0,\sigma^2)}{e ~ N(0,sigma^2)}.\cr

This function returns an object that can be
used to sample \eqn{f} given \eqn{y,\sigma}.

\eqn{f} is the sum of many tree models.
The goal is to have very flexible inference for the uknown
function \eqn{f}.

In the spirit of \dQuote{ensemble models},
each tree is constrained by a prior to be a weak learner
so that it contributes a
small amount to the overall fit.
}
\usage{
open_modbart=function(
  x,
  offset=0.0,#how much to offset the y data by
  sparse=FALSE, theta=0, omega=1,
  a=0.5, b=1, augment=FALSE, rho=NULL,
  xinfo=matrix(0.0,0,0), usequants=FALSE,
  cont=FALSE, rm.const=TRUE,
  k=2.0, power=2.0, base=.95,
  sigmaf=NA,
  ntree=200L, numcut=100L,
  transposed=FALSE)
}
\arguments{
   \item{x}{
   Explanatory variables. Must be a matrix with (as usual) rows corresponding
   to observations and columns to variables.\cr
   }
   \item{offset}{
     How much to center future \eqn{y} values when conditioning.
     For example, if \eqn{y} is given, the \code{sample_modbart} conditions
     on \eqn{y-offset}.
   }
   \item{sparse}{Whether to perform variable selection based on a
     sparse Dirichlet prior rather than simply uniform; see Linero 2016.}
   \item{theta}{Set \eqn{theta} parameter; zero means random.}
   \item{omega}{Set \eqn{omega} parameter; zero means random.}
   \item{a}{Sparse parameter for \eqn{Beta(a, b)} prior:
     \eqn{0.5<=a<=1} where lower values inducing more sparsity.}
   \item{b}{Sparse parameter for \eqn{Beta(a, b)} prior; typically,
     \eqn{b=1}.}
   \item{rho}{Sparse parameter: typically \eqn{rho=p} where \eqn{p} is the
     number of covariates under consideration.}
   \item{augment}{Whether data augmentation is to be performed in sparse
     variable selection.}
   \item{xinfo}{ You can provide the cutpoints to BART or let BART
     choose them for you.  To provide them, use the \code{xinfo}
     argument to specify a list (matrix) where the items (rows) are the
     covariates and the contents of the items (columns) are the
     cutpoints.  }
   \item{usequants}{ If \code{usequants=FALSE}, then the
    cutpoints in \code{xinfo} are generated uniformly; otherwise,
    if \code{TRUE}, uniform quantiles are used for the cutpoints. }
   \item{cont}{ Whether or not to assume all variables are continuous.}
   \item{rm.const}{ Whether or not to remove constant variables.}
   \item{k}{
   For numeric y,
   k is the number of prior standard deviations \eqn{E(Y|x) = f(x)} is away from +/-.5.
   The response (y.train) is internally scaled to range from -.5 to .5.
   %% For binary y,
   k is the number of prior standard deviations \eqn{f(x)} is away from +/-3.
   %In both cases,
   The bigger k is, the more conservative the fitting will be.
   }
   \item{power}{
   Power parameter for tree prior.
   }
   \item{base}{
   Base parameter for tree prior.
   }
   \item{sigmaf}{
   The SD of f.
   }
   \item{ntree}{
   The number of trees in the sum.
   }
   \item{numcut}{
   The number of possible values of c (see usequants).
   If a single number if given, this is used for all variables.
   Otherwise a vector with length equal to ncol(x.train) is required,
   where the \eqn{i^{th}}{i^th} element gives the number of c used for
   the \eqn{i^{th}}{i^th} variable in x.train.
   If usequants is false, numcut equally spaced cutoffs
   are used covering the range of values in the corresponding
   column of x.train.  If usequants is true, then  min(numcut, the number of unique values in the
   corresponding columns of x.train - 1) c values are used.
   }
   \item{transposed}{
   When running \code{wbart} in parallel, it is more memory-efficient
   to transpose \code{x.train} and \code{x.test}, if any, prior to
   calling \code{mc.wbart}.
   }

   %% \item{treesaslists}{
   %%   The default behavior is to return the trees as a text string.  If you
   %%   specify \code{treesaslists=TRUE}, then the trees will be returned
   %%   as a list as well.
   %% }
}
\details{
  This function returns an object that can be
  used to sample \eqn{f|y,\sigma}.
}
\value{
  This function returns an object that can be passed into \code{sample_modbart} and
  convert_modbart.
}
\references{
}
\author{
Daniel Bourgeois: \email{dcb10@rice.edu}.
}
\seealso{
\code{\link{sample_modbart}}
\code{\link{convert_modbart}}
}
\examples{
sampler = function(x, y, xtest, n_burn, n_post)
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
