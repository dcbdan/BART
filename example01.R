library(BART, lib.loc = "~/ps/BART/bld")
library(coda)

f_quantile = function(x){ quantile(x, probs = c(0.025, 0.975)) }

nN = 10000
nP = 3
true_s2e = 0.01
u = array(rnorm(nN*nP), dim = c(nN,nP))
v = array(rnorm(nN*nP), dim = c(nN,nP))

true_beta1 = ifelse(sqrt(u[,1]*u[,1]) < 0.7, -3, 3)
true_beta2 = 3*sin(2*v[,1])

y = rnorm(nN, true_beta1 + true_beta2, sd = sqrt(true_s2e))

n_burn = 1000
n_post = 1000

utest = cbind(
  seq(-2, 2, length.out = 100),
  array(rnorm(100*(nP-1)), dim = c(100, nP-1)))
vtest = cbind(
  seq(-2, 2, length.out = 100),
  array(rnorm(100*(nP-1)), dim = c(100, nP-1)))

true_utest = ifelse(sqrt(utest[,1]*utest[,1]) < 0.7, -3, 3)
true_vtest = 3*sin(vtest[,1])


#######################################################
# plot functions
plot_errors = function(post_mean, true_mean, plot_mean = FALSE)
{
  hpds = t(apply(post_mean, 2, f_quantile))
  lower = hpds[,1] - true_mean
  upper = hpds[,2] - true_mean

  nn = length(true_mean)
  if(nn > 1000)
  {
    lower = lower[1:1000]
    upper = upper[1:1000]
    true_mean = true_mean[1:1000]
  }

  plot(1:length(true_mean), lower, ylim = range(c(lower, upper)), type = "l",
       col = "green", lwd = 3, main = "errors")
  lines(1:length(true_mean), upper, col = "darkgreen", lwd = 3)
  abline(h = 0, lwd = 3)
}

plot_f = function(x, f, post_f)
{
  hpds = t(apply(post_f, 2, f_quantile))
  lower = hpds[,1]
  upper = hpds[,2]

  plot(x, f, ylim = range(hpds, f), pch = 19, lwd = 3, main = "f(x)")
  points(x, lower, pch = 19, lwd = 3, col = "green")
  points(x, upper, pch = 19, lwd = 3, col = "darkgreen")
}

#######################################################

#######################################################
# run straight bart
bt = wbart(cbind(u, v), y, nskip = n_burn, ndpost = n_post, numcut = 5, ntree=15)

par(mfrow = c(1,3))
plot_errors(predict(bt, cbind(u, v)), true_beta1 + true_beta2)
plot_errors(predict(bt, cbind(utest, vtest)), true_utest + true_vtest)
plot_f(utest[,1], true_utest + true_vtest, predict(bt, cbind(utest, vtest)))
#######################################################

#######################################################
# run sampler with two barts -- one for u, one for v
sampler = function(u, v, y, n_burn, n_post)
{
  beta1_modbart = open_modbart(u, sigest = 1.0, k = 2.0, numcut = 10, ntree = 5,
                               cont = TRUE, usequants = TRUE)
  beta2_modbart = open_modbart(v, sigest = 1.0, k = 2.0, numcut = 10, ntree = 20,
                               cont = TRUE, usequants = TRUE)

  s2e = var(y)
  beta1 = rep(0, nN)
  beta2 = rep(0, nN)

  post_beta1 = array(dim = c(n_post, nN))
  post_beta2 = array(dim = c(n_post, nN))
  post_s2e   = array(dim = c(n_post))

  is_good_out = function(x){ !is.nan(x) && !is.na(x) }

  for(idx in 1:(n_burn+n_post)){
    take_sample = idx > n_burn

    beta1 = sample_modbart(beta1_modbart, y - beta2, save_draw = take_sample)$out
    beta2 = sample_modbart(beta2_modbart, y - beta1, save_draw = take_sample)$out

    stopifnot(all(is_good_out(beta1)))
    stopifnot(all(is_good_out(beta2)))

    z = y - beta1 - beta2
    s2e = 1/rgamma(1, 0.1 + nN/2, 0.1 + 0.5*sum(z*z))

    if(take_sample){
      mcmc_idx = idx - n_burn
      post_beta1[mcmc_idx,] = beta1
      post_beta2[mcmc_idx,] = beta2
      post_s2e[mcmc_idx] = s2e
    }

    if(idx %% 100 == 0){
      print(paste0(idx, " out of ", n_burn + n_post))
    }
  }

  eval_beta1 = convert_modbart(beta1_modbart)
  eval_beta2 = convert_modbart(beta2_modbart)

  return(list(
    "post_beta1"=post_beta1,
    "post_beta2"=post_beta2,
    "post_s2e"=post_s2e,
    "eval_beta1"=eval_beta1,
    "eval_beta2"=eval_beta2))
}

out = sampler(u, v, y, n_burn, n_post)
post_beta1 = out$post_beta1
post_beta2 = out$post_beta2
post_s2e = out$post_s2e
eval1 = out$eval_beta1
eval2 = out$eval_beta2

par(mfrow = c(2,2))
plot(post_s2e, type = "l")
plot_errors(post_beta1 + post_beta2, true_beta1 + true_beta2)
plot_f(utest[,1], true_utest, predict(eval1, utest))
plot_f(vtest[,1], true_vtest, predict(eval2, vtest))


