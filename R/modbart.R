open_modbart=function(
x,
#ndpost,
offset=0.0,#how much to offset the y data by
sparse=FALSE, theta=0, omega=1,
a=0.5, b=1, augment=FALSE, rho=NULL,
xinfo=matrix(0.0,0,0), usequants=FALSE,
cont=FALSE, rm.const=TRUE,
sigest=NA, sigdf=3, sigquant=.90,
k=2.0, power=2.0, base=.95,
sigmaf=NA, lambda=NA,
w=NA, ntree=200L, numcut=100L,
transposed=FALSE)
{
#------------------------------------------------------------------------------
if(!transposed) {
    temp = bartModelMatrix(x, numcut, usequants=usequants,
                           cont=cont, xinfo=xinfo, rm.const=rm.const)
    x = t(temp$X)
    numcut = temp$numcut
    xinfo = temp$xinfo
    rm.const <- temp$rm.const
    grp <- temp$grp
    rm(temp)
}
else {
    rm.const <- NULL
    grp <- NULL
}

n = ncol(x)
p = nrow(x)
if(length(rho)==0) rho=p
if(length(rm.const)==0) rm.const <- 1:p
if(length(grp)==0) grp <- 1:p

#------------------------------------------------------------------------------

nu=sigdf
if(is.na(lambda)) {
   if(is.na(sigest)) {
     sigest = 1.0 # MOD
   }
   qchi = qchisq(1.0-sigquant,nu)
   lambda = (sigest*sigest*qchi)/nu #lambda parameter for sigma prior
} else {
   sigest=sqrt(lambda)
}

if(is.na(sigmaf)) {
   tau=3.0/(2*k*sqrt(ntree)) # MOD
} else {
   tau = sigmaf/sqrt(ntree)
}


if(is.na(w))
  w = rep(1,n)
stopifnot(length(w) == n)

#------------------------------------------------------------------------------
res = .Call("copen_modbart",
            n,  #number of observations in training data
            p,  #dimension of x
            x,   #pxn training data x
            #ndpost,
            ntree,
            numcut,
            offset,
            power,
            base,
            tau,
            nu,
            lambda,
            sigest,
            w,
            sparse,
            theta,
            omega,
            grp,
            a,
            b,
            rho,
            augment,
            xinfo)
return(res)
}

# MOD add documentation when this is done
sample_modbart = function(
  object,
  y,
  save_draw = TRUE,
  steps = 1,
  start_dart = TRUE)
{
  if(!object[["dart"]])
    start_dart = FALSE

  y_offset = y - object[["offset"]]

  res = .Call("csample_modbart",
              object,
              save_draw,
              y_offset,
              steps,
              start_dart)

  # returns "sigma", "out", "saved"
  # where sigma is the variance term and
  # out is bart(x)
  return(res);
}

convert_modbart = function(object)
{
  ret = .Call("cconvert_modbart",
              object)
  ret[["mu"]] = c() # needed for call to predict to use object[["offset"]]
  ret[["offset"]] = object[["offset"]]

  attr(ret, "class") <- c("converted_modbart", "wbart")
  return(ret)
}

#close_modbart = function(object)
#{
#  res = .Call("cclose_modbart",
#              object)
#  return()
#}

predict.converted_modbart = function(object, x, silent = TRUE)
{
  class(object) <- "wbart"
  if(silent)
    capture.output(ret <- predict(object, x))
  else
    ret <- predict(object, x)

  return(ret)
}


