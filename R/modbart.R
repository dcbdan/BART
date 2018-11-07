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
{
#------------------------------------------------------------------------------
if(!transposed) {
    temp = bartModelMatrix(x, numcut, usequants=usequants,
                           cont=cont, xinfo=xinfo, rm.const=rm.const)
    x = t(temp$X)
    numcut = temp$numcut
    xinfo = temp$xinfo
    rm.const <- temp$rm.const
    rm(temp)
}
else {
    rm.const <- NULL
}

n = ncol(x)
p = nrow(x)
if(length(rho)==0) rho=p
if(length(rm.const)==0) rm.const <- 1:p

#------------------------------------------------------------------------------

if(is.na(sigmaf)) {
   tau=3.0/(2*k*sqrt(ntree)) # MOD
} else {
   tau = sigmaf/sqrt(ntree)
}

#------------------------------------------------------------------------------
res = .Call("copen_modbart",
            n,  #number of observations in training data
            p,  #dimension of x
            x,   #pxn training data x
            ntree,
            numcut,
            offset,
            power,
            base,
            tau,
            sparse,
            theta,
            omega,
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
  sig,
  save_draw = TRUE,
  steps = 1,
  start_dart = TRUE)
{
  n = object[["n"]]

  if(!object[["dart"]])
    start_dart = FALSE

  y_offset = y - object[["offset"]]

  if(length(sig) != n)
    sig = rep(1, n)*sig

  res = .Call("csample_modbart",
              object,
              sig,
              save_draw,
              y_offset,
              steps,
              start_dart)

  # returns sampled values for bart(x)
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


