/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
#include "heterbart.h"

#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)

RcppExport SEXP copen_modbart(
   SEXP _in,            //number of observations in training data
   SEXP _ip,		//dimension of x
   SEXP _ix,		//x, train,  pxn (transposed so rows are contiguous in memory)
   SEXP _im,		//number of trees
   SEXP _inc,		//number of cut points
   SEXP _ioffset,
   SEXP _ipower,
   SEXP _ibase,
   SEXP _itau,
   SEXP _idart,
   SEXP _itheta,
   SEXP _iomega,
   SEXP _ia,
   SEXP _ib,
   SEXP _irho,
   SEXP _iaug,
   SEXP _Xinfo
)
{
   //--------------------------------------------------
   //process args
   size_t n = Rcpp::as<int>(_in);
   size_t p = Rcpp::as<int>(_ip);
   Rcpp::NumericVector  xv(_ix);
   double *ix = &xv[0];
   size_t m = Rcpp::as<int>(_im);
   Rcpp::IntegerVector _nc(_inc);
   double offset = Rcpp::as<double>(_ioffset);
   int *numcut = &_nc[0];
   double mybeta = Rcpp::as<double>(_ipower);
   double alpha = Rcpp::as<double>(_ibase);
   double tau = Rcpp::as<double>(_itau);

   bool dart;
   if(Rcpp::as<int>(_idart)==1) dart=true;
   else dart=false;
   double a = Rcpp::as<double>(_ia);
   double b = Rcpp::as<double>(_ib);
   double rho = Rcpp::as<double>(_irho);
   bool aug;
   if(Rcpp::as<int>(_iaug)==1) aug=true;
   else aug=false;
   double theta = Rcpp::as<double>(_itheta);
   double omega = Rcpp::as<double>(_iomega);
   Rcpp::NumericMatrix Xinfo(_Xinfo);

   heterbart* raw_ptr_bm = new heterbart(m);
   Rcpp::XPtr<heterbart> ptr_bm = Rcpp::XPtr<heterbart>(raw_ptr_bm, true);
   heterbart& bm = *ptr_bm;

   if(Xinfo.size()>0) {
     xinfo _xi;
     _xi.resize(p);
     for(size_t i=0;i<p;i++) {
       _xi[i].resize(numcut[i]);
       //Rcpp::IntegerVector cutpts(Xinfo[i]);
       for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
     }
     bm.setxinfo(_xi);
   }

   //--------------------------------------------------
   //heterbart bm(m);
   bm.setprior(alpha,mybeta,tau);
   bm.setdata_without_y(p,n,ix,numcut);
   bm.setdart(a,b,rho,aug,dart,theta,omega);

   //--------------------------------------------------

   // string stream to write trees to
   Rcpp::XPtr<std::vector<tree> > ptr_trees =
     Rcpp::XPtr<std::vector<tree> >(new std::vector<tree>(), true);

   //--------------------------------------------------
   //return

   Rcpp::List ret;
   ret["bm"]=ptr_bm;
   ret["n"]=n;
   ret["p"]=p;
   ret["trees"] = ptr_trees;
   ret["offset"] = offset;
   ret["dart"] = dart;
   return ret;
}

RcppExport SEXP csample_modbart(
    SEXP _iobject,
    SEXP _isvec,
    SEXP _isave_draw,
    SEXP _iy,
    SEXP _isteps,
    SEXP _istart_dart)
{
  Rcpp::List object = Rcpp::as<Rcpp::List>(_iobject);

  Rcpp::XPtr<heterbart> ptr_bm = Rcpp::as<Rcpp::XPtr<heterbart> >(object["bm"]);
  heterbart* raw_ptr_bm = ptr_bm.checked_get();
  heterbart& bm = *raw_ptr_bm;

  size_t n = Rcpp::as<int>(object["n"]);

  Rcpp::NumericVector _svec(_isvec);
  double* tmp_svec = &_svec[0];
  double* svec = new double[n];
  for(size_t idx = 0; idx != n; ++idx){
    svec[idx] = tmp_svec[idx];
  }

  Rcpp::NumericVector  yv(_iy);
  double *iy = &yv[0];

  size_t steps = Rcpp::as<int>(_isteps);

  bool save_draw;
  if(Rcpp::as<int>(_isave_draw)==1) save_draw=true;
  else save_draw=false;

  if(Rcpp::as<int>(_istart_dart)==1)
    bm.startdart();

  Rcpp::XPtr<std::vector<tree> > ptr_trees =
    Rcpp::as<Rcpp::XPtr<std::vector<tree> > >(object["trees"]);
  std::vector<tree>& trees = *ptr_trees;

  // Return data structure
  Rcpp::NumericVector out(n);

  //-----------------------------------------------------------------

  bm.sety(iy);

  arn gen; // MOD make sure setting random number generator
    // in R works as expected

  xinfo& xi = bm.getxinfo();

  for(size_t step_idx=0; step_idx<steps; ++step_idx)
  {
    // draw bart
    bm.draw(svec,gen);
  }

  if(save_draw)
  {
    size_t m = bm.getm();
    for(size_t j=0;j<m;j++)
    {
      trees.push_back(bm.gettree(j));
    }
  }

  for(size_t k=0;k<n;k++)
  {
    out[k]=bm.f(k);
  }

  return out;
}

RcppExport SEXP cconvert_modbart(
    SEXP _iobject)
{
  Rcpp::List object = Rcpp::as<Rcpp::List>(_iobject);

  Rcpp::XPtr<heterbart> ptr_bm = Rcpp::as<Rcpp::XPtr<heterbart> >(object["bm"]);
  heterbart* raw_ptr_bm = ptr_bm.checked_get();
  heterbart& bm = *raw_ptr_bm;

  Rcpp::XPtr<std::vector<tree> > ptr_trees =
    Rcpp::as<Rcpp::XPtr<std::vector<tree> > >(object["trees"]);
  std::vector<tree> const& trees = *ptr_trees;

  size_t p = Rcpp::as<int>(object["p"]);

  xinfo const& xi = bm.getxinfo();
  Rcpp::List xiret(xi.size());
  for(size_t i=0;i<xi.size();i++) {
    Rcpp::NumericVector vtemp(xi[i].size());
    std::copy(xi[i].begin(),xi[i].end(),vtemp.begin());
    xiret[i] = Rcpp::NumericVector(vtemp);
  }

  std::stringstream treess;
  treess.precision(10);
  treess << trees.size() << " " << bm.getm() << " " << p << endl;
  for(tree const& a_tree: trees)
  {
    treess << a_tree;
  }

  Rcpp::List treesL;
  treesL["cutpoints"] = xiret;
  treesL["trees"] = Rcpp::CharacterVector(treess.str());

  Rcpp::List ret;
  ret["treedraws"] = treesL;

  return ret;
}

//RcppExport SEXP cclose_modbart(
//    SEXP _iobject)
//{
//  Rcpp::List object = Rcpp::as<Rcpp::List>(_iobject);
//
//  Rcpp::XPtr<heterbart> ptr_bm = Rcpp::as<Rcpp::XPtr<heterbart> >(object["bm"]);
//  heterbart* raw_ptr_bm = ptr_bm.checked_get();
//  heterbart& bm = *raw_ptr_bm;
//
//  Rcpp::XPtr<double> ptr_svec = Rcpp::as<Rcpp::XPtr<double> >(object["svec"]);
//
//  Rcpp::XPtr<double> ptr_iw = Rcpp::as<Rcpp::XPtr<double> >(object["iw"]);
//
//  Rcpp::XPtr<std::vector<tree> > ptr_trees =
//    Rcpp::as<Rcpp::XPtr<std::vector<tree> > >(object["trees"]);
//
//  //Rcpp::XPtr<std::stringstream> ptr_treess =
//  //  Rcpp::as<Rcpp::XPtr<std::stringstream> >(object["treess"]);
//  //std::stringstream& treess = *ptr_treess;
//
//  Rcpp::XPtr<size_t> ptr_save_idx =
//    Rcpp::as<Rcpp::XPtr<size_t> >(object["save_idx"]);
//  size_t ndpost = *ptr_save_idx;
//
//  //template typename<T>
//  //using vecvec = std::vector<std::vector<T> >;
//
//  //Rcpp::XPtr<vecvec<double> > ptr_varprb =
//  //  Rcpp::as<Rcpp::XPtr<vecvec<double> > >(object["varprb"])
//  //vecvec<double>& varprb = *ptr_varprb;
//
//  //Rcpp::XPtr<vecvec<size_t> > ptr_varcnt =
//  //  Rcpp::as<Rcpp::XPtr<vecvec<size_t> > >(object["varcnt"])
//  //vecvec<size_t>& varcnt = *ptr_varcnt;
//
//  // delete the pointers since no use now
//  ptr_bm.release();
//  ptr_svec.release();
//  ptr_iw.release();
//  ptr_trees.release();
//  //ptr_treess.release();
//  ptr_save_idx.release();
//  //ptr_varprb.release();
//  //ptr_varcnt.release();
//
//  return Rcpp::List();
//}
