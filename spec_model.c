/* spec_model.c */

#include <stdio.h>
#include <math.h>

double L_l( double x, int l){
	double result;
	
	if(l==0){
		result = 1;
	}

	else if(l==2){
		result = 0.5 * ( 3. * pow(x,2.) - 1. );
	}
	
	else if(l==4){
		result = ( 1. / 8. ) * ( 35 * pow(x,4) - 30 * pow(x,2) + 3);
	}
	else{
		printf(" The multipole of order %d is not implemented \n ", l);
	}

	return result;	
}

double P(int n, double* x){
	/* 
	 * mu -> Here we will pass the values of mu on which it will
	 * 	be integrated.
	 *
	 * In the user_data vector we'll pass additional arguments.
	 * k      -> 0 -> physical k at which it will be evaluated
	 * bias   -> 1 -> bias of the tracer
	 * f      -> 2 -> matter growth-rate
	 * sigma  -> 3 -> factor controlling the FoG modeling
	 * l      -> 4 -> integer controlling order of multipole
	 */

	double result, k, bias, f, sigma, ll;
	int l;
	
	k     = x[1];
	bias  = x[2];
	f     = x[3];
	sigma = x[4];
	ll    = x[5];

	l = (int) ll;

	result = pow( ( bias + f * pow(x[0],2) ), 2) * exp( - pow( k * sigma * x[0], 2) ) * L_l( x[0], l ) ;

	return result;
}
