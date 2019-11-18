#include <math.h>
#include <stdlib.h>
#include "blackbox.h"
 
static const double eps = 1e-13;
 
double tau(double *rk, int n, double *tmp) {
    // Ar_k
    blackbox_mult(rk, tmp);
    double top = 0;
    double bot = 0;
    for (int i=0; i<n; ++i) {
        top += tmp[i] * rk[i];
        bot += tmp[i] * tmp[i];
    }
    double t = top / bot;
    // Update r_k+1
    for (int i=0; i<n; ++i) {
        rk[i] -= t * tmp[i];
    }
    return t;
}
 
double residual(const double *b, const double *x0, const int n, const double t, double *x1) {
    double ret = 0;
    blackbox_mult(x0, x1);
    for (int r = 0; r < n; ++r) {
        x1[r] = x0[r] + t*(b[r] - x1[r]);
        double tmp = x1[r] - x0[r];
        ret += tmp*tmp;//fabs(x1[r] - x0[r]);
    }
    // printf("%.12f\n", sqrt(ret));
    return sqrt(ret);
}
 
int main() {
    blackbox_init();
    const int n = blackbox_size();
    double *b = (double *) malloc(n*sizeof(double));
    double *x = (double *) malloc(n*sizeof(double));
    double *d = (double *) malloc(n*sizeof(double));
    double *r = (double *) malloc(n*sizeof(double));
    double t = 1;
    blackbox_rhs(b);
    for (int k = 0; k < n; ++k) {
        x[k] = 0;
        r[k] = -0.1 * b[k];
    }
 
    while(residual(b, x, n, t, d) > eps) {
        double *swap = x;
        x = d;
        d = swap;
        // Find tau
        t = tau(r, n, d);
    }
    blackbox_submit(x);
    return 0;
}
