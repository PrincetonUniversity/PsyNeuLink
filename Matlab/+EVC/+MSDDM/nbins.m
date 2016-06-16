function N = nbins(x)
n = length(x);
N = (range(x)*n^(1/3)) / (2 * iqr(x));

