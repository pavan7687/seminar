# Sparse Square Search

This repository contains SageMath scripts to search for **polynomials whose squares are sparse** using computations over the finite field **F3 = {0,1,2}**.

A polynomial has a **sparse square** if the number of non-zero terms in `f^2` is **less than** the number of terms in `f`.

## 1. Univariate Polynomial Search

Searches for sparse squares of polynomials

```
f(x) = a0 + a1*x + ... + ad*x^d
```

where `ai ∈ {1,2}`.

The program:

* takes the degree `d` as input
* generates all possible polynomials
* computes `f(x)^2`
* checks if the square is sparse

## 2. Bivariate Polynomial Search

Searches for sparse squares of polynomials

```
f(x,y) = sum a(i,j) * x^i * y^j   where i + j <= d
```

over `F3`.

The program:

* generates monomials with `i + j <= d`
* constructs polynomials with coefficients in `{1,2}`
* computes `f(x,y)^2`
* detects sparse squares

## Requirement

* SageMath

Run using:

```
sage filename.sage
```
