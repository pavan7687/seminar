#!/usr/bin/env sage

from itertools import product

# field F3
F = GF(3)
R.<x> = PolynomialRing(F)

# -------- input degree --------
d = int(input("Enter degree d: "))

# -------- sparsity function --------
def sparsity(poly):
    return len(poly.dict())

solutions = []
count = 0

# generate all complete polynomials
for coeffs in product([1,2], repeat=d+1):

    # build polynomial
    f = sum(F(coeffs[i]) * x^i for i in range(d+1))

    f_terms = sparsity(f)

    # compute square
    sq = f^2

    sq_terms = sparsity(sq)

    # check sparsity condition
    if sq_terms < f_terms:

        print("\nFound sparse square:")
        print("f(x) =", f)
        print("f(x)^2 =", sq)
        print("terms in f:", f_terms)
        print("terms in f^2:", sq_terms)

        solutions.append(f)

    count += 1

print("\nTotal polynomials checked:", count)
print("Total sparse squares found:", len(solutions))
