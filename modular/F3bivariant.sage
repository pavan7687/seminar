#!/usr/bin/env sage

from itertools import product

# Field
F = GF(3)

# Polynomial ring in two variables
R.<x,y> = PolynomialRing(F)

# ---- input degree ----
d = int(input("Enter degree d: "))

# ---- build list of monomials with i+j ≤ d ----
monoms = []
for i in range(d+1):
    for j in range(d+1):
        if i + j <= d:
            monoms.append((i,j))

print("Number of monomials in f:", len(monoms))

# ---- sparsity function ----
def sparsity(poly):
    return len(poly.dict())

solutions = []
count = 0

# coefficients 1 or 2 (complete polynomial)
for coeffs in product([1,2], repeat=len(monoms)):

    f = 0

    for k,(i,j) in enumerate(monoms):
        f += F(coeffs[k]) * x^i * y^j

    f_terms = sparsity(f)

    sq = f^2
    sq_terms = sparsity(sq)

    if sq_terms < f_terms:

        print("\nFound sparse square:")
        print("f(x,y) =", f)
        print("f(x,y)^2 =", sq)
        print("terms in f:", f_terms)
        print("terms in f^2:", sq_terms)

        solutions.append(f)

    count += 1

print("\nTotal polynomials checked:", count)
print("Total sparse squares found:", len(solutions))
