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

# Variables to track the absolute sparsest f^2
min_sq_terms = float('inf')
best_solutions = []

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
        solutions.append(f)
        
        # --- NEW LOGIC: Track the minimum terms in f^2 ---
        if sq_terms < min_sq_terms:
            # We found a new record! Update the minimum and reset the list
            min_sq_terms = sq_terms
            best_solutions = [(f, sq, f_terms, sq_terms)]
        elif sq_terms == min_sq_terms:
            # We found a tie! Add it to the list of current winners
            best_solutions.append((f, sq, f_terms, sq_terms))

    count += 1

# --- PRINTING THE RESULTS ---
print("\nTotal polynomials checked:", count)
print("Total sparse squares found:", len(solutions))

if best_solutions:
    print(f"\n=== WINNERS: Polynomials with the LEAST terms in f^2 ({min_sq_terms} terms) ===")
    for best_f, best_sq, best_f_terms, best_sq_terms in best_solutions:
        print("\nf(x) =", best_f)
        print("f(x)^2 =", best_sq)
        print("terms in f:", best_f_terms)
        print("terms in f^2:", best_sq_terms)
else:
    print("\nNo sparse squares found for this degree.")
