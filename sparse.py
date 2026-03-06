#!/usr/bin/env sage
"""
Sparse squares of complete polynomials – Abbott (2000) implementation.
Run with: sage this_file.sage
"""

import itertools, time
from math import comb

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def count_terms(poly):
    return sum(1 for c in poly if c != 0)

def poly_square(coeffs):
    n = len(coeffs)
    res = [0] * (2*n - 1)
    for i in range(n):
        for k in range(n):
            res[i+k] += coeffs[i] * coeffs[k]
    return res

def mirror_subset(J, d):
    return tuple(sorted(2*d - j for j in J))

def reduce_by_symmetry_c(subsets, d):
    seen = set()
    reduced = []
    for J in subsets:
        m = mirror_subset(J, d)
        canon = J if J <= m else m
        if canon not in seen:
            seen.add(canon)
            reduced.append(J)
    return reduced

# ----------------------------------------------------------------------
# Build ring and coefficients b_j
# ----------------------------------------------------------------------
def build_ring_and_coeffs(d):
    a_names = ['a{}'.format(i) for i in range(d-1)]
    R = PolynomialRing(QQ, names=a_names + ['z'], order='lex')
    a_vars = R.gens()[:-1]
    z = R.gens()[-1]
    coeffs = list(a_vars) + [R(1), R(1)]          # a0..a_{d-2}, 1, 1
    b = [R(0)] * (2*d + 1)
    for i in range(d+1):
        for k in range(d+1):
            b[i+k] += coeffs[i] * coeffs[k]
    return R, b, a_vars, z

# ----------------------------------------------------------------------
# Ideal membership test
# ----------------------------------------------------------------------
def ideal_contains_1(I):
    gb = I.groebner_basis()
    return any(g.is_constant() and g != 0 for g in gb)

def check_subset(J, R, b, a_vars, z, d):
    eqs = [b[j] for j in J]
    # weak condition: only a0 ≠ 0
    weak = eqs + [z * a_vars[0] - 1]
    if ideal_contains_1(R.ideal(weak)):
        return None
    # full condition: all a_i ≠ 0
    prod = R(1)
    for v in a_vars:
        prod *= v
    full = eqs + [z * prod - 1]
    if ideal_contains_1(R.ideal(full)):
        return None
    return 'exists'

# ----------------------------------------------------------------------
# Main search
# ----------------------------------------------------------------------
def search_sparse_complete(d, max_subsets=None):
    print(f"\n=== Degree {d} ===")
    if d+1 < 5:
        print("Degree too low (Lemma 2).")
        return []

    cand = list(range(2, 2*d - 1))
    k = d + 1
    total_raw = comb(len(cand), k)
    all_raw = list(itertools.combinations(cand, k))
    print("raw data")
    print(len(all_raw))
    subsets = reduce_by_symmetry_c(all_raw, d)
    print(f"Subsets to check: {len(subsets)} (raw {total_raw})")
    if max_subsets:
        subsets = subsets[:max_subsets]
        print(f"Limited to first {max_subsets}")

    R, b, a_vars, z = build_ring_and_coeffs(d)
    found = []
    total = len(subsets)
    start = time.time()

    for idx, J in enumerate(subsets):
        if idx % 100 == 0 and idx > 0:
            elapsed = time.time() - start
            print(f"  ... {idx}/{total} subsets ({elapsed:.1f}s)")
        if check_subset(J, R, b, a_vars, z, d) is not None:
            print(f"\n>>> Found candidate for J = {J}")
            found.append(J)

    elapsed = time.time() - start
    print(f"\nFinished degree {d} in {elapsed:.1f}s")
    print(f"Candidates found: {len(found)}")
    if not found:
        print("Result: No sparse square (consistent with Abbott).")
    else:
        print("WARNING: Candidate found – would contradict Abbott's theorem!")
    return found

# ----------------------------------------------------------------------
# User interaction
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        d = int(input("Enter degree d: "))
    except:
        d = 5
    if d >= 12:
        proceed = input("Degree ≥12 is very heavy. Continue? (y/n): ").strip().lower()
        if proceed != 'y':
            exit()
    search_sparse_complete(d)