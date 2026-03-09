#!/usr/bin/env sage
"""
Sparse squares of COMPLETE BIVARIATE polynomials
Based on Abbott-style Gröbner approach
"""

from itertools import combinations
from sage.all import *



def monomials_deg_leq(d):
    mons = []
    for i in range(d+1):
        for j in range(d+1-i):
            mons.append((i,j))
    return mons




def build_ring(d):

    mons = monomials_deg_leq(d)

    fixed = {(d,0), (d-1,1)}

    vars = []
    for m in mons:
        if m not in fixed:
            vars.append(f"a_{m[0]}_{m[1]}")

    R = PolynomialRing(QQ, vars + ["z"], order='lex')
    return R, mons, fixed


def build_square(R, mons, fixed, d):

    gens = R.gens()
    a_vars = gens[:-1]
    z = gens[-1]

    coeff = {}
    k = 0
    for m in mons:
        if m in fixed:
            coeff[m] = R(1)
        else:
            coeff[m] = a_vars[k]
            k += 1

    b = {}
    for m1 in mons:
        for m2 in mons:
            exp = (m1[0]+m2[0], m1[1]+m2[1])
            b[exp] = b.get(exp, R(0)) + coeff[m1]*coeff[m2]

    return b, a_vars, z




def count_terms_dict(b):
    return sum(1 for v in b.values() if v != 0)




def check_sparse_subset(J, b, R, a_vars, z):

    eqs = [b[m] for m in J]

    weak = eqs + [z*a_vars[0] - 1]
    if 1 in R.ideal(weak).groebner_basis():
        return False

    prod = prod(a_vars)
    full = eqs + [z*prod - 1]

    if 1 in R.ideal(full).groebner_basis():
        return False

    sols = R.ideal(full).variety()
    if sols:
        print("Solution:", sols[0])
        return True

    return False


def search_bivariate_sparse(d):

    print(f"\n=== Bivariate degree {d} ===")

    R, mons, fixed = build_ring(d)
    b, a_vars, z = build_square(R, mons, fixed, d)

    total_terms = len(mons)
    print("Original terms:", total_terms)

    b_keys = list(b.keys())

    k = total_terms

    for r in range(k, len(b_keys)):
        for J in combinations(b_keys, r):
            if check_sparse_subset(J, b, R, a_vars, z):
                print("Sparse square exists!")
                return

    print("No sparse square found.")



if __name__ == "__main__":
    d = int(input("Enter bivariate degree d: "))
    search_bivariate_sparse(d)