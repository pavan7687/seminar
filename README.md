
# Sparse Squares of Complete Polynomials

This project implements the algorithm from **Abbott (2000)** to search for sparse squares of complete polynomials using **SageMath**.

The program generates all possible subsets of coefficients that could become zero when squaring a complete polynomial of degree **d**. It then checks these subsets using **Groebner basis computations** to determine whether such a sparse square can exist.

## Requirements

* SageMath

## How to Run

Run the program using Sage:

```bash
sage filename.sage
```

Then enter the **degree `d`** when prompted.

## What the Program Does

1. Builds a polynomial ring with symbolic coefficients.
2. Computes the square of a complete polynomial.
3. Generates candidate subsets of coefficients that may vanish.
4. Reduces symmetric cases to avoid duplicate checks.
5. Uses Groebner basis calculations to test if a valid solution exists.

## Output

* If no valid subset is found, the result is consistent with **Abbott’s theorem**.
* If a subset is found, it indicates a potential candidate that would contradict the theorem.
