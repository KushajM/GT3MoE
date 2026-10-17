# Solver replay: balanced greedy vs the exact LAP

All three solvers are scored against the **ceiling-only** optimum, which is the feasible set `solve_greedy` works in. `exact_floor` additionally enforces the capacity floor, so its ratio above 1.0 is the price of the floor, not solver suboptimality.

Averaged over seeds and epochs. The trace was generated under one assignment path, so this measures solver suboptimality on the costs actually observed, not the training effect of switching solver.

**`solve_br_weighted` is deliberately excluded.** It minimises the sample-mass weighted potential plus a quadratic congestion term, not the objective above, so a single phi ratio does not compare it to anything. It is also degenerate at lambda = 0 when C/K is an integer: cap_lo = cap_hi leaves no feasible single-class move, so it returns its incumbent unchanged (measured churn 0.0 at K = 4, C = 100). It is evaluated in the E7 arms at its deployed congestion_lambda instead.

| dir | K | solver | epochs | phi/phi_opt mean | phi/phi_opt max | churn mean (classes/epoch) | churn max |
|---|---|---|---|---|---|---|---|
| F_c100lt | 8 | exact_ceiling | 180 | 1.0 | 1.0 | 2.1 | 89.0 |
| F_c100lt | 8 | exact_floor | 180 | 1.0011 | 1.0129 | 2.1 | 90.0 |
| F_c100lt | 8 | greedy | 180 | 1.0437 | 1.9326 | 2.7 | 88.0 |
| ksweep | 4 | exact_ceiling | 180 | 1.0 | 1.0 | 1.7 | 79.0 |
| ksweep | 4 | exact_floor | 180 | 1.0 | 1.0 | 1.7 | 79.0 |
| ksweep | 4 | greedy | 180 | 1.0411 | 1.9741 | 2.0 | 78.0 |
| ksweep | 16 | exact_ceiling | 180 | 1.0 | 1.0 | 2.4 | 97.0 |
| ksweep | 16 | exact_floor | 180 | 1.0043 | 1.0418 | 2.5 | 97.0 |
| ksweep | 16 | greedy | 180 | 1.0092 | 1.1407 | 3.0 | 97.0 |
| ksweep | 32 | exact_ceiling | 180 | 1.0 | 1.0 | 3.2 | 98.0 |
| ksweep | 32 | exact_floor | 180 | 1.0157 | 1.1202 | 3.4 | 97.0 |
| ksweep | 32 | greedy | 180 | 1.0048 | 1.1197 | 3.5 | 98.0 |
