---
title: "Solving the Unsolvable: A Heuristic Approach to MAX-SAT in Python"
excerpt: "Faced with an 'unsolvable' NP-hard problem, how do you find a great answer when the perfect one is out of reach? This post details the journey of building a Weighted MAX-SAT solver from scratch in Python. I explore why brute-force solutions fail spectacularly and break down the design of a greedy, iterative heuristic that quickly hones in on high-quality solutions under tight time constraints. Dive into the strategy, the results, and the lessons learned from tackling a classic combinatorial nightmare."
collection: portfolio
---


-----

Imagine you're planning a project. You have a list of desired features, but some of them conflict. Feature A is crucial and has a "value" of 100. Feature B is also important, worth 20. However, implementing A means you can't have Feature C, which is worth 10. How do you choose the set of features that gives you the maximum possible value?

You've just walked through a classic logic puzzle—one that computer scientists know as the **Weighted Maximum Satisfiability Problem**, or MAX-SAT. It’s a notoriously difficult challenge that appears everywhere, from software verification to AI planning and bioinformatics.

The goal is to find an assignment of TRUE or FALSE to a set of variables that satisfies the maximum possible weight of given logical "clauses" or rules. The problem is that as the number of variables grows, the number of possible solutions explodes exponentially. It’s a classic NP-hard problem, meaning a perfect, efficient solution for all cases is believed not to exist.

Faced with this challenge, I set out to build a solver from scratch in pure Python. The goal wasn't to find the *perfect* answer every time—that's computationally infeasible—but to find a *very good* one within a strict time limit. This is the story of moving from a brute-force nightmare to a practical, heuristic-based solution.

### The Naive Approach: Why We Can't Just Check Everything

The most straightforward way to solve a MAX-SAT problem is to try every single possible solution. If you have N variables (or "atoms" in logic terms), each can be either TRUE or FALSE. This gives you $2^N$ possible combinations, or Knowledge Bases (KBs), to check.

For a tiny problem with 4 variables, like the `hermione.txt` example from the project, we have $2^4 = 16$ possible solutions. We can write a simple script to check them all:

1.  Generate all unique atoms (`harryIsSmart`, `likesHermioneRon`, etc.).
2.  Create every possible subset of these atoms (these are our candidate KBs).
3.  For each KB, calculate the total weight of the rules it satisfies.
4.  Keep the one with the highest score.

This exhaustive search works perfectly for small examples. But what about a problem with 70 variables? The number of combinations becomes $2^{70}$, which is over a billion-trillion possibilities. No computer on Earth could check them all in our lifetime. This combinatorial explosion is why we need a smarter approach.

### Crafting a Heuristic: A Greedy Strategy for a Hard Problem

Since finding the perfect solution is off the table, we turn to heuristics—clever shortcuts that guide us to a good-enough solution quickly. My approach is an iterative, greedy algorithm that prioritizes high-value decisions to rapidly narrow in on a high-weight KB.

Here’s how it works.

#### Step 1: The Greedy Premise — Prioritize by Weight

The foundation of the strategy is simple and intuitive: **satisfy the most important rules first.** Each clause in a MAX-SAT problem has a weight representing its importance. A rule with a weight of 100 is far more critical than one with a weight of 1.

The first step is to parse all the clauses and sort them in descending order of their weight. This creates a prioritized to-do list for our solver.

To handle very large problems, I added another heuristic: only consider the top 1000 highest-weight clauses to build the solution. This is a trade-off, sacrificing theoretical perfection for a massive gain in speed.

#### Step 2: The Core Algorithm — Iterative Simplification

With our prioritized list of clauses, the solver begins an iterative process of building and evaluating candidate solutions. The core idea in each iteration is to make a tentative decision and then see how it simplifies the rest of the problem.

This process is inspired by a technique used in modern SAT solvers called **Unit Propagation**. Here's the logic in plain English:

1.  **Pick a Variable**: The algorithm looks at the highest-weight unsatisfied clause and picks a variable from it.
2.  **Assign a Value**: It tentatively assigns this variable a truth value (TRUE or FALSE) based on how it appears in that high-weight clause. The goal is to satisfy that clause.
3.  **Propagate the Decision**: This is where the magic happens. This single decision ripples through the entire set of clauses.
      * Any other clause that is now satisfied by this assignment is marked as complete and set aside.
      * If a clause contains the *negation* of our assigned variable, that negated literal is now FALSE and can be removed from the clause. This makes the clause shorter and easier to satisfy.
4.  **Find the Next Forced Move**: Sometimes, removing a literal leaves a clause with only one variable left (a "unit clause"). To satisfy the overall system, we are now *forced* to assign a value to that variable. This new forced decision triggers another round of propagation, which can cascade through the problem, rapidly simplifying it.

This chain reaction continues until no more forced moves can be made. At the end of this process, we have a complete assignment of TRUE/FALSE values for all variables—our candidate Knowledge Base.

#### Step 3: Explore, Evaluate, Repeat

The solver doesn't just do this once. It runs this iterative simplification process multiple times, starting with slightly different initial decisions based on the highest-weight clauses. In each round, it calculates the total weight of the resulting KB and compares it to the best solution found so far.

The algorithm stops when it either hits the time limit or achieves the maximum possible score (meaning all clauses are satisfied).

### Putting It to the Test: Results and Analysis

I ran the solver against 72 MAX-SAT problems of varying sizes and complexity, imposing different time limits to see how it performed under pressure. The results were fascinating.

The evaluation metric is a simple score: `(achieved_weight / total_possible_weight) * 100`.

Even with a time limit of just 5 seconds per problem, the results were impressive.

As the histogram shows, the solver consistently achieves high scores, with the vast majority falling between 80% and 100%. This confirms that the greedy, weight-based heuristic is effective at finding high-quality solutions. For several problems, it found the provably optimal solution (a score of 100%).

But perhaps the most interesting insight came from looking at *when* the best solution was found during each run.

This chart reveals that for most problems, the best KB was discovered very early in the process, often in under a second. The greedy approach appears to hone in on a great solution almost immediately. The remaining time is spent exploring other possibilities that rarely beat the one found in the initial greedy rush.

### Final Thoughts

Tackling an NP-hard problem like Weighted MAX-SAT is a humbling and enlightening experience. It forces you to abandon the pursuit of perfection and instead embrace the art of approximation.

This project was a journey from understanding the sheer impossibility of a brute-force search to designing a practical heuristic that delivers excellent results under real-world constraints. The key takeaways are clear:

1.  **Greedy Algorithms Are Powerful**: For many complex optimization problems, a simple greedy strategy—like prioritizing by weight—can get you 80-90% of the way to the optimal solution with a fraction of the computational cost.
2.  **Iterative Simplification Works**: Techniques like unit propagation, which create a cascade of logical deductions, are incredibly effective at cutting down the search space.
3.  **The Best Solution is Often Found Early**: With a strong heuristic, you often find your best-guess answer quickly. The law of diminishing returns applies heavily afterward.

While this solver might not beat a state-of-the-art industrial tool, it’s a testament to what can be accomplished from scratch with a solid understanding of the problem and a bit of algorithmic creativity.
