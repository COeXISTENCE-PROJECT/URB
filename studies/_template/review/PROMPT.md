# Independent review prompt

Use this prompt in a fresh LLM session that has no history of the study. The review must be supplied by an agent that took no role in developing or running the study. You are encouraged to include more than one review. Replace the bracketed text and give the agent access to the repository.

---

You are assigned to independently review the completed URB study at `[study path]`.

You must be independent. You must not have helped design the study, change its code, run its experiments, choose its results, or interpret it. If you were involved in any of these, say so and do not present the review as independent.

Read the full study and inspect the relevant repository code, configurations, saved experiment settings, metrics, and RouteRL implementation. Inspect existing studies and reviews under `studies/`, if any, as references. Do not rely only on the study's own description. Do not change the proposal, experiments, results, or conclusion.

This review decides whether a focused study is clear and trustworthy enough to appear on the URB website. It is not publication peer review. Do not require broad novelty, exhaustive baselines, publication-scale statistics, or claims beyond the stated scope. Narrow scope and lightweight experiments are acceptable when they can answer the stated question and their limits are clear. Reject work whose central evidence or conclusion cannot be trusted.

Your only output should be your review report. Use the template `review/REVIEW.md`. Do not change the structure of this file, as it will be automatically parsed. Name your review file after your model, version, the date and time of the review (e.g., `gpt6-sol-extra-high_20260909_165315.md`). Your review must:

1. Give the model, provider, and version shown by your interface. Write `unavailable` for anything you cannot verify.
2. Confirm whether you had any role in the study.
3. List the important repository files you inspected and briefly explain how the tested experiment works.
4. Restate the hypothesis, tested setting, main metric, meaningful-effect threshold, and decision rules in your own words. Confirm whether you understand them. Assess whether the hypothesis is interesting and useful for future work in URB.
5. Check the controls, uncertainty, strength of evidence, and conclusion against the proposal and final report. Assess whether the experiments and evidence sufficiently cover the scope of the hypothesis.
6. State the strongest evidence or explanation against the conclusion.
7. Score the study using the table and recommendation guidelines in `review/REVIEW.md`. Score its fitness as a focused URB study, not whether its result is positive or whether it is publication-ready.
8. Give one verdict: `accept`, `revise`, or `reject`.
9. Be concise and decisive. The review is meant to be read by humans, so do not be overly verbose.

**Be concise and specific**. Refer to exact repository paths and result values when they matter. Mark anything you cannot check as unverified rather than guessing.

In case you cannot access your full model/reasoning budget, you are: `[LLM model]`.