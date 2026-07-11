# The Disagreement Problem in Explainable Machine Learning

- Source: https://arxiv.org/abs/2202.01602
- Local source: `paper.pdf`
- Extracted text: `paper_extracted.md`

## What the paper actually says

Different post-hoc explanation methods can explain the same prediction with different important features, rankings, signs, and magnitudes. The paper formalizes this as the disagreement problem and studies it through practitioner interviews, experiments across four datasets, six explanation methods and six predictive models, plus a user study.

The central result is that disagreement is common, not an edge case. LIME, SHAP-family methods, gradient-based methods and other explainers can disagree because they make different assumptions, use different perturbations or baselines, and approximate different notions of importance. Practitioners often resolve conflicts with ad hoc rules, such as trusting familiar methods or choosing the explanation that appears most intuitive. That can create false confidence in high-stakes decisions.

The paper argues that explanation comparison must state what is being compared: top-k feature membership, rank order, sign, or importance magnitude. There is no universally correct explainer independent of the goal. Evaluation should be tied to the desired property and should acknowledge uncertainty or disagreement.

## CEOAI syllabus mapping

- Closest mapping: `2(a-c)` for explaining classical models and ensembles.
- Closest mapping: `3(a-c)` for gradient-based explanations of neural networks.
- Supports `4` and `5` when explanations concern text tokens or image regions.

Explainability methods and disagreement are not explicit CEOAI syllabus items. Treat this as a likely competition extension around supplied models, not as theory you must memorize exhaustively.

## What to retain for competition

Never report "the explanation" without naming the method and parameters. If two methods disagree, compare them with an explicit criterion and check prediction sensitivity by masking or perturbing the claimed features. For a timed task, inspect baseline/reference choice, random seed, top-k definition, and whether positive and negative attributions are handled separately.
