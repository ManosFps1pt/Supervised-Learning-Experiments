# CEOAI Priority Handoff

## Status

Current date: 2026-07-04. CEOAI starts 2026-07-14.

Calendar days left: 10. Effective work days left excluding 2026-07-13: 9.

Since the previous automation run, pace is FAST, but the overall sprint is still behind. Search/RL is already analyzed enough to stop being the next-block priority. CV now has one executed CNN baseline. The remaining hole is broad CEOAI ML and DL coverage: classical models, ensembles, clustering, dimensionality reduction, and more deep-learning rows still lack enough executed repo evidence.

Baseline for comparison: previous automation run timestamp 2026-07-04T05:06:54.114Z and the automation memory entry dated 2026-07-04T08:11:05.8250361+03:00.

## New Since Previous Run

- `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/exercises/search_rl_comparison.md`
  - Counts for CEOAI `1(c)` Monte Carlo method and strengthens `1(e)` Temporal Difference Learning.
  - Evidence: no blank recognition rows, completed Monte Carlo vs TD vs Q-learning table, and one-sentence contracts for all six Search/RL methods.
- `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/exercises/solution2.ipynb`
  - Counts as the saved minimax artifact replacing the split block files for CEOAI `1(b)`.
  - Evidence: executed move-score telemetry and a saved completion markdown cell.
- `olympiads/IOAI Material/8. Computer Vision/exercises/cv_mnist_personal_benchmark.ipynb`
  - Counts for CEOAI `5(b)` CNN architectures and IOAI image-classification practice.
  - Evidence: executed MNIST CNN training with saved accuracy up to about `0.988`, saved adversarial-image cells, and saved prediction/confidence output.
- `olympiads/reviews/error_journal.jsonl`
  - New resolved MNIST preview-label and CrossEntropy target-dtype entries support the CV notebook.
  - One open evaluation-loop entry still looks stale against the saved notebook and should be cleaned only after rechecking the notebook cell contract.

Already counted before: A* notebook, earlier gridworld Q-learning notebook, NLP TF-IDF/BERT/language-model/submission notebooks, and the basic PyTorch regression/two-moons notebooks.

Current unfinished CEOAI-only concept scope:

- Section 2 ML: classical algorithms, clustering, ensemble methods, dimensionality reduction.
- Section 3 DL: perceptron, MLP, backpropagation, SGD/Adam/RMSProp, learning-rate schedules, regularization, dropout, batch normalization, CNNs, RNNs, LSTM, GRU, transformers, autoencoders, VAE, GANs, diffusion models.
- Section 4 NLP: preprocessing, embeddings, related architectures such as Seq2Seq, T5, LLaMA where repo evidence is still weaker than Search/RL.
- Section 5 CV: processing, CNN architectures, related architectures such as YOLO, Stable Diffusion, ViT beyond the single MNIST CNN baseline.

Exclude completely from this handoff because they are not in CEOAI syllabus:

- Audio.
- HuBERT.
- Whisper.
- Voxtral.
- Qwen-Audio.
- Any IOAI-only audio workflow.

## Study Next

1. Target file: `olympiads/IOAI Material/2. (Mostly) Linear models/exercises/ml_digits_benchmark.ipynb`
   - Syllabus tag: CEOAI `2(a)` Classical algorithms and `2(c)` Ensemble Methods.
   - Required visible evidence: executed `load_digits()` baseline with at least `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`, and `RandomForestClassifier`, plus one metrics table and one confusion matrix.
   - Why highest value: one notebook closes the largest remaining CEOAI Section 2 gap fastest.

2. Target file: `olympiads/IOAI Material/2. (Mostly) Linear models/exercises/ml_clustering_pca_drill.ipynb`
   - Syllabus tag: CEOAI `2(b)` Clustering and `2(d)` Dimensionality Reduction.
   - Required visible evidence: executed PCA-to-2D projection, K-Means labels, DBSCAN labels, and one short comparison note saying when each method is useful.
   - Why highest value: after the supervised benchmark, this is the quickest way to remove the remaining CEOAI ML blind spot before more polishing.

## Pass/Fail Check Before Next Run

Pass only if:

- `ml_digits_benchmark.ipynb` exists with executed cells, a model-comparison table, and at least one confusion matrix or classification report saved in outputs.
- `ml_clustering_pca_drill.ipynb` exists with executed PCA scatter output and both K-Means and DBSCAN results saved.

Fail if:

- Work is only reading, source collection, markdown planning, or empty notebook scaffolding.
- Search/RL is revisited instead of closing one of the ML artifacts above.

## Avoid Until This Is Done

- More Search/RL cleanup.
- More broad reading.
- More slide/PDF collection.
- Any audio work.
- Refactoring notebooks that already count.
- From-scratch implementations when scikit-learn already covers the syllabus row.

## Evidence To Recheck

- `olympiads/ceoai_syllabus.md`
- `olympiads/ioai_syllabus.md`
- `olympiads/schedule.csv`
- `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/exercises/search_rl_comparison.md`
- `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/exercises/solution2.ipynb`
- `olympiads/IOAI Material/8. Computer Vision/exercises/cv_mnist_personal_benchmark.ipynb`
- `olympiads/IOAI Material/2. (Mostly) Linear models/exercises/`
- `olympiads/reviews/error_journal.jsonl`
