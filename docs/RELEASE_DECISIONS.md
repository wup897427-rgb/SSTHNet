# Release Decisions

This repository is a selected core-code release, not an approximately
reproducible copy of the complete private research workspace. Inclusion is
based on coherent functionality rather than randomly removing a percentage
of lines or selecting favourable results.

Included: final architecture, cached-input training logic, supporting model
analyses, ranking/bootstrap/calibration utilities, LR-660 code and synthetic
tests. Excluded: dataset construction, preprocessing, labels, split generation,
all patient arrays, all result tables, weights, logs, patch archives, paper
drafts, server launch records, tree-model experiments and third-party model
source packages. Other baseline adapters can be released separately after
licence and dependency review.

The authors must select the software licence. Until a licence is added,
this repository does not advertise permissive reuse rights. Upstream model
code is not automatically covered by a future licence selected for SSTHNet.

Publication statements should describe this as availability of selected core
model and analysis code. They should not state that the entire dataset-to-paper
pipeline, all baselines or trained weights are publicly available.
