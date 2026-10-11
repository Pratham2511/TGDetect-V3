# Tested architectures

`V3.mmd` describes the trained V3 configuration in `../../results/v3_vs_orthrus/V3_accuracy.json`; `ORTHRUS.mmd` describes `../orthrus_comparison/official.py` and the copied authors' encoder/decoder code.

V3 components: offline typed key/value tokens; two-hash embeddings and ordered subtokens; typed field encoder; frozen identity lookup at inference; endpoint history rings; causal 4+4 history planner; two GAT layers; dilated causal CNN; importance pooling; novelty/count-min rarity/write gates; compressed short-memory rings; shared second CNN pass and short-memory attention; active/short fusion; fast/slow affine endpoint memory scan; gated memory read; relation reconstruction; endpoint-pair plausibility; validation calibration and node evidence aggregation. The tested V3 has 785,327 learned parameters. Later V4 and hybrid configurations are excluded.

ORTHRUS comparison components: pretrained 128-dimensional attribute features plus three node-type features; separate endpoint projections; graph reindexing; 20-neighbor cache; historical relation features; two Graph Transformer layers; source/destination relation decoder; relation cross entropy; matched calibration/node evidence scoring. The comparison adapter has 764,202 learned parameters. Consult `../orthrus_official/config/orthrus.yml` for the authors' broader pipeline settings.

The original ORTHRUS paper includes its own detection/postprocessing protocol. Our recorded comparison uses the unchanged neural architecture with matched V3 checkpoint selection and node-evidence calibration; it is not a reproduction of every published postprocessing rule. The actual source files and result notes take precedence over diagrams.
