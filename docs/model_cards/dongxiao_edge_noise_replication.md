# Edge-noise replication checkpoints

At validation-selected checkpoints, edge noise improved raw energy MAE in 2/3 fine-tuning seeds and force MAE in 0/3 seeds. These are paired runs from one pretrained checkpoint, not three independent end-to-end training initializations. No test evaluation was performed.

Protocol and validation-only limitations: `notebooks/dongxiao_dimenet/edge_noise/seed_replication/Replication_Report.md`.

### Seed 42: deterministic

Selected additional epoch 18. Checkpoint `experiments/results/dongxiao_edge_noise_20/deterministic_best.pt` (Git ignored). SHA256 `ae5027a03ff60ea46469b1857c74bab0906d77168accf96f6ed8f533995fc323`.

### Seed 42: edge_gaussian

Selected additional epoch 19. Checkpoint `experiments/results/dongxiao_edge_noise_20/edge_gaussian_best.pt` (Git ignored). SHA256 `8565d7349f9768cde9ee9abe199eddbaf52bb3d7da00e9a9a1cbf7346d89f0fb`.

### Seed 43: deterministic

Selected additional epoch 18. Checkpoint `experiments/results/dongxiao_edge_noise_20_seed43/deterministic_best.pt` (Git ignored). SHA256 `1b2840e27891a7647b52bd08a61664d2329d1e09dd442019c34f134dcc3f1247`.

### Seed 43: edge_gaussian

Selected additional epoch 19. Checkpoint `experiments/results/dongxiao_edge_noise_20_seed43/edge_gaussian_best.pt` (Git ignored). SHA256 `ec6fcf1c87c416539ac50ee52821bde3969165669dc88ca09c8aa06ffd17c64f`.

### Seed 44: deterministic

Selected additional epoch 19. Checkpoint `experiments/results/dongxiao_edge_noise_20_seed44/deterministic_best.pt` (Git ignored). SHA256 `e6c736f6fcf86c0e44220cd491b4abb857d19f81f409b1260181fa84c535a8a2`.

### Seed 44: edge_gaussian

Selected additional epoch 17. Checkpoint `experiments/results/dongxiao_edge_noise_20_seed44/edge_gaussian_best.pt` (Git ignored). SHA256 `09b66e7013e356e3b2a8f977d55aaf59971d00cc658b8e155734f30397c3d2bf`.
