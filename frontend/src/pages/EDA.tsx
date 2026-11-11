type Figure = {
  file: string;
  title: string;
  caption: string;
};

// Real exported figures (reports/figures/, copied into frontend/public/eda/)
// - the exact images embedded in notebooks/eda_md17.ipynb's output cells,
// not redrawn here, so this page can never show something the notebook
// and docs/eda.md disagree on. Captions are Shijin's own, from
// reports/figures/README.md, not paraphrased.
const FIGURES: Figure[] = [
  {
    file: "md17_02_energy_distributions.png",
    title: "Energy distributions — all 9 gold datasets",
    caption:
      "All nine MD17 datasets have clean unimodal energy distributions — no corruption, no outlier clusters.",
  },
  {
    file: "md17_03_force_magnitude_distributions.png",
    title: "Force-magnitude distributions — all 9 gold datasets",
    caption: "Force magnitudes are similarly well behaved across every dataset.",
  },
  {
    file: "md17_04_interatomic_distances_vs_the_5_cutoff.png",
    title: "Interatomic distances vs. the 5 Å cutoff",
    caption:
      "The 5 Å cutoff captures 100% of ethanol's atom pairs but only 59% of azobenzene's — the one " +
      "worth a slide of its own. For 9-atom molecules the neighbour graph is the complete graph, so the " +
      "cutoff provides no locality structure at that size. Not a problem for MD17, but the same default " +
      "cannot be assumed to behave sensibly on MD22's systems of up to 370 atoms.",
  },
  {
    file: "md17_05_same_molecule_two_theory_levels_ethanol.png",
    title: "Ethanol at two theory levels — DFT vs. CCSD(T)",
    caption: "Neither theory level looks anomalous relative to the other.",
  },
  {
    file: "md17_06_train_val_test_split_integrity_check.png",
    title: "Train/val/test split integrity check",
    caption: "Train/val/test are contiguous blocks with no interleaving — no leakage.",
  },
];

export default function EDA() {
  return (
    <>
      <div className="page-header">
        <h1>Exploratory data analysis</h1>
        <p>
          Real figures from <code>notebooks/eda_md17.ipynb</code>, exported as-is — not redrawn for
          this page. Full write-up: <code>docs/eda.md</code>. Every model on this site trains and is
          scored on exactly the gold-stage data these charts describe.
        </p>
      </div>

      {FIGURES.map((fig) => (
        <div className="card" key={fig.file}>
          <p className="card-title">{fig.title}</p>
          <img
            src={`/eda/${fig.file}`}
            alt={fig.title}
            style={{ width: "100%", borderRadius: 8, display: "block", marginTop: 8 }}
          />
          <p style={{ margin: "10px 0 0", fontSize: 14, lineHeight: 1.6, color: "var(--ink-secondary)" }}>
            {fig.caption}
          </p>
        </div>
      ))}
    </>
  );
}
