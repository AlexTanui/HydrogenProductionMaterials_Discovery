import { IconLayers, IconNetwork, IconScales } from "../components/icons";

const FEATURES = [
  {
    icon: IconNetwork,
    color: "var(--series-blue)",
    title: "Message-passing potentials",
    body: "A SchNet-style MPNN predicts total energy from atomic structure; per-atom forces follow via automatic differentiation, keeping the potential energy-conserving.",
  },
  {
    icon: IconScales,
    color: "var(--series-aqua)",
    title: "Calibrated uncertainty",
    body: "Beyond a single prediction, the model reports how much to trust it — reproducing BLIP's Bayesian weight-space approach and comparing it against a graph-space alternative.",
  },
  {
    icon: IconLayers,
    color: "var(--series-orange)",
    title: "Where should uncertainty live?",
    body: "The core research question: is stochasticity better placed in network weights, or directly in the graph's node and edge representations?",
  },
];

const ROADMAP = [
  {
    title: "Phase 1 — Basic MPNN",
    body: "Deterministic message-passing baseline on MD17: predicts total potential energy and, via autograd, atomic forces.",
  },
  {
    title: "Phase 2 — BLIP reproduction",
    body: "Reproduces BLIP's input-dependent Gaussian stochasticity in the MPNN's weights, enabling uncertainty estimation via stochastic forward passes.",
  },
  {
    title: "Phase 3 — Graph-space stochasticity",
    body: "The team's own contribution: moving stochasticity from model weights into node/edge representations, evaluated against Phases 1 and 2.",
  },
];

export default function Dashboard() {
  return (
    <>
      <div className="hero">
        <div className="hero-eyebrow">
          <span className="badge-dot" /> Adelaide University · College of Engineering &amp; IT
        </div>
        <h1>Adaptive graph-space uncertainty modelling for interatomic potentials</h1>
        <p>
          Investigating where stochasticity should be introduced in molecular graph neural networks —
          model weights or graph representations — to make machine learning interatomic potentials
          know when their predictions can be trusted.
        </p>
      </div>

      <div className="feature-grid">
        {FEATURES.map((f) => (
          <div className="feature-card" key={f.title}>
            <div className="feature-icon" style={{ background: f.color }}>
              <f.icon />
            </div>
            <h3>{f.title}</h3>
            <p>{f.body}</p>
          </div>
        ))}
      </div>

      <div className="stat-grid">
        <div className="stat-tile">
          <p className="stat-label">MD17 molecules</p>
          <p className="stat-value">8</p>
        </div>
        <div className="stat-tile">
          <p className="stat-label">Predicted quantities</p>
          <p className="stat-value">2</p>
        </div>
        <div className="stat-tile">
          <p className="stat-label">Research phases</p>
          <p className="stat-value">3</p>
        </div>
        <div className="stat-tile">
          <p className="stat-label">Serving model</p>
          <p className="stat-value" style={{ fontSize: 16 }}>
            <span className="badge badge-warning">
              <span className="badge-dot" />
              Phase 1 (preview)
            </span>
          </p>
        </div>
      </div>

      <div className="card">
        <p className="card-title">Research plan</p>
        <div className="stepper">
          {ROADMAP.map((r, i) => (
            <div className={`step${i === 0 ? " current" : ""}`} key={r.title}>
              <div className="step-rail">
                <div className="step-dot">{i + 1}</div>
                {i < ROADMAP.length - 1 && <div className="step-line" />}
              </div>
              <div className="step-body">
                <h3>{r.title}</h3>
                <p>{r.body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <p className="card-title">Try it</p>
        <p style={{ margin: "0 0 8px", fontSize: 14, lineHeight: 1.6, color: "var(--ink-secondary)" }}>
          <b style={{ color: "var(--ink-primary)" }}>Predict</b> — pick an MD17 molecule and
          trajectory frame to see predicted energy, force magnitude, and uncertainty.
        </p>
        <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "var(--ink-secondary)" }}>
          <b style={{ color: "var(--ink-primary)" }}>Benchmarks</b> — compares all three phases as
          each is trained and evaluated.
        </p>
      </div>
    </>
  );
}
