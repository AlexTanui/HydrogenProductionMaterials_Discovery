import { IconShield } from "../components/icons";

type Case = {
  title: string;
  risk: string;
  evidence: string;
  mitigation: string;
};

const CASES: Case[] = [
  {
    title: "Overconfident uncertainty misdirecting real resources",
    risk:
      "If a materials-screening pipeline trusted BLIP's uncertainty as-is, a '95% confidence interval' " +
      "that is actually right only ~12% of the time would make a weak prediction look trustworthy. " +
      "Downstream, that can mean committing real lab time, compute, or (at larger scale) hazardous " +
      "synthesis work to a candidate the model was never entitled to be confident about.",
    evidence:
      "Measured, not hypothesised: phase2_blip's 95% energy interval covers the true DFT value only " +
      "~12% of the time (ECE 0.70, see Benchmarks). The NLL fix (SCRUM-61) made training numerically " +
      "stable but did not fix this — calibration got slightly worse, not better.",
    mitigation:
      "Don't ship a confidence number without a calibration check behind it. This project's own release " +
      "discipline: every reported model card states its real coverage/ECE next to its accuracy, and the " +
      "team treated 'uncertainty got worse' as a result to report, not a result to hide or re-run until " +
      "it looked better.",
  },
  {
    title: "Uninformatively wide uncertainty hiding the opposite failure",
    risk:
      "The opposite mistake is just as real: an uncertainty estimate so wide it is never wrong, but also " +
      "never useful for prioritising which candidates deserve scarce lab time. That has a cost too — " +
      "wasted compute and researcher time, which matters for a clean-energy project competing for both.",
    evidence:
      "Phase 3's first-draft mechanism (graph-space noise) avoids BLIP's overconfidence but overshoots: " +
      "its 90%-nominal energy interval actually covers 99.5% of true values, with predicted uncertainty " +
      "~8-17x larger than BLIP's. More honest, but not yet actionable.",
    mitigation:
      "Track both failure directions explicitly (ECE plus a coverage curve, not ECE alone) before calling " +
      "any phase 'calibrated' - a single summary number can hide which direction a model is wrong in.",
  },
  {
    title: "Scope creep from benchmark to deployment",
    risk:
      "This is a validated-methodology exercise on MD17 (a standard, well-understood benchmark), not a " +
      "deployed catalyst-discovery tool. Presenting early-benchmark numbers as if they already applied " +
      "to real hydrogen-production materials screening would overstate technology readiness to a client " +
      "or the public - a known failure mode in ML-for-science communication.",
    evidence:
      "CLAUDE.md states this explicitly as project scope: prove the methodology on MD17 'before it would " +
      "ever be pointed at real catalyst screening.' No part of this prototype claims a discovered material.",
    mitigation:
      "Every page in this app that reports a number also reports what it is a number *of* - a held-out " +
      "MD17 test frame, not a novel material - and this page exists so that distinction is explicit, not " +
      "assumed.",
  },
  {
    title: "Security: an unauthenticated API and unvalidated data",
    risk:
      "An open prediction API with no access control, pointed at real experimental data, is a path to " +
      "either resource abuse or quietly-wrong results feeding a real decision if deployed carelessly.",
    evidence:
      "glossary.md documents this directly: 'Auth: None... do not deploy publicly without adding auth " +
      "first.' Separately, the data pipeline's own `validate_sample` step rejects malformed or " +
      "out-of-convention input (NaN, wrong shape, undeclared units) before it ever reaches a model - " +
      "the one data-integrity safeguard already built, not aspirational.",
    mitigation:
      "Auth is a named precondition for any deployment beyond this internal prototype, not an " +
      "afterthought - and the data-validation gate that already exists is the template for what any new " +
      "data source (e.g. MD22) must pass before training on it.",
  },
];

export default function Ethics() {
  return (
    <>
      <div className="page-header">
        <h1>Ethics, security &amp; responsible use</h1>
        <p>
          Grounded in what this prototype actually found this sprint, not a generic checklist — see{" "}
          <b>Benchmarks</b> and <b>Predict</b> for the live numbers each case below refers to.
        </p>
      </div>

      <div className="feature-grid">
        <div className="feature-card">
          <div className="feature-icon" style={{ background: "var(--series-orange)" }}>
            <IconShield />
          </div>
          <h3>Why this page exists</h3>
          <p>
            A machine-learning interatomic potential's confidence estimate is only useful if it is honest.
            This sprint's central technical finding — that neither phase's uncertainty is well-calibrated,
            in opposite directions — is also the clearest real-world case this project has for what goes
            wrong when that confidence is trusted uncritically. The cases below are that finding, read
            through an ethics and security lens, with a concrete mitigation for each.
          </p>
        </div>
      </div>

      {CASES.map((c) => (
        <div className="card" key={c.title}>
          <p className="card-title">{c.title}</p>
          <div style={{ display: "grid", gap: 10, marginTop: 6 }}>
            <div>
              <p className="stat-label" style={{ margin: "0 0 2px" }}>
                The risk
              </p>
              <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "var(--ink-secondary)" }}>{c.risk}</p>
            </div>
            <div>
              <p className="stat-label" style={{ margin: "0 0 2px" }}>
                The evidence, from this project
              </p>
              <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "var(--ink-secondary)" }}>
                {c.evidence}
              </p>
            </div>
            <div>
              <p className="stat-label" style={{ margin: "0 0 2px" }}>
                What we did / what it requires
              </p>
              <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "var(--ink-secondary)" }}>
                {c.mitigation}
              </p>
            </div>
          </div>
        </div>
      ))}

      <div className="card">
        <p className="card-title">The societal case, stated plainly</p>
        <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "var(--ink-secondary)" }}>
          Hydrogen production materials discovery is motivated by the clean-energy transition — a
          genuine societal benefit. That motivation is exactly why overclaiming readiness, or shipping an
          uncertainty estimate that is not actually trustworthy, would be a disservice to it: resources
          misdirected on a false-confidence signal are resources not spent on real progress. Honest
          reporting of a negative calibration result, rather than a flattering one, is this project's
          concrete answer to that pressure.
        </p>
      </div>
    </>
  );
}
