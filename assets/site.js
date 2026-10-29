// Shared pieces for every page: nav, footer, the phase staircase, prev / next links, number formats.
(function () {
  const D = window.CASE;
  const REPO = "https://github.com/chaialishekar/AI-ROI-Navigator";
  // Public URL of the Streamlit app. Until it is set, "live analysis" links go to the repository.
  const APP_URL = "https://ai-roi-navigator-lwqttyuudthdtcwenmycja.streamlit.app";
  const ROOT = document.body.dataset.root || ".";

  const money = (x, digits) => {
    if (x === null || x === undefined) return "n/a";
    const sign = x < 0 ? "−" : "", a = Math.abs(x);
    if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(digits ?? 1)}M`;
    if (a >= 1e3) return `${sign}$${Math.round(a / 1e3)}k`;
    return `${sign}$${Math.round(a)}`;
  };
  const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
  const months = x => (x === null ? "never" : `${x.toFixed(1)} months`);
  const doc = path => `${REPO}/blob/main/${path}`;
  const app = page => (APP_URL ? `${APP_URL.replace(/\/$/, "")}/${page}` : REPO);

  const delivered = D.tracking.delivered / D.tracking.promised;
  const PHASES = [
    { slug: "discover", name: "Discover", q: "What's going wrong, and what does it cost?",
      made: "Problem statement, stakeholders, as-is process", gate: [money(D.plant.total), "a year at stake"] },
    { slug: "evaluate", name: "Evaluate options", q: "Which AI should they buy first?",
      made: "Priced options with sources, cautious view", gate: ["Predictive", "maint. first"] },
    { slug: "business-case", name: "Business case", q: "Will it pay for itself, and how sure are we?",
      made: "ROI model with ranges, break-even", gate: [`${money(D.published.net)} a year`, "planned"] },
    { slug: "readiness", name: "Readiness", q: "Can the team actually use it?",
      made: "Readiness check, change impact", gate: [`${D.readiness.weakest.length} gaps`, "to close"] },
    { slug: "decide", name: "Decide", q: "What exactly is finance signing?",
      made: "Two-page decision brief, options considered",
      gate: D.decision.label === "Proceed after fixes" ? ["Pilot after", "fixes"] : [D.decision.label, ""] },
    { slug: "deliver", name: "Deliver", q: "What gets built, and how do we know it works?",
      made: "Requirements, integration spec, 13 UATs, traceability", gate: ["13 tests", "pass"] },
    { slug: "measure", name: "Measure", q: "Did the savings show up?",
      made: "Monthly tracking, gap analysis by cause", gate: [`${pct(delivered)} of`, "year-1 target"] },
  ];
  const href = p => `${ROOT}/phases/${p.slug}.html`;

  function nav() {
    const el = document.querySelector("[data-nav]");
    if (!el) return;
    el.innerHTML = `<div class="wrap">
      <a class="logo" href="${ROOT}/index.html"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="15" width="5" height="7" rx="1" fill="#111"/><rect x="9.5" y="10" width="5" height="12" rx="1" fill="#111"/><rect x="17" y="4" width="5" height="18" rx="1" fill="#0f6d74"/></svg>AI ROI Navigator</a>
      <div class="nav-links">
        <a class="hide-sm" href="${ROOT}/index.html#phases">Phases</a>
        <a class="hide-sm" href="${ROOT}/index.html#decisions">Decisions</a>
        <a class="hide-sm" href="${REPO}" target="_blank" rel="noopener">GitHub</a>
        <a class="btn" href="${app("")}" target="_blank" rel="noopener">Live analysis ↗</a>
      </div></div>`;
  }

  function footer() {
    const el = document.querySelector("[data-footer]");
    if (!el) return;
    el.innerHTML = `<div class="wrap">
      <div><h2 style="margin:0">Built by Chaitali Shekar</h2>
        <p class="muted" style="margin:6px 0 0">AI business systems analysis, end to end.</p>
        <p class="small" style="margin:10px 0 0">Client costs and provider prices are estimates. The model is tested on a public, synthetic dataset. The year after launch is simulated.</p></div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <a class="btn ghost" href="${REPO}" target="_blank" rel="noopener">View the code</a>
        <a class="btn" href="${app("")}" target="_blank" rel="noopener">Open the live analysis ↗</a></div></div>`;
  }

  // Which phases this visitor has opened, kept in their own browser only. Storage can be blocked, so fail quietly.
  function visited() {
    try { return JSON.parse(localStorage.getItem("roi-visited")) || []; } catch { return []; }
  }
  function markVisited(i) {
    try { localStorage.setItem("roi-visited", JSON.stringify([...new Set([...visited(), i])])); } catch {}
  }

  // The 3D staircase. opts: current (index to highlight), link (steps open their page), panel (element for details)
  function stairs(el, opts = {}) {
    const n = 7, w = 78, dx = 22, dy = 13, base = 250, rise = 24, ly = base + 60;
    let sel = opts.current ?? 0, s = "";
    const seen = visited();
    PHASES.forEach((p, i) => {
      const x = 20 + i * w, h = 40 + i * rise, y = base - h;
      const words = p.name.split(" ");
      s += `<a href="${opts.link ? href(p) : "#"}" class="step" data-i="${i}" aria-label="Phase ${i + 1}: ${p.name}">
        <polygon class="top" points="${x},${y} ${x + w},${y} ${x + w + dx},${y - dy} ${x + dx},${y - dy}"/>
        <polygon points="${x + w},${y} ${x + w + dx},${y - dy} ${x + w + dx},${base - dy} ${x + w},${base}" fill="var(--step-side)"/>
        <rect class="front" x="${x}" y="${y}" width="${w}" height="${h}"/>
        <text x="${x + 8}" y="${y + 18}" font-size="11" fill="#9fb0a6">0${i + 1}</text>
        <text x="${x + 8}" y="${y + 34}" font-size="12" font-weight="600" fill="#f7f7f5">${words[0]}</text>
        ${words[1] ? `<text x="${x + 8}" y="${y + 48}" font-size="12" font-weight="600" fill="#f7f7f5">${words[1]}</text>` : ""}
        <text x="${x + 4}" y="${base + 18}" font-size="11" fill="#3b3b38">${p.gate[0]}</text>
        <text x="${x + 4}" y="${base + 31}" font-size="11" fill="#3b3b38">${p.gate[1]}</text></a>`;
    });
    const gx = 20 + 6 * w - 4;
    s += `<g><rect x="${gx - 26}" y="${ly - 13}" width="52" height="18" rx="5" fill="#111"/>
      <text x="${gx}" y="${ly}" font-size="11" text-anchor="middle" fill="#f7f7f5">Go-live</text></g>
      <path d="M ${20 + 6 * w + w / 2} ${ly + 8} C ${20 + 6 * w + w / 2} ${ly + 36}, 40 ${ly + 36}, 40 ${ly - 12}" fill="none" stroke="#8a8a85" stroke-width="1.2" stroke-dasharray="4 4" marker-end="url(#ar)"/>
      <text x="${(20 + 7 * w) / 2}" y="${ly + 52}" font-size="11" text-anchor="middle" fill="#6b6b6b">Results feed the next Discover</text>
      <path class="climb-arrow" fill="none" stroke="var(--teal)" stroke-width="2" stroke-linecap="round" marker-end="url(#arTeal)"/>
      <g class="climber" aria-hidden="true">
        <path d="M-4 0 L-1 -11 M6 -6 L2 -11" stroke="#111" stroke-width="2.6" stroke-linecap="round" fill="none"/>
        <path d="M0 -11 L2 -22" stroke="var(--teal)" stroke-width="6" stroke-linecap="round"/>
        <path d="M1.5 -20 L9 -27 M1.5 -20 L-5 -15" stroke="#111" stroke-width="2.4" stroke-linecap="round" fill="none"/>
        <circle cx="3.5" cy="-28.5" r="4.2" fill="#111"/>
      </g>`;
    const vertical = PHASES.map((p, i) =>
      `<a href="${href(p)}" class="${i === sel ? "on" : seen.includes(i) ? "done" : ""}"><span>0${i + 1}</span>${p.name}<em>${p.gate.join(" ")}</em></a>`).join("");
    el.innerHTML = `<svg viewBox="0 0 ${20 + 7 * w + dx + 10} ${ly + 60}" role="img" aria-label="Seven phases drawn as a rising staircase, each with its gate, and a person climbing it">
      <defs><marker id="arTeal" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#0f6d74" stroke-width="1.8" stroke-linecap="round"/></marker><marker id="ar" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#8a8a85" stroke-width="1.5"/></marker></defs>${s}</svg>
      <div class="stairs-v">${vertical}</div>`;

    function paint() {
      el.querySelectorAll(".step").forEach(g => {
        // Phases this visitor has opened turn green; the one the climber stands on stays teal
        const i = +g.dataset.i, on = i === sel, done = !on && seen.includes(i);
        g.querySelector(".top").style.fill = on ? "var(--teal)" : done ? "var(--done-top)" : "var(--step-top)";
        g.querySelector(".front").style.fill = on ? "var(--teal-dark)" : done ? "var(--done)" : "var(--step)";
      });
      // The climber stands on the selected step; the arrow shows the next step up
      const foot = i => [20 + i * w + w / 2 + dx / 2 - 6, base - (40 + i * rise) - dy / 2 + 1];
      const [cx, cy] = foot(sel);
      el.querySelector(".climber").style.transform = `translate(${cx}px, ${cy}px)`;
      const arrow = el.querySelector(".climb-arrow");
      if (sel < n - 1) {
        const [nx, ny] = foot(sel + 1);
        arrow.setAttribute("d", `M ${cx + 12} ${cy - 30} Q ${(cx + nx) / 2 + 4} ${Math.min(cy, ny) - 52}, ${nx - 4} ${ny - 10}`);
        arrow.style.display = "";
        arrow.classList.remove("draw"); void arrow.getBBox(); arrow.classList.add("draw");
      } else arrow.style.display = "none";
      if (opts.panel) {
        const p = PHASES[sel];
        opts.panel.innerHTML = `<div><span>Phase ${sel + 1} · ${p.name}</span>${p.q}</div>
          <div><span>Deliverable</span>${p.made}</div><div><span>Gate passed</span>${p.gate.join(" ")}</div>`;
      }
    }
    el.querySelectorAll(".step").forEach(g => {
      g.addEventListener("mouseenter", () => { sel = +g.dataset.i; paint(); });
      if (!opts.link) g.addEventListener("click", e => e.preventDefault());
    });
    if (!opts.link || opts.current === undefined) el.addEventListener("mouseleave", () => { sel = opts.current ?? sel; paint(); });
    paint();
  }

  function pager(i) {
    const el = document.querySelector("[data-pager]");
    if (!el) return;
    const prev = PHASES[i - 1], next = PHASES[i + 1];
    el.innerHTML = `<div class="wrap pager">
      ${prev ? `<a href="${href(prev)}"><div class="card"><div class="label">← Phase ${i}</div><h3 style="margin:4px 0 0">${prev.name}</h3></div></a>`
             : `<a href="${ROOT}/index.html"><div class="card"><div class="label">← Overview</div><h3 style="margin:4px 0 0">Home</h3></div></a>`}
      ${next ? `<a href="${href(next)}" class="next"><div class="card"><div class="label">Phase ${i + 2} →</div><h3 style="margin:4px 0 0">${next.name}</h3></div></a>`
             : `<a href="${ROOT}/index.html#decisions" class="next"><div class="card"><div class="label">Back to →</div><h3 style="margin:4px 0 0">Key decisions</h3></div></a>`}</div>`;
  }

  // A phase page's header: mini staircase plus the phase's question
  function phaseHead(i) {
    const p = PHASES[i];
    document.title = `${p.name} · AI ROI Navigator`;
    markVisited(i);
    const head = document.querySelector("[data-phase-head]");
    if (head) head.innerHTML = `<p class="eyebrow">Phase ${i + 1} of 7 · ${p.name}</p><h1>${p.q}</h1>`;
    const st = document.querySelector("[data-stairs]");
    if (st) stairs(st, { current: i, link: true });
    pager(i);
  }

  function reveal() {
    const els = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window)) { els.forEach(e => e.classList.add("in")); return; }
    const io = new IntersectionObserver(entries => entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }), { threshold: 0.12 });
    els.forEach(e => io.observe(e));
  }

  // Fill [data-v="path.to.value"] elements from the data, formatted by data-f (money | pct | months | int)
  function fillValues() {
    document.querySelectorAll("[data-v]").forEach(el => {
      const v = el.dataset.v.split(".").reduce((o, k) => (o == null ? o : o[k]), D);
      const f = el.dataset.f;
      el.textContent = f === "money" ? money(v) : f === "pct" ? pct(v) : f === "months" ? months(v)
        : f === "int" ? Math.round(v).toLocaleString() : v;
    });
  }

  nav(); footer(); fillValues();
  document.addEventListener("DOMContentLoaded", reveal);
  window.Site = { D, PHASES, REPO, money, pct, months, doc, app, stairs, phaseHead, delivered };
})();
