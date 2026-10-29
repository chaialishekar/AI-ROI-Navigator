// Shared pieces for every page: nav, footer, the phase staircase, prev / next links, number formats.
(function () {
  const D = window.CASE;
  const REPO = "https://github.com/chaialishekar/AI-ROI-Navigator";
  // Chaitali's Substack post on revisiting AI decisions. Links to it stay hidden until it is set.
  const ESSAY = { url: null, title: "AI's Hidden Problem: Nobody Revisits the Original Decision" };
  const ROOT = document.body.dataset.root || ".";

  const money = (x, digits) => {
    if (x === null || x === undefined) return "n/a";
    const sign = x < 0 ? "−" : "", a = Math.abs(x);
    if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(digits ?? 1)}M`;
    if (a >= 1e3) return `${sign}$${Math.round(a / 1e3)}k`;
    return `${sign}$${Math.round(a)}`;
  };
  const usd = x => `${x < 0 ? "−" : ""}$${Math.round(Math.abs(x)).toLocaleString("en-US")}`;
  const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
  const months = x => (x === null ? "never" : `${x.toFixed(1)} months`);
  const doc = path => `${REPO}/blob/main/${path}`;

  // Same formula as src/value.py evaluate(): the test-set counts at an alert level, scaled to the plant's breakdowns
  function value(c, level) {
    const i = Math.min(99, Math.max(1, Math.round(level * 100))) - 1;
    return priced(c, ...D.curve.levels[i]);
  }
  // The maintenance manual's rules, priced the same way (same test records, same costs)
  const rulesValue = c => priced(c, D.rules.tp, D.rules.fp);
  function priced(c, tp, fp) {
    const scale = c.failures_per_year / D.curve.failures;
    const caught = tp * scale, missed = (D.curve.failures - tp) * scale, false_alarms = fp * scale;
    const net = c.failures_per_year * c.cost_missed
      - (caught * c.cost_caught + missed * c.cost_missed + false_alarms * c.cost_false_alarm + c.running_cost);
    const live3 = Math.max(0, 36 - c.months_to_deploy);
    return { caught, missed, false_alarms, net, recall: tp / D.curve.failures, precision: tp + fp ? tp / (tp + fp) : 0,
             payback: net > 0 ? c.months_to_deploy + c.deployment_cost / (net / 12) : null,
             three_year: net * live3 / 12 - c.deployment_cost };
  }
  // The board: three executives react to the numbers. Plain rules, no live AI: each reaction names the number behind it.
  // Thresholds are set for this plant: 18 months is the budget horizon; an alert level 10+ points above break-even
  // that costs over $10k a year is accuracy bought with money; crews tune out when fewer than half the alerts are
  // real or false alarms pass one a week (US-6.2 adoption risk); the COO also reads the Readiness gaps; a catch
  // rate under 80% leaves 8+ surprise breakdowns a year; extra misses only count when a lower alert level would avoid them. Returns [{role, who, mood: "backs" | "concerns" | "blocks",
  // say, fix}], where fix (optional) is {label, set: {cost: value}}, {label, level} or {label, href}: the change that
  // would answer the concern. The board never changes anything itself; the page applies a fix only when clicked.
  function board(c, r, level) {
    const wk = r.false_alarms / 52, gross = r.net + c.running_cost, be = breakEven(c), atBe = value(c, be);
    const lost = atBe.net - r.net, gaps = D.readiness.weakest.filter(w => w.score < 3.5);
    const toBe = { label: `Apply: go back to the ${Math.round(be * 100)}% break-even level`, level: be };
    // Lowest alert level that keeps false alarms to one a week and at least half the alerts real
    const calm = [...Array(99).keys()].map(k => (k + 1) / 100).find(t => { const v = value(c, t); return v.false_alarms <= 52 && v.precision >= 0.5; });
    const toCalm = calm && calm > level ? { label: `Apply: raise the alert level to ${Math.round(calm * 100)}%`, level: calm } : null;
    const maxSetup = Math.floor((18 - c.months_to_deploy) * r.net / 12 / 5000) * 5000;
    const cfo = r.net <= 0 ? ["blocks", `It loses ${usd(-r.net)} a year at these costs. I can't take that to the board.`, null]
      : r.payback > 18 ? ["blocks", `${r.payback.toFixed(1)} months to pay back is past our 18-month budget horizon. Cut the ${usd(c.deployment_cost)} setup or show more savings.`,
          maxSetup > 0 ? { label: `Apply: cap setup at ${usd(maxSetup)}`, set: { deployment_cost: maxSetup } } : null]
      : level > be + 0.1 && lost > 10000 ? ["concerns", `An alert level of ${Math.round(level * 100)}% gives up ${usd(lost)} a year against the ${Math.round(be * 100)}% break-even level, to avoid ${Math.round(atBe.false_alarms - r.false_alarms)} inspections. Accuracy isn't the goal; savings are.`, toBe]
      : c.running_cost > 0.35 * gross ? ["concerns", `Running costs eat ${Math.round(c.running_cost / gross * 100)}% of the gross saving. I want a fixed-price contract before I sign.`,
          { label: `Apply: negotiate running cost to ${usd(Math.floor(0.35 * gross / 1000) * 1000)}`, set: { running_cost: Math.floor(0.35 * gross / 1000) * 1000 } }]
      : r.payback > 12 ? ["concerns", `${r.payback.toFixed(1)} months to pay back. Fine, but release the money in stages: one production line first.`,
          { label: "See the staged rollout in Decide →", href: "decide.html" }]
      : ["backs", `Pays back in ${r.payback.toFixed(1)} months and nets ${usd(r.net)} a year. I'll still budget on the published studies, not the model test, and treat the rest as upside.`];
    const coo = r.precision < 0.3 ? ["blocks", `Only ${Math.round(r.precision * 100)}% of alerts are real. My crew will stop answering them inside a month.`, toCalm]
      : wk > 1 ? ["concerns", `${Math.round(r.false_alarms)} false alarms a year is more than one a week. Expect technicians to start ignoring it.`, toCalm]
      : r.precision < 0.5 ? ["concerns", `Half the alerts are wrong (${Math.round(r.precision * 100)}% real). Show the reason with each one, or trust goes.`, toCalm]
      : level > be && r.missed - atBe.missed >= 1 ? ["concerns", `This alert level lets through ${(r.missed - atBe.missed).toFixed(1)} more surprise breakdowns a year than break-even would. Those are the night-shift call-outs I wanted gone.`, toBe]
      : D.readiness.verdict !== "Go" && gaps.length ? ["concerns", `${Math.round(r.false_alarms)} false alarms a year is fine. My worry is the crew: readiness scored "${gaps.map(g => g.name.toLowerCase()).join("\" and \"")}" at ${gaps.map(g => g.score.toFixed(1)).join(" and ")} of 5. Fix that before go-live.`,
          { label: "See the readiness fixes →", href: "readiness.html#details" }]
      : ["backs", `About ${Math.round(r.false_alarms)} false alarms a year, one every ${Math.round(52 / Math.max(r.false_alarms, 0.1))} weeks. My crew can work with that if alerts land in the maintenance system.`];
    const cto = r.recall < 0.6 ? ["blocks", `At a ${Math.round(level * 100)}% alert level it catches only ${Math.round(r.recall * 100)}% of breakdowns. We'd be paying for a model and still missing most failures.`, level > be ? toBe : null]
      : r.recall < 0.8 ? ["concerns", `Tuned this cautiously, it misses ${Math.round((1 - r.recall) * 100)}% of breakdowns. Lower the alert level before go-live.`, level > be ? toBe : null]
      : c.running_cost < 30000 ? ["concerns", `${usd(c.running_cost)} a year won't cover monitoring and retraining. The model will drift and nobody will notice.`,
          { label: "Apply: budget $30,000 a year for monitoring", set: { running_cost: 30000 } }]
      : ["backs", `Catches ${Math.round(r.recall * 100)}% on the test. The data is synthetic, so I want the 3-month catch rate monitored from day one.`];
    return [["CFO", "Finance chief", ...cfo], ["COO", "Operations chief", ...coo], ["CTO", "Technology chief", ...cto]]
      .map(([role, who, mood, say, fix]) => ({ role, who, mood, say, fix }));
  }

  // Break-even alert level, as src/value.py alert_threshold()
  const breakEven = c => (c.cost_missed > c.cost_caught
    ? Math.min(0.99, Math.max(0.01, Math.round(c.cost_false_alarm / (c.cost_missed - c.cost_caught + c.cost_false_alarm) * 100) / 100)) : 0.99);

  const delivered = D.tracking.delivered / D.tracking.promised;   // share of the model's estimate
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
      made: "Monthly tracking, gap analysis by cause", gate: [`${money(D.tracking.delivered)} vs`, `${money(D.tracking.plan)} plan`] },
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
        <a class="btn" href="${ROOT}/phases/business-case.html#details">Try the model</a>
      </div></div>`;
  }

  function footer() {
    const el = document.querySelector("[data-footer]");
    if (!el) return;
    el.innerHTML = `<div class="wrap">
      <div><h2 style="margin:0">Built by Chaitali Shekar</h2></div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        ${ESSAY.url ? `<a class="btn ghost" href="${ESSAY.url}" target="_blank" rel="noopener">Read my writing ↗</a>` : ""}
        <a class="btn ghost" href="${REPO}" target="_blank" rel="noopener">View the code</a>
        <a class="btn" href="${ROOT}/phases/business-case.html#details">Try the model</a></div></div>`;
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
    const LIVE = `<b>Go-live!</b> Sentry is now watching every machine on the floor. Next question: did the savings show up?`;
    el.innerHTML = `<svg viewBox="0 0 ${20 + 7 * w + dx + 10} ${ly + 60}" role="img" aria-label="Seven phases drawn as a rising staircase, each with its gate, and a person climbing it">
      <defs><marker id="arTeal" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#0f6d74" stroke-width="1.8" stroke-linecap="round"/></marker><marker id="ar" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#8a8a85" stroke-width="1.5"/></marker></defs>${s}</svg>
      <div class="stairs-v">${vertical}</div>
      <div class="golive-msg" role="status" aria-live="polite"></div>`;

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
      // Reaching the top step means the project went live
      const msg = el.querySelector(".golive-msg"), top = sel === n - 1;
      msg.innerHTML = top ? LIVE : "";
      msg.classList.toggle("show", top);
      el.classList.toggle("is-live", top);
      if (opts.panel) {
        const p = PHASES[sel];
        opts.panel.innerHTML = `<div><span>Phase ${sel + 1} · ${p.name}</span>${p.q}</div>
          <div><span>Deliverable</span>${p.made}</div><div><span>Gate passed</span>${p.gate.join(" ")}</div>`;
      }
    }
    let climbing = null;
    const stop = () => { clearInterval(climbing); climbing = null; };
    el.querySelectorAll(".step").forEach(g => {
      g.addEventListener("mouseenter", () => { stop(); sel = +g.dataset.i; paint(); });
      if (!opts.link) g.addEventListener("click", e => e.preventDefault());
    });
    if (!opts.link || opts.current === undefined) el.addEventListener("mouseleave", () => { sel = opts.current ?? sel; paint(); });
    paint();
    // opts.auto: climb from the first step to the top once the staircase is on screen
    function climb() {
      stop(); sel = 0; paint();
      if (matchMedia("(prefers-reduced-motion: reduce)").matches) { sel = n - 1; paint(); return; }
      climbing = setInterval(() => { sel += 1; paint(); if (sel >= n - 1) stop(); }, 750);
    }
    if (opts.auto) {
      if (opts.replay) opts.replay.addEventListener("click", climb);
      if ("IntersectionObserver" in window) {
        const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { io.disconnect(); climb(); } }, { threshold: 0.5 });
        io.observe(el);
      } else { sel = n - 1; paint(); }
    }
  }

  // Phase pages: a small staircase that stays under the nav while you scroll; each step opens its phase
  function rail(i) {
    const nav = document.querySelector("[data-nav]");
    if (!nav) return;
    const seen = visited(), climber = `<svg class="rail-climber" viewBox="-8 -32 20 34" aria-hidden="true"><path d="M-4 0 L-1 -11 M6 -6 L2 -11" stroke="#111" stroke-width="2.6" stroke-linecap="round" fill="none"/><path d="M0 -11 L2 -22" stroke="#0f6d74" stroke-width="6" stroke-linecap="round"/><path d="M1.5 -20 L9 -27 M1.5 -20 L-5 -15" stroke="#111" stroke-width="2.4" stroke-linecap="round" fill="none"/><circle cx="3.5" cy="-28.5" r="4.2" fill="#111"/></svg>`;
    const steps = PHASES.map((p, j) => `${j === n - 1 ? `<span class="rail-live" title="Go-live">Go-live</span>` : ""}
      <a href="${href(p)}" class="${j === i ? "on" : seen.includes(j) ? "done" : ""}" ${j === i ? 'aria-current="page"' : ""}>
        ${j === i ? climber : ""}<span class="blk" style="height:${8 + j * 5}px"></span><span class="nm"><em>0${j + 1}</em> ${p.name}</span></a>`).join("");
    nav.insertAdjacentHTML("beforeend", `<div class="wrap rail" aria-label="Phases">${steps}</div>`);
  }
  const n = PHASES.length;

  // A short note that slides in and fades out
  function toast(html) {
    const t = document.createElement("div");
    t.className = "toast"; t.setAttribute("role", "status"); t.innerHTML = html;
    document.body.appendChild(t);
    requestAnimationFrame(() => t.classList.add("show"));
    setTimeout(() => { t.classList.remove("show"); setTimeout(() => t.remove(), 600); }, 6000);
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
    rail(i);
    if (i === n - 1) toast(`<b>Go-live!</b> Sentry went live across the plant. This page checks whether the savings showed up.`);
    pager(i);
    detailsToggle();
  }

  // "Show the details": every phase page keeps its full working in a hidden section. #details in the URL opens it.
  function detailsToggle() {
    const sec = document.getElementById("details"), btns = document.querySelectorAll("[data-details-btn]");
    if (!sec) return;
    const set = (open, scroll) => {
      sec.hidden = !open;
      btns.forEach(b => { b.textContent = open ? "Hide the details" : "Show the details"; b.setAttribute("aria-expanded", open); });
      if (open && scroll) sec.scrollIntoView({ behavior: "smooth" });
      if (open) sec.querySelectorAll(".reveal").forEach(e => e.classList.add("in"));
    };
    btns.forEach(b => b.addEventListener("click", () => set(sec.hidden, true)));
    set(location.hash === "#details", location.hash === "#details");
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
  window.Site = { D, PHASES, REPO, ESSAY, money, usd, pct, months, doc, value, rulesValue, breakEven, board, stairs, phaseHead, delivered };
})();
