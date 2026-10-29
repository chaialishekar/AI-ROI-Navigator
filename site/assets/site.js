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
    const live3 = Math.max(0, 36 - c.months_to_deploy), setup = c.deployment_cost + (c.change_cost || 0);
    return { caught, missed, false_alarms, net, recall: tp / D.curve.failures, precision: tp + fp ? tp / (tp + fp) : 0,
             payback: net > 0 ? c.months_to_deploy + setup / (net / 12) : null,
             three_year: net * live3 / 12 - setup };
  }
  // Break-even alert level, as src/value.py alert_threshold()
  const breakEven = c => (c.cost_missed > c.cost_caught
    ? Math.min(0.99, Math.max(0.01, Math.round(c.cost_false_alarm / (c.cost_missed - c.cost_caught + c.cost_false_alarm) * 100) / 100)) : 0.99);

  const delivered = D.tracking.delivered / D.tracking.promised;   // share of the model's estimate
  const PHASES = [
    { slug: "discover", name: "Discover", q: "What's going wrong, and what does it cost?",
      answer: `${money(D.plant.total)} a year lost to downtime, scrap and shortages.`, made: "Problem statement, stakeholders, as-is process", gate: [money(D.plant.total), "a year at stake"] },
    { slug: "evaluate", name: "Evaluate options", q: "Which AI should they buy first?",
      from: `Discover put ${money(D.plant.total)} a year of losses on the table.`, now: "Which AI tool, if any, attacks them best?",
      answer: "Predictive maintenance first. The runner-up rests on a vendor's own claim.", made: "Priced options with sources, cautious view", gate: ["Predictive", "maint. first"] },
    { slug: "business-case", name: "Business case", q: "Will it pay for itself, and how sure are we?",
      from: "Evaluate picked Doc, predictive maintenance, on published results.", now: "Test it on machine data and price it for this plant.",
      answer: `Plan on ${money(D.published.net)} a year. Still ${money(D.decision.worst_case)} in the worst case.`, made: "ROI model with ranges, break-even", gate: [`${money(D.published.net)} a year`, "planned"] },
    { slug: "readiness", name: "Readiness", q: "Can the team actually use it?",
      from: `The business case says it pays: ${money(D.published.net)} a year planned, ${money(D.decision.worst_case)} in the worst case.`, now: "Money isn't enough. Is the team ready to act on warnings?",
      answer: `Not yet: ${D.readiness.weakest.length} gaps in how warnings reach the crew, and whether they trust them.`, made: "Readiness check, change impact", gate: [`${D.readiness.weakest.length} gaps`, "to close"] },
    { slug: "decide", name: "Decide", q: "What exactly is finance signing?",
      from: `Readiness found ${D.readiness.weakest.length} gaps: ${D.readiness.weakest.map(w => w.name.toLowerCase()).join(" and ")}.`, now: "Put the money and the gaps in front of finance.",
      answer: "Not \"buy it\": pilot it on one line, after the gaps are closed.", made: "Two-page decision brief, options considered",
      gate: D.decision.label === "Proceed after fixes" ? ["Pilot after", "fixes"] : [D.decision.label, ""] },
    { slug: "deliver", name: "Deliver", q: "What gets built, and how do we know it works?",
      from: "Decide approved a pilot on one line, after the readiness gaps are closed.", now: "Specify it, build it, and set the checks the pilot must pass.",
      answer: "Specified, built and tested. The pilot had to pass three exit checks.", made: "Requirements, integration spec, 13 UATs, traceability", gate: ["13 tests", "pass"] },
    { slug: "measure", name: "Measure", q: "Did the savings show up?",
      from: "Deliver built Doc and ran the one-line pilot against its exit checks.", now: "A year after go-live, did the savings show up?",
      answer: `${money(D.tracking.delivered)} in year one against ${money(D.tracking.plan)} planned, and why it missed the model.`, made: "Monthly tracking, gap analysis by cause", gate: [`${money(D.tracking.delivered)} vs`, `${money(D.tracking.plan)} plan`] },
  ];
  // The short version of each phase, shown under the home staircase
  const L = D.plant.losses;
  const POINTS = [
    [`Downtime ${money(L[0].cost)}, scrap and rework ${money(L[1].cost)}, shortages ${money(L[2].cost)} a year.`,
      "Put a price on the problem before talking about any AI.", "Mapped who is affected and how a breakdown is handled today."],
    ["Priced every AI in the provider's library against the plant's own losses. Three fit.",
      "Judged each one on cautious figures too, and on how strong its evidence is.", "Every AI also had to beat the manual's rules and doing nothing."],
    [`The model test said ${money(D.value.annual_net)} a year. Finance plans on ${money(D.published.net)}, from published studies.`,
      `Still saves ${money(D.decision.worst_case)} a year if costs are worse than expected.`,
      `The manual's rules save more on paper, but ${Math.round(D.rules.false_alarms)} false alarms a year wear a crew down.`],
    ["Rated the team on ten questions about data, process and people.",
      `Weakest: ${D.readiness.weakest.map(w => w.name.toLowerCase()).join(" and ")}.`, "Listed who changes how they work, and what they need first."],
    ["The decision rule was agreed before the numbers came in.", "It saves money even in the worst case, but the team isn't ready yet.",
      "Finance signs a two-page brief that also shows the options it turned down."],
    ["36+ user stories and 13 acceptance tests, with every Must-have traced to code.",
      "A working model, an S3 → Databricks data path, and two GenAI experiments on Amazon Bedrock.", "The pilot had to pass three exit checks before go-live."],
    [`${money(D.tracking.delivered)} in year one: above the ${money(D.tracking.plan)} plan, ${Math.round(delivered * 100)}% of the model's estimate.`,
      "Fewer breakdowns than assumed, a lower catch rate and skipped alerts explain the gap.", "Checked every month through the S3 → Databricks pipeline."],
  ];
  const href = p => `${ROOT}/phases/${p.slug}.html`;

  function nav() {
    const el = document.querySelector("[data-nav]");
    if (!el) return;
    el.innerHTML = `<div class="wrap">
      <a class="logo" href="${ROOT}/index.html"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="15" width="5" height="7" rx="1" fill="#111"/><rect x="9.5" y="10" width="5" height="12" rx="1" fill="#111"/><rect x="17" y="4" width="5" height="18" rx="1" fill="#0f6d74"/></svg>The ROI Compass</a>
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

  // The 3D staircase. opts: current (index to highlight), link (steps open their page), panel (element for details),
  // tour (clicking a step opens its short version in the panel instead of leaving the page)
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
    const LIVE = `<b>Go-live!</b> Doc is now watching every machine on the floor. Next question: did the savings show up?`;
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
      el.querySelectorAll(".stairs-v a").forEach((a, i) => { a.classList.toggle("on", i === sel); a.classList.toggle("done", i !== sel && seen.includes(i)); });
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
      if (opts.panel && opts.tour) {
        const p = PHASES[sel];
        opts.panel.innerHTML = `<div class="tour-head"><span class="label">Phase ${sel + 1} of 7 · ${p.name}</span>
            <span class="tour-nav"><button class="chip" type="button" data-go="-1" ${sel ? "" : "disabled"}>← Back</button><button class="chip" type="button" data-go="1" ${sel < n - 1 ? "" : "disabled"}>Next step →</button></span></div>
          <h3>${p.q}</h3><p class="tour-answer">${p.answer}</p>
          <ul>${POINTS[sel].map(t => `<li>${t}</li>`).join("")}</ul>
          <a class="tour-more" href="${href(p)}">Read the full ${p.name} page →</a>`;
      } else if (opts.panel) {
        const p = PHASES[sel];
        opts.panel.innerHTML = `<div><span>Phase ${sel + 1} · ${p.name}</span>${p.q}</div>
          <div><span>The answer</span>${p.answer}</div><div><span>Deliverable</span>${p.made}</div>`;
      }
    }
    let climbing = null, pinned = sel;
    const stop = () => { clearInterval(climbing); climbing = null; };
    const go = i => { stop(); pinned = sel = Math.max(0, Math.min(n - 1, i)); paint(); };
    el.querySelectorAll(".step").forEach(g => {
      g.addEventListener("mouseenter", () => { stop(); sel = +g.dataset.i; paint(); });
      if (!opts.link || opts.tour) g.addEventListener("click", e => {
        e.preventDefault();
        if (opts.tour) { go(+g.dataset.i); opts.panel.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
      });
    });
    if (opts.tour) {
      // Phones get the vertical list: it steps through the tour too, and brings the short version into view
      el.querySelectorAll(".stairs-v a").forEach((a, i) => a.addEventListener("click", e => {
        e.preventDefault(); go(i); opts.panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }));
      opts.panel.addEventListener("click", e => { const d = e.target.dataset.go; if (d) go(sel + +d); });
      el.addEventListener("mouseleave", () => { sel = pinned; paint(); });
    } else if (!opts.link || opts.current === undefined) el.addEventListener("mouseleave", () => { sel = opts.current ?? sel; paint(); });
    paint();
    // opts.auto: climb from the first step to the top once the staircase is on screen
    function climb() {
      stop(); pinned = sel = 0; paint();
      if (matchMedia("(prefers-reduced-motion: reduce)").matches) { pinned = sel = n - 1; paint(); return; }
      climbing = setInterval(() => { pinned = sel += 1; paint(); if (sel >= n - 1) stop(); }, 750);
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
    const prev = PHASES[i - 1], next = PHASES[i + 1], last = PHASES[n - 1];
    el.innerHTML = `<div class="wrap">
      ${next ? `<a href="${href(next)}" class="upnext"><div class="label">Up next · Phase ${i + 2} of 7 · ${next.name}</div>
        <h3>${next.q}</h3><span class="go">Continue →</span></a>` : `<div class="upnext done"><div class="label">That's the whole case</div>
        <h3>Seven phases, from ${money(D.plant.total)} of losses to ${money(D.tracking.delivered)} delivered.</h3>
        <a class="btn" href="${ROOT}/index.html#decisions">See the key decisions</a> <a class="btn ghost" href="${REPO}" target="_blank" rel="noopener">View the code ↗</a></div>`}
      <div class="pager-links small">${prev ? `<a href="${href(prev)}">← ${prev.name}</a>` : `<a href="${ROOT}/index.html">← Home</a>`}
        ${i < n - 2 ? `<a href="${href(last)}">Skip to the ending: ${last.q.toLowerCase().replace("?", "")} →</a>` : ""}</div></div>`;
    // A small "next" button once the reader is most of the way down, hidden when the card above is on screen
    if (!next) return;
    const fab = document.createElement("a");
    fab.className = "nextfab"; fab.href = href(next); fab.innerHTML = `Next: ${next.name} <span aria-hidden="true">→</span>`;
    document.body.appendChild(fab);
    let pagerSeen = false;
    new IntersectionObserver(([e]) => { pagerSeen = e.isIntersecting; tick(); }).observe(el);
    const tick = () => fab.classList.toggle("show", !pagerSeen && scrollY + innerHeight > document.body.scrollHeight * 0.45);
    addEventListener("scroll", tick, { passive: true });
  }

  // A phase page's header: mini staircase plus the phase's question
  function phaseHead(i) {
    const p = PHASES[i];
    document.title = `${p.name} · The ROI Compass`;
    markVisited(i);
    const head = document.querySelector("[data-phase-head]");
    if (head) head.innerHTML = `<p class="eyebrow">Phase ${i + 1} of 7 · ${p.name}</p><h1>${p.q}</h1>` +
      (p.from ? `<p class="bridge"><span class="muted">${p.from}</span> <b>Now:</b> ${p.now}</p>` : "");
    rail(i);
    if (i === n - 1) toast(`<b>Go-live!</b> Doc went live across the plant. This page checks whether the savings showed up.`);
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
  window.Site = { D, PHASES, REPO, ESSAY, money, usd, pct, months, doc, value, rulesValue, breakEven, stairs, phaseHead, delivered };
})();
