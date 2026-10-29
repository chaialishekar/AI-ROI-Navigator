// The five contenders, as puffy fabric characters (Three.js). Each one always stands for the same option:
// sentry = predictive maintenance (shown as Doc), lens = visual inspection, foresight = demand forecasting,
// rulebook = the maintenance manual's rules (no AI), snooze = do nothing.
(function () {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  function tex(draw, rep) {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    draw(c.getContext("2d"));
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(rep, rep);
    t.encoding = THREE.sRGBEncoding;
    return t;
  }
  const fill = (g, color) => { g.fillStyle = color; g.fillRect(0, 0, 256, 256); };
  const PATTERNS = {
    gingham: (b, l) => tex(g => { fill(g, b); g.fillStyle = l; for (let i = 0; i < 256; i += 32) { g.fillRect(i, 0, 12, 256); g.fillRect(0, i, 256, 12); } }, 1.3),
    dots: (b, l) => tex(g => { fill(g, b); g.fillStyle = l; for (let y = 0; y < 256; y += 32) for (let x = (y / 32 % 2) * 16; x < 256; x += 32) { g.beginPath(); g.arc(x + 8, y + 8, 5, 0, 7); g.fill(); } }, 1.2),
    squiggle: (b, l) => tex(g => { fill(g, b); g.strokeStyle = l; g.lineWidth = 4; g.lineCap = "round"; let s = 7; const rnd = () => (s = (s * 9301 + 49297) % 233280) / 233280; for (let k = 0; k < 40; k++) { const x = rnd() * 256, y = rnd() * 256; g.beginPath(); g.moveTo(x, y); g.bezierCurveTo(x + 10, y - 12, x + 18, y + 12, x + 28, y); g.stroke(); } }, 1.1),
    ruled: (b, l) => tex(g => { fill(g, b); g.strokeStyle = l; g.lineWidth = 3; for (let y = 12; y < 256; y += 24) { g.beginPath(); g.moveTo(0, y); g.lineTo(256, y); g.stroke(); } g.strokeStyle = "rgba(200,40,40,.45)"; g.beginPath(); g.moveTo(40, 0); g.lineTo(40, 256); g.stroke(); }, 1),
  };

  function polar(f, n = 160) {
    const s = new THREE.Shape();
    for (let i = 0; i <= n; i++) {
      const a = i / n * Math.PI * 2, r = f(a);
      i ? s.lineTo(Math.cos(a) * r, Math.sin(a) * r) : s.moveTo(Math.cos(a) * r, Math.sin(a) * r);
    }
    return s;
  }
  function roundRect(w, h, r) {
    const s = new THREE.Shape(), x = -w / 2, y = -h / 2;
    s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r);
    s.lineTo(x + w, y + h - r); s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    s.lineTo(x + r, y + h); s.quadraticCurveTo(x, y + h, x, y + h - r);
    s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
    return s;
  }
  // Each character is shaped like what it does: a gear with a stethoscope, a magnifying glass, a telescope, a book
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const M = (geo, mat, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.rotation.set(rx, ry, rz); return m; };
  const cloth = color => new THREE.MeshStandardMaterial({ color, bumpMap: weave, bumpScale: 0.02, roughness: 0.9 });
  const metal = color => new THREE.MeshStandardMaterial({ color, roughness: 0.3, metalness: 0.45 });
  const glass = () => new THREE.MeshStandardMaterial({ color: 0xdff3ff, roughness: 0.05, transparent: true, opacity: 0.32 });
  function label(text, bg, ink) {
    const c = document.createElement("canvas"); c.width = 256; c.height = 80;
    const g = c.getContext("2d"); g.fillStyle = bg; g.fillRect(0, 0, 256, 80);
    g.fillStyle = ink; g.font = "700 46px Inter, Arial, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, 128, 43);
    const t = new THREE.CanvasTexture(c); t.encoding = THREE.sRGBEncoding; return t;
  }

  const DEFS = {
    sentry: { name: "Doc", role: "Predictive maintenance", eyeY: 0.22, tilt: 0.12,
              shape: () => polar(a => 0.98 + 0.16 * Math.tanh(4 * Math.cos(10 * a))),
              fabric: () => PATTERNS.gingham("#1f8f86", "rgba(255,255,255,.28)"),
              extras: () => {
                const tube = metal(0x2f3338), steel = metal(0xc9ced6);
                const loop = new THREE.CatmullRomCurve3([V(-0.7, 0.95, 0.4), V(-0.74, 0.15, 0.7), V(-0.42, -0.42, 0.8), V(0, -0.56, 0.82), V(0.42, -0.42, 0.8), V(0.74, 0.15, 0.7), V(0.7, 0.95, 0.4)]);
                const drop = new THREE.CatmullRomCurve3([V(0, -0.56, 0.82), V(0.12, -0.86, 0.9), V(0.34, -1.04, 0.95)]);
                return [M(new THREE.TubeGeometry(loop, 64, 0.055, 10), tube), M(new THREE.TubeGeometry(drop, 16, 0.055, 10), tube),
                  M(new THREE.CylinderGeometry(0.2, 0.2, 0.1, 32), steel, 0.38, -1.1, 0.98, 1.2),
                  M(new THREE.SphereGeometry(0.08, 16, 12), steel, -0.7, 0.97, 0.4), M(new THREE.SphereGeometry(0.08, 16, 12), steel, 0.7, 0.97, 0.4)];
              } },
    lens: { name: "Lens", role: "Visual inspection", eyeY: 0, eyeX: 0.25, eyeS: 0.8, eyeZ: 0.08, tilt: -0.08,
            shape: () => { const s = polar(() => 1.08), hole = new THREE.Path(); hole.absarc(0, 0, 0.72, 0, Math.PI * 2, true); s.holes.push(hole); return s; },
            extrude: { depth: 0.25, bevelThickness: 0.25, bevelSize: 0.12 },
            fabric: () => PATTERNS.dots("#e2603f", "#ffe7d6"),
            extras: () => [M(new THREE.CircleGeometry(0.75, 48), glass(), 0, 0, -0.06),
              M(new THREE.CylinderGeometry(0.2, 0.2, 0.22, 24), metal(0xd9d2c5), 0.96, -0.8, 0, 0, 0, 0.87),
              M(new THREE.CylinderGeometry(0.17, 0.2, 0.7, 24), cloth(0x8a4b2a), 1.22, -1.02, 0, 0, 0, 0.87)] },
    foresight: { name: "Foresight", role: "Demand forecasting", eyeY: 0.1, eyeZ: 0.5, tilt: 0.22,
              geometry: () => new THREE.CylinderGeometry(0.5, 0.64, 1.9, 48, 1).rotateZ(-Math.PI / 2),
              fabric: () => PATTERNS.squiggle("#6b5fd8", "#ffd36b"),
              extras: () => {
                const dark = cloth(0x3a3360), legs = metal(0x3a3360);
                return [M(new THREE.TorusGeometry(0.64, 0.09, 16, 48), metal(0xffd36b), 0.95, 0, 0, 0, Math.PI / 2),
                  M(new THREE.CircleGeometry(0.62, 48), glass(), 0.97, 0, 0, 0, Math.PI / 2),
                  M(new THREE.CylinderGeometry(0.17, 0.22, 0.4, 24), dark, -1.12, 0, 0, 0, 0, Math.PI / 2),
                  M(new THREE.SphereGeometry(0.16, 16, 12), legs, 0, -0.56, 0),
                  M(new THREE.CylinderGeometry(0.05, 0.05, 0.8, 10), legs, -0.22, -0.93, 0.12, 0.12, 0, -0.32),
                  M(new THREE.CylinderGeometry(0.05, 0.05, 0.8, 10), legs, 0.22, -0.93, 0.12, 0.12, 0, 0.32),
                  M(new THREE.CylinderGeometry(0.05, 0.05, 0.8, 10), legs, 0, -0.93, -0.25, -0.35, 0, 0)];
              } },
    rulebook: { name: "Rulebook", role: "The manual's rules (no AI)", eyeY: -0.05, eyeZ: 0.5, tilt: -0.1, bodyZ: 0.36,
                shape: () => roundRect(1.8, 2.05, 0.12), extrude: { depth: 0.06, bevelThickness: 0.05, bevelSize: 0.05, bevelSegments: 3 },
                fabric: () => PATTERNS.ruled("#f2b33d", "rgba(160,95,0,.12)"),
                extras: mat => {
                  const pages = new THREE.MeshStandardMaterial({ map: PATTERNS.ruled("#fbf6e9", "rgba(0,0,0,.10)"), roughness: 0.95 });
                  const back = new THREE.Mesh(geometryFor("rulebook").geo, mat); back.position.z = -0.36;
                  return [back, M(new THREE.BoxGeometry(1.66, 1.92, 0.62), pages, 0.05, 0, 0),
                    M(new THREE.BoxGeometry(0.24, 2.12, 0.92), cloth(0xd9962a), -0.9, 0, 0),
                    M(new THREE.BoxGeometry(0.95, 0.3, 0.03), new THREE.MeshStandardMaterial({ map: label("RULES", "#fff4d6", "#9a5b00"), roughness: 0.8 }), 0.06, 0.66, 0.47),
                    M(new THREE.BoxGeometry(0.14, 0.5, 0.03), cloth(0xd8453b), 0.5, -1.2, 0.12)];
                } },
    snooze: { name: "Snooze", role: "Do nothing", eyeY: 0.1, tilt: 0.09, sleepy: true,
              shape: () => polar(a => (1 + 0.11 * Math.cos(6 * a)) * (Math.sin(a) > 0 ? 1 : 0.82)),
              fabric: () => PATTERNS.dots("#9fb3c8", "rgba(255,255,255,.55)") },
  };

  let weave;
  const cache = {};
  function geometryFor(key) {
    if (!cache[key]) {
      const d = DEFS[key];
      const geo = d.geometry ? d.geometry() : new THREE.ExtrudeGeometry(d.shape(), {
        depth: 0.25, bevelEnabled: true, bevelThickness: 0.5, bevelSize: 0.3, bevelSegments: 10, curveSegments: 48, ...d.extrude });
      if (!d.geometry) geo.center();
      cache[key] = { geo, map: d.fabric() };
    }
    return cache[key];
  }

  // Flat stand-ins for browsers without WebGL (or when three.js fails to load), so the page never shows a gap
  const FLAT = { sentry: "#1f8f86", lens: "#e2603f", foresight: "#6b5fd8", rulebook: "#f2b33d", snooze: "#9fb3c8" };
  function flatCast(el, keys, opts) {
    el.classList.add("flat");
    el.innerHTML = keys.map(k => `<button type="button" data-k="${k}" style="--c:${FLAT[k]}" aria-label="${DEFS[k].name}: ${DEFS[k].role}">
      <span class="eyes${DEFS[k].sleepy ? " sleepy" : ""}"><i></i><i></i></span><b>${DEFS[k].name}</b></button>`).join("");
    if (opts.onSelect) el.querySelectorAll("button").forEach(b => {
      b.addEventListener("mouseenter", () => opts.onSelect(b.dataset.k));
      b.addEventListener("click", () => opts.onSelect(b.dataset.k));
    });
  }

  // Mount a row of characters into el. opts: scale, spacing, sink (how far they sit below the frame), onSelect(key)
  function mountCast(el, keys, opts = {}) {
    try { mount3d(el, keys, opts); } catch (e) { el.innerHTML = ""; flatCast(el, keys, opts); }
  }

  function mount3d(el, keys, opts) {
    if (!window.THREE) throw new Error("three.js not loaded");
    const probe = document.createElement("canvas");
    if (!(probe.getContext("webgl2") || probe.getContext("webgl"))) throw new Error("WebGL unavailable");
    const scale = opts.scale || 1.1, spacing = opts.spacing || 2.45, sink = opts.sink ?? 0.15;
    weave = weave || tex(g => { fill(g, "#808080"); let s = 3; const rnd = () => (s = (s * 9301 + 49297) % 233280) / 233280; for (let y = 0; y < 256; y += 4) for (let x = 0; x < 256; x += 4) { const v = 100 + rnd() * 80; g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(x, y, 3, 3); } }, 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputEncoding = THREE.sRGBEncoding;
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(30, 1, 0.1, 200);
    scene.add(new THREE.HemisphereLight(0xffffff, 0xe9e4dc, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 0.85); key.position.set(2, 4, 6); scene.add(key);
    const fillLight = new THREE.DirectionalLight(0xfff4e8, 0.3); fillLight.position.set(-5, 1, 3); scene.add(fillLight);

    const white = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.12 });
    const black = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.08 });
    const eyeG = new THREE.SphereGeometry(0.27, 32, 24), pupilG = new THREE.SphereGeometry(0.13, 24, 16);
    const glintG = new THREE.SphereGeometry(0.035, 10, 8);

    const chars = keys.map((k, i) => {
      const d = DEFS[k], { geo, map } = geometryFor(k);
      const mat = new THREE.MeshStandardMaterial({ map, bumpMap: weave, bumpScale: 0.025, roughness: 0.92 });
      const g = new THREE.Group();
      const body = new THREE.Mesh(geo, mat); body.position.z = d.bodyZ || 0; g.add(body);
      if (d.extras) d.extras(mat).forEach(m => g.add(m));
      const pupils = [], ex = d.eyeX || 0.33, es = d.eyeS || 1, ez = d.eyeZ ?? 0.62;
      [-ex, ex].forEach(x => {
        const eye = new THREE.Mesh(eyeG, white); eye.scale.set(es, es, 0.75 * es); eye.position.set(x, d.eyeY, ez); g.add(eye);
        const p = new THREE.Mesh(pupilG, black); p.scale.setScalar(es); p.position.set(x, d.eyeY, ez + 0.16 * es); g.add(p);
        const gl = new THREE.Mesh(glintG, white); gl.position.set(x - 0.04, d.eyeY + 0.05, ez + 0.28 * es); g.add(gl);
        pupils.push({ p, gl, x });
        if (d.sleepy) {
          const lid = new THREE.Mesh(new THREE.SphereGeometry(0.285, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.48), mat);
          lid.position.set(x, d.eyeY, ez); lid.rotation.x = 0.35; g.add(lid);
        }
      });
      g.scale.setScalar(scale);
      g.rotation.z = d.tilt;
      g.position.set((i - (keys.length - 1) / 2) * spacing, -sink, 0);
      scene.add(g);
      return { key: k, g, meshes: g.children, pupils, lift: 0, phase: i * 1.7, tilt: d.tilt, eyeY: d.eyeY };
    });

    function resize() {
      const w = el.clientWidth, h = el.clientHeight, aspect = w / h, half = Math.tan(15 * Math.PI / 180);
      const halfW = (spacing * (keys.length - 1)) / 2 + 1.25 * scale, halfH = 1.65 * scale;
      const byHeight = halfH / half, byWidth = halfW / (half * aspect);
      // Wide screens may crop the outer ones a little; phones fit the whole row
      const dist = aspect < 1.4 ? Math.max(byHeight, byWidth) : Math.max(byHeight, Math.min(byWidth, byHeight * 1.7));
      renderer.setSize(w, h);
      cam.aspect = aspect; cam.position.set(0, 0.2 * scale, dist); cam.lookAt(0, 0.2 * scale, 0);
      cam.updateProjectionMatrix();
    }
    resize();
    new ResizeObserver(resize).observe(el);

    const mouse = new THREE.Vector2(), ray = new THREE.Raycaster();
    let hover = -1;
    function pick(e) {
      const r = el.getBoundingClientRect();
      mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
      mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
      ray.setFromCamera(mouse, cam);
      const hit = ray.intersectObjects(chars.flatMap(c => c.meshes));
      const i = hit.length ? chars.findIndex(c => c.meshes.includes(hit[0].object)) : -1;
      if (i >= 0 && i !== hover && opts.onSelect) opts.onSelect(chars[i].key);
      hover = i;
    }
    el.addEventListener("pointermove", pick);
    el.addEventListener("pointerdown", pick);
    el.addEventListener("pointerleave", () => { hover = -1; mouse.set(0, 0); });

    let visible = true;
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(el);
    const clock = new THREE.Clock();
    (function loop() {
      requestAnimationFrame(loop);
      if (!visible) return;
      const t = clock.getElapsedTime();
      chars.forEach((c, i) => {
        c.lift += ((i === hover ? 0.5 : 0) - c.lift) * 0.1;
        c.g.position.y = -sink + c.lift + (reduce ? 0 : Math.sin(t * 1.4 + c.phase) * 0.04);
        c.g.rotation.z += ((i === hover ? 0 : c.tilt) - c.g.rotation.z) * 0.08;
        c.g.rotation.y += (mouse.x * (i === hover ? 0.45 : 0.18) - c.g.rotation.y) * 0.07;
        c.g.rotation.x += (-mouse.y * 0.12 - c.g.rotation.x) * 0.07;
        c.pupils.forEach(p => {
          p.p.position.x += (p.x + mouse.x * 0.1 - p.p.position.x) * 0.2;
          p.p.position.y += (c.eyeY + mouse.y * 0.09 - p.p.position.y) * 0.2;
          p.gl.position.x = p.p.position.x - 0.04; p.gl.position.y = p.p.position.y + 0.05;
        });
      });
      renderer.render(scene, cam);
    })();
  }

  window.Characters = { DEFS, mountCast };
})();
