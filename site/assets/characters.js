// The five contenders, as puffy fabric characters (Three.js). Each one always stands for the same option:
// sentry = predictive maintenance, lens = visual inspection, oracle = demand forecasting,
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
  function ring() {
    const s = polar(() => 1.05), hole = new THREE.Path();
    hole.absarc(0, -0.42, 0.26, 0, Math.PI * 2, true);
    s.holes.push(hole);
    return s;
  }

  const DEFS = {
    sentry: { name: "Sentry", role: "Predictive maintenance", eyeY: 0.22, tilt: 0.12,
              shape: () => polar(a => 0.95 + 0.17 * Math.tanh(3 * Math.cos(8 * a))),
              fabric: () => PATTERNS.gingham("#1f8f86", "rgba(255,255,255,.28)") },
    lens: { name: "Lens", role: "Visual inspection", eyeY: 0.42, tilt: -0.08, shape: ring,
            fabric: () => PATTERNS.dots("#e2603f", "#ffe7d6") },
    oracle: { name: "Oracle", role: "Demand forecasting", eyeY: 0.12, tilt: 0.06,
              shape: () => polar(a => 0.82 + 0.3 * Math.pow((1 + Math.cos(5 * (a - Math.PI / 2))) / 2, 1.6)),
              fabric: () => PATTERNS.squiggle("#6b5fd8", "#ffd36b") },
    rulebook: { name: "Rulebook", role: "The manual's rules (no AI)", eyeY: 0.3, tilt: -0.1,
                shape: () => roundRect(1.8, 1.95, 0.42), fabric: () => PATTERNS.ruled("#f2b33d", "rgba(120,70,0,.35)") },
    snooze: { name: "Snooze", role: "Do nothing", eyeY: 0.1, tilt: 0.09, sleepy: true,
              shape: () => polar(a => (1 + 0.11 * Math.cos(6 * a)) * (Math.sin(a) > 0 ? 1 : 0.82)),
              fabric: () => PATTERNS.dots("#9fb3c8", "rgba(255,255,255,.55)") },
  };

  let weave;
  const cache = {};
  function geometryFor(key) {
    if (!cache[key]) {
      const geo = new THREE.ExtrudeGeometry(DEFS[key].shape(), {
        depth: 0.25, bevelEnabled: true, bevelThickness: 0.5, bevelSize: 0.3, bevelSegments: 10, curveSegments: 48 });
      geo.center();
      cache[key] = { geo, map: DEFS[key].fabric() };
    }
    return cache[key];
  }

  // Mount a row of characters into el. opts: scale, spacing, sink (how far they sit below the frame), onSelect(key)
  function mountCast(el, keys, opts = {}) {
    if (!window.THREE) return;
    const scale = opts.scale || 1.1, spacing = opts.spacing || 2.45, sink = opts.sink ?? 0.3;
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
      g.add(new THREE.Mesh(geo, mat));
      const pupils = [];
      [-0.33, 0.33].forEach(x => {
        const eye = new THREE.Mesh(eyeG, white); eye.scale.z = 0.75; eye.position.set(x, d.eyeY, 0.62); g.add(eye);
        const p = new THREE.Mesh(pupilG, black); p.position.set(x, d.eyeY, 0.78); g.add(p);
        const gl = new THREE.Mesh(glintG, white); gl.position.set(x - 0.04, d.eyeY + 0.05, 0.9); g.add(gl);
        pupils.push({ p, gl, x });
        if (d.sleepy) {
          const lid = new THREE.Mesh(new THREE.SphereGeometry(0.285, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.48), mat);
          lid.position.set(x, d.eyeY, 0.62); lid.rotation.x = 0.35; g.add(lid);
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
      const halfW = (spacing * (keys.length - 1)) / 2 + 1.25 * scale, halfH = 1.55 * scale;
      const byHeight = halfH / half, byWidth = halfW / (half * aspect);
      // Wide screens may crop the outer ones a little; phones fit the whole row
      const dist = aspect < 1.4 ? Math.max(byHeight, byWidth) : Math.max(byHeight, Math.min(byWidth, byHeight * 1.7));
      renderer.setSize(w, h);
      cam.aspect = aspect; cam.position.set(0, 0.6 * scale, dist); cam.lookAt(0, 0.6 * scale, 0);
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
