// Scripted walkthrough of the MEOL demo, injected into every page by record_walkthrough.py
// (Page.addScriptToEvaluateOnNewDocument). It drives the real app -- picks pilots, sets the start,
// runs fights, opens results -- with a synthetic cursor, a spotlight and captions. Progress across
// page loads lives in sessionStorage; console "[tour] TOUR_DONE" tells the recorder to stop.
(() => {
  if (window.__tourLoaded) return;
  window.__tourLoaded = true;
  const KEY = 'meolTourStage';
  const stage = () => sessionStorage.getItem(KEY) || 'start';
  const setStage = s => sessionStorage.setItem(KEY, s);
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const log = m => console.log('[tour] ' + m);
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];

  const CSS = `
  #tour-cap { position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%); max-width: 1180px; width: max-content;
    background: rgba(8,12,18,.92); color: #fff; font: 500 22px/1.42 "Segoe UI", system-ui, sans-serif; padding: 14px 26px 16px;
    border-radius: 14px; box-shadow: 0 12px 40px rgba(0,0,0,.45); z-index: 2147483600; opacity: 0; transition: opacity .35s; }
  #tour-cap .ch { display: block; font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: #f5b400; margin-bottom: 3px; }
  #tour-cap code { background: rgba(255,255,255,.14); padding: 1px 7px; border-radius: 6px; font-size: 20px; }
  #tour-ring { position: fixed; border: 3px solid #f5b400; border-radius: 14px; z-index: 2147483500; pointer-events: none;
    box-shadow: 0 0 0 9999px rgba(0,0,0,.42), 0 0 22px rgba(245,180,0,.55); opacity: 0;
    transition: left .7s cubic-bezier(.3,.7,.2,1), top .7s cubic-bezier(.3,.7,.2,1), width .7s cubic-bezier(.3,.7,.2,1),
                height .7s cubic-bezier(.3,.7,.2,1), opacity .4s; }
  #tour-cursor { position: fixed; left: 0; top: 0; width: 28px; height: 28px; z-index: 2147483647; pointer-events: none;
    transition: transform .85s cubic-bezier(.3,.7,.2,1), opacity .4s; filter: drop-shadow(0 3px 4px rgba(0,0,0,.5)); }
  .tour-ripple { position: fixed; width: 18px; height: 18px; margin: -9px 0 0 -9px; border-radius: 50%; border: 3px solid #f5b400;
    z-index: 2147483646; pointer-events: none; animation: tourRipple .55s ease-out forwards; }
  @keyframes tourRipple { from { transform: scale(.4); opacity: 1; } to { transform: scale(3.2); opacity: 0; } }
  #tour-slide { position: fixed; inset: 0; z-index: 2147483550; display: flex; align-items: center; justify-content: center;
    background: radial-gradient(1200px 700px at 30% 20%, #18324f 0%, #0b1220 55%, #070b12 100%); color: #fff;
    opacity: 0; transition: opacity .6s; font-family: "Segoe UI", system-ui, sans-serif; pointer-events: none; }
  #tour-slide .box { max-width: 1120px; padding: 0 64px; }
  #tour-slide h1 { font-size: 60px; margin: 0 0 16px; letter-spacing: -.01em; }
  #tour-slide h1 b { color: #5aa2ff; }
  #tour-slide .sub { font-size: 26px; color: #c8d3e0; line-height: 1.45; }
  #tour-slide .fine { margin-top: 28px; font-size: 18px; color: #8798ad; }
  #tour-slide ol { font-size: 27px; line-height: 1.6; color: #dde6f0; padding-left: 38px; }
  #tour-slide pre { font: 19px/1.5 Consolas, monospace; color: #dde6f0; background: rgba(255,255,255,.06); padding: 22px 30px;
    border-radius: 12px; margin: 0; }
  #tour-slide code { font: 24px Consolas, monospace; background: rgba(255,255,255,.1); padding: 4px 12px; border-radius: 8px; color: #fff; }
  #tour-light { position: fixed; inset: 0; z-index: 2147483560; background: rgba(5,8,12,.88); display: flex; align-items: center;
    justify-content: center; opacity: 0; transition: opacity .45s; pointer-events: none; }
  #tour-light img { width: 90vw; height: 76vh; object-fit: contain; transform: scale(.94); margin-top: -64px;
    transition: transform .6s cubic-bezier(.3,.7,.2,1); }
  #tour-light.on img { transform: scale(1); }
  #tour-badge { position: fixed; top: 12px; right: 14px; z-index: 2147483400; font: 600 12px "Segoe UI", sans-serif;
    color: #fff; background: rgba(42,120,214,.92); padding: 6px 14px; border-radius: 99px; letter-spacing: .02em; }
  `;

  const ARROW = `<svg viewBox="0 0 24 24" width="28" height="28"><path d="M4 2 L4 20 L9 15.5 L12.4 22.5 L15.3 21.2 L11.9 14.3 L18.5 14.3 Z"
    fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>`;

  function setup() {
    const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);
    const add = (id, html = '') => { const d = document.createElement('div'); d.id = id; d.innerHTML = html; document.body.appendChild(d); return d; };
    add('tour-cap'); add('tour-ring'); add('tour-slide'); add('tour-light', '<img alt="">');
    add('tour-badge', 'MEOL demo · walkthrough');
    const c = add('tour-cursor', ARROW);
    const p = JSON.parse(sessionStorage.getItem('tourCursor') || `[${innerWidth / 2},${innerHeight / 2}]`);
    c.style.transition = 'none'; c.style.transform = `translate(${p[0]}px,${p[1]}px)`;
    requestAnimationFrame(() => { c.style.transition = ''; });
  }

  // ---------- primitives ----------
  let cur = JSON.parse(sessionStorage.getItem('tourCursor') || `[${innerWidth / 2},${innerHeight / 2}]`);
  async function cursorAt(x, y, ms = 900) {
    cur = [x, y]; sessionStorage.setItem('tourCursor', JSON.stringify(cur));
    $('#tour-cursor').style.transform = `translate(${x}px,${y}px)`; await sleep(ms);
  }
  async function cursorTo(el, ms = 900, fx = 0.5, fy = 0.5) {
    const r = el.getBoundingClientRect(); await cursorAt(r.left + r.width * fx, r.top + r.height * fy, ms);
  }
  async function click(el) {
    const d = document.createElement('div'); d.className = 'tour-ripple';
    d.style.left = cur[0] + 'px'; d.style.top = cur[1] + 'px'; document.body.appendChild(d);
    setTimeout(() => d.remove(), 700); await sleep(180); el.click(); await sleep(350);
  }
  async function caption(html, ms, chapter = '') {
    const c = $('#tour-cap');
    if (c.style.opacity === '1') { c.style.opacity = 0; await sleep(260); }
    c.innerHTML = (chapter ? `<span class="ch">${chapter}</span>` : '') + html; c.style.opacity = 1; await sleep(ms);
  }
  const hideCaption = async () => { $('#tour-cap').style.opacity = 0; await sleep(300); };
  async function spot(el, pad = 10) {
    const r = el.getBoundingClientRect(), g = $('#tour-ring');
    Object.assign(g.style, { left: r.left - pad + 'px', top: r.top - pad + 'px', width: r.width + 2 * pad + 'px',
                             height: r.height + 2 * pad + 'px', opacity: 1 });
    await sleep(500);
  }
  const unspot = async () => { $('#tour-ring').style.opacity = 0; await sleep(350); };
  async function scrollToY(y, ms = 1000) {
    const y0 = window.scrollY, dy = Math.max(0, Math.min(y, document.documentElement.scrollHeight - innerHeight)) - y0, t0 = performance.now();
    await new Promise(res => { (function step(t) { const k = Math.min(1, (t - t0) / ms), e = k < .5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
      window.scrollTo(0, y0 + dy * e); k < 1 ? requestAnimationFrame(step) : res(); })(t0); });
    await sleep(150);
  }
  const scrollToEl = (el, offset = 110, ms = 1000) => scrollToY(el.getBoundingClientRect().top + scrollY - offset, ms);
  async function slide(html, ms) {     // the cursor hides while a slide is up
    const s = $('#tour-slide'); s.innerHTML = `<div class="box">${html}</div>`; s.style.opacity = 1;
    $('#tour-cursor').style.opacity = 0; await sleep(ms);
  }
  const hideSlide = async () => { $('#tour-slide').style.opacity = 0; $('#tour-cursor').style.opacity = 1; await sleep(650); };
  async function lightbox(src, html, ms, chapter = '') {
    const L = $('#tour-light'), img = $('img', L);
    await new Promise(r => { img.onload = img.onerror = r; img.src = src; });
    L.style.opacity = 1; L.classList.add('on');
    await caption(html, ms, chapter);
    L.style.opacity = 0; L.classList.remove('on'); await sleep(500);
  }
  async function animateInput(input, to, ms = 1400) {
    const from = +input.value, n = Math.max(1, Math.round(ms / 40));
    for (let i = 1; i <= n; i++) {
      input.value = Math.round(from + (to - from) * i / n);
      input.dispatchEvent(new Event('input', { bubbles: true })); await sleep(40);
    }
  }
  async function type(input, text) {
    input.focus(); input.value = '';
    for (const ch of String(text)) { input.value += ch; input.dispatchEvent(new Event('input', { bubbles: true })); await sleep(170); }
    input.blur(); await sleep(250);
  }
  async function preset(prefix) {
    const sel = $('#preset'), opt = [...sel.options].find(o => o.text.startsWith(prefix));
    await cursorTo(sel, 700, 0.3); await click(sel);
    sel.value = opt.value; sel.dispatchEvent(new Event('change', { bubbles: true })); await sleep(400);
  }
  const card = (side, key) => $(`input[name=${side}][value=${key}]`).closest('.pilot');
  const kpiText = i => $$('.kpi')[i]?.innerText.split('\n')[0] || '';

  // ---------- pages ----------
  async function indexRun1() {
    await slide(`<h1>MEOL <b>Air Combat</b> Demo</h1>
      <div class="sub">Any pilot against any pilot: hierarchical reinforcement learning for within-visual-range gun combat.</div>
      <div class="fine">Reproduction of Li et al., <i>Drones</i> 2025, 9, 384 · JSBSim F-16 · SAC tactics + WGAN curriculum + MEOL option layer</div>`, 6500);
    await slide(`<h1>What the demo does</h1><ol>
      <li>Choose who flies <b style="color:#5aa2ff">our jet</b> and the <b style="color:#ff7a5c">adversary</b>: five pilots</li>
      <li>Choose the start: range, aspect angle (AA), antenna train angle (ATA)</li>
      <li>Fly N fights in parallel, then read the results and a report folder</li></ol>`, 8000);
    await hideSlide();
    await caption('A local web page: <code>python demo/app.py</code> opens it, and every fight is simulated live in JSBSim on this machine.', 5500, 'The demo page');

    const own = $('.panel.own');
    await spot(own); await caption('Our aircraft, in blue, can be flown by any of five pilots.', 3500, '1 · Choose the pilots');
    const texts = {
      angle: '<b>Angle tactic</b>: a SAC policy rewarded for pointing at the bandit from its six (paper Eq. 38). Against the expert from 38 fixed starts: 23 wins, 12 draws, 3 losses.',
      snapshot: '<b>Snapshot tactic</b>: the aggressive one. It takes the shooting window at any aspect (Eq. 40).',
      energy: '<b>Energy tactic</b>: converts altitude and speed into position (Eq. 41). Fast, decisive wins.',
      meol: '<b>MEOL option layer</b>: the paper\'s high level (Algorithm 3). Every second it picks which tactic flies the jet.',
      bfm: '<b>Rule-based BFM expert</b>: the Shaw-doctrine pilot every policy was trained against.',
    };
    for (const k of ['angle', 'snapshot', 'energy', 'meol', 'bfm']) {
      const c = card('own', k); await cursorTo(c, 600, 0.08, 0.35); await spot(c, 6); await caption(texts[k], 5000, '1 · Choose the pilots');
    }
    await cursorTo($('input[name=own][value=meol]'), 500); await click(card('own', 'meol'));
    await spot(card('own', 'meol'), 6); await caption('For the first run, the MEOL option layer flies our jet…', 3000, '1 · Choose the pilots');
    const advB = card('adversary', 'bfm');
    await spot($('.panel.adv')); await cursorTo(advB, 800, 0.08, 0.35); await click(advB);
    await spot(advB, 6);
    await caption('…against the rule-based expert. Either seat takes any pilot: a policy in the red seat sees the fight from its own side.', 6000, '1 · Choose the pilots');
    await unspot();

    const geo = $$('.panel').find(p => p.innerText.startsWith('Start geometry'));
    await scrollToEl(geo, 90); await spot(geo);
    await caption('The start: <b>range</b>; <b>AA</b>, where the bandit\'s nose points (0° = we sit on its six, 180° = it points at us); <b>ATA</b>, where ours points.', 8000, '2 · Choose the start');
    const prev = $('#preview').closest('.panel');
    await spot(prev); await caption('The preview draws both jets from above, each with its gun cone: 150–1000 m, ±15°.', 4500, '2 · Choose the start');
    await unspot();
    await preset('Tail chase'); await caption('Presets, such as a tail chase…', 2800, '2 · Choose the start');
    await preset('Defensive (R'); await caption('…or defensive: the bandit 2.5 km behind us, nose on.', 3000, '2 · Choose the start');
    const aaS = $('input[data-for=aa_deg]'), ataS = $('input[data-for=ata_deg]');
    await cursorTo(aaS, 600, 0.95); await caption('Or any custom start with the sliders.', 400, '2 · Choose the start');
    await animateInput(aaS, 60, 1500); await cursorTo(ataS, 500, 0.9); await animateInput(ataS, 30, 1500); await sleep(700);
    await preset('Beam'); await spot(prev);
    await caption('This run starts from a <b>beam</b>: 3 km apart, each 90° off the other\'s nose. Neither side has the advantage.', 5500, '2 · Choose the start');
    await unspot();

    const runP = $$('.panel').find(p => p.innerText.startsWith('Run'));
    await scrollToEl(runP, 140); await spot(runP);
    const it = $('#iterations'), seed = $('input[name=seed]');
    await cursorTo(it, 600); await type(it, '12'); await caption('Twelve fights from the same start…', 2600, '3 · Run');
    await cursorTo(seed, 600); await type(seed, '2026'); await caption('…with a fixed seed, so this exact run can be repeated.', 3200, '3 · Run');
    await cursorTo($('.checks'), 600, 0.2);
    await caption('Fights differ because each policy <b>samples</b> from its learned action distribution, and the expert gets a fresh seed. Tacview files get gun cones and lock lines.', 7000, '3 · Run');
    await cursorTo($('#eta'), 500); await caption('Six fights fly at a time.', 2200, '3 · Run');
    await hideCaption(); await unspot();
    const btn = $('button[type=submit]'); await cursorTo(btn, 700); setStage('run1-job'); await click(btn);
  }

  async function job(next, text) {
    setStage(next);
    await cursorAt(innerWidth / 2, 220, 600);
    await caption(text, 60000, 'Flying');
  }

  async function resultsRun1() {
    await caption('Done. The results open on their own.', 2500, '4 · Results');
    const head = $('.panel');
    await spot(head);
    await caption('The matchup, start, seeds and runtime, and the run\'s folder. report.md, Open folder and Run again are on the right.', 5500, '4 · Results');
    const k = $$('.kpi');
    const say = [   // [kpi card index, caption]
      [0, () => `<b>${kpiText(0)}</b> win rate over 12 fights, with a 95% confidence interval: twelve fights is still a small sample.`],
      [2, () => `<b>${kpiText(1)}</b> lost and <b>${kpiText(2)}</b> drawn on time; median <b>time to kill ${kpiText(3)}</b>.`],
      [4, () => `How long each side held a <b>gun solution</b> per fight. A kill needs a 0.5 s hold inside the cone.`],
      [7, () => `Which tactic MEOL flew, <b>angle / snapshot / energy</b>, and how often it switched.`],
    ];
    for (const [i, text] of say) {
      if (!k[i]) continue;
      await cursorTo(k[i], 550, 0.5, 0.35); await spot(k[i], 6);
      await caption(text(), i === 0 ? 6500 : 5000, '4 · Results');
    }
    await unspot();
    const panels = $$('.panel'), sum = panels.find(p => p.innerText.startsWith('Summary'));
    await scrollToEl(sum, 80); await spot(sum);
    await caption('The summary: outcomes, fight lengths, range over time for every fight, and how offensive each fight was.', 4500, '4 · Results');
    await unspot();
    await lightbox($('img', sum).src, 'Range over time (lower left): MEOL closes into the blue gun band and finishes most fights within half a minute.', 8000, '4 · Results');
    const ov = panels.find(p => p.innerText.startsWith('Every fight'));
    await scrollToEl(ov, 80);
    await lightbox($('img', ov).src, 'Every fight in 3D: blue is MEOL, red the expert. The titles show each outcome.', 6500, '4 · Results');
    const tab = panels.find(p => p.innerText.startsWith('Iterations'));
    await scrollToEl(tab, 80); await spot(tab);
    await caption('One row per fight: outcome, length, closest approach, gun time, and the lowest altitude each jet reached.', 6000, '4 · Results');
    await unspot();
    const row = $('table tr:nth-child(2)', tab), links = $$('a', row).map(a => [a.textContent.trim(), a.href]);
    const href = name => (links.find(l => l[0] === name) || [])[1];
    await cursorTo($('img.thumb', row), 600); await click($('td', row));
    await lightbox(href('panels'), 'Each fight\'s flight log, like the paper\'s Fig. 5/6: its 3D path, AA, ATA, HCA, range, closure rate and energy over time…', 8500, '5 · One fight');
    if (href('tactics')) await lightbox(href('tactics'), '…and MEOL\'s tactic over time, with the range below: watch it switch as the fight develops.', 8000, '5 · One fight');
    const tv = $$('a', row).find(a => a.textContent.trim() === 'Tacview');
    await cursorTo(tv, 600); await spot(tv, 6);
    await caption('Plus a Tacview file per fight, with gun cones, lock lines and the outcome bookmarked on the timeline.', 5000, '5 · One fight');
    await unspot(); await hideCaption();
    await scrollToY(0, 1000);
    const rep = $$('a.btn').find(a => a.textContent.includes('report.md'));
    await cursorTo(rep, 700); setStage('run1-report'); await click(rep);
  }

  async function report() {
    await caption('<b>report.md</b>: the same results as a document, written into the run\'s folder.', 4000, '6 · The report');
    const sections = [['Setup', 'The setup: who flew, the start, the variation between fights, the weights, the seeds.'],
                      ['How it fought', 'How it fought: time to kill, gun time, offensive and defensive time, energy, altitude, MEOL\'s tactic use.'],
                      ['Every iteration', 'Every iteration, linking to its 3D plot, flight log, Tacview file and tactic timeline.']];
    for (const [h, text] of sections) {
      const el = $$('.md h2').find(e => e.textContent.trim() === h);
      if (!el) continue;
      await scrollToEl(el, 90, 1100); await spot(el, 8);
      await caption(text, 4500, '6 · The report'); await unspot();
    }
    await hideCaption(); setStage('run2'); location.href = '/';
  }

  async function indexRun2() {
    await caption('Second run: <b>policy against policy</b>, the paper\'s confrontation experiment.', 5500, '7 · Policy vs policy');
    await cursorTo(card('own', 'energy'), 800, 0.08, 0.35); await click(card('own', 'energy'));
    await cursorTo(card('adversary', 'angle'), 900, 0.08, 0.35); await click(card('adversary', 'angle'));
    await spot($('.grid2')); await caption('The energy tactic in our jet against the angle tactic in the red seat.', 5000, '7 · Policy vs policy');
    await unspot();
    const geo = $$('.panel').find(p => p.innerText.startsWith('Start geometry'));
    await scrollToEl(geo, 90); await preset('Crossing'); await spot($('#preview').closest('.panel'));
    await caption('A crossing start: the bandit 3.5 km out on our right, crossing ahead of us.', 5500, '7 · Policy vs policy');
    await unspot();
    const runP = $$('.panel').find(p => p.innerText.startsWith('Run'));
    await scrollToEl(runP, 140);
    await cursorTo($('#iterations'), 600); await type($('#iterations'), '8');
    await cursorTo($('input[name=seed]'), 600); await type($('input[name=seed]'), '2026');
    const btn = $('button[type=submit]'); await cursorTo(btn, 700); setStage('run2-job'); await click(btn);
  }

  async function resultsRun2() {
    const k = $$('.kpi');
    await spot(k[0].parentElement);
    await caption(`Energy against angle: <b>${kpiText(0)}</b> won, <b>${kpiText(1)}</b> lost and <b>${kpiText(2)}</b> drawn. A contested matchup.`, 7000, '7 · Policy vs policy');
    await unspot();
    const panels = $$('.panel'), sum = panels.find(p => p.innerText.startsWith('Summary'));
    await scrollToEl(sum, 80);
    await lightbox($('img', sum).src, 'Long, even fights: neither tactic was trained against the other, only against the expert.', 7000, '7 · Policy vs policy');
    const ov = panels.find(p => p.innerText.startsWith('Every fight'));
    await scrollToEl(ov, 80);
    await lightbox($('img', ov).src, 'Every fight in 3D: energy in blue, angle in red.', 6000, '7 · Policy vs policy');
    await hideCaption(); setStage('final'); location.href = '/';
  }

  async function final() {
    const recent = $$('.panel').find(p => p.innerText.startsWith('Recent runs'));
    await scrollToEl(recent, 120); await spot(recent);
    await caption('Every run is kept. The history links back to each results page.', 5000, '8 · Every run kept');
    await unspot(); await hideCaption();
    const run = $$('a', recent).map(a => a.getAttribute('href')).pop()?.split('/').pop() || 'run';
    await slide(`<h1>One folder per run</h1><pre>demo_runs/${run}/
├── report.md                  win rate, time to kill, a table of every fight
├── summary.png                outcomes · fight lengths · range · offence
├── overview_3d.png            every fight in 3D
├── summary.csv · summary.json · config.json
├── iter_001_win_3d.png        3D trajectory
├── iter_001_win_panels.png    flight log (paper Fig. 5/6)
├── iter_001_win_options.png   MEOL's tactic over time
├── iter_001_win.acmi          Tacview, with gun cones
└── … the same for every iteration</pre>`, 9000);
    await slide(`<h1>Run it yourself</h1><div class="sub"><code>python demo/app.py</code>&nbsp; → &nbsp;http://127.0.0.1:5000</div>
      <div class="fine" style="font-size:21px;margin-top:34px">Five pilots · either seat · any start · N fights · one report folder</div>`, 7000);
    log('TOUR_DONE');
  }

  async function main() {
    setup();
    const p = location.pathname, s = stage();
    log(`page ${p} stage ${s}`);
    try {
      if (p === '/' && s === 'start') await indexRun1();
      else if (p === '/' && s === 'run2') await indexRun2();
      else if (p === '/' && s === 'final') await final();
      else if (p.startsWith('/job/') && s === 'run1-job') await job('run1-results', 'Twelve fights are flying in parallel right now. The bar and the outcome counts update as each one lands.');
      else if (p.startsWith('/job/') && s === 'run2-job') await job('run2-results', 'Eight policy-vs-policy fights flying…');
      else if (p.startsWith('/runs/') && p.endsWith('/report') && s === 'run1-report') await report();
      else if (p.startsWith('/runs/') && s === 'run1-results') await resultsRun1();
      else if (p.startsWith('/runs/') && s === 'run2-results') await resultsRun2();
      else log(`no step for ${p} at stage ${s}`);
    } catch (e) { log('ERROR ' + e + ' ' + e.stack); }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', main); else main();
})();
