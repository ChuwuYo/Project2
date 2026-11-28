import { $, $$ } from "./dom.js";

export function initPlayer(scenesData) {
  const stage = $("#stage");
  stage.innerHTML = "";
  scenesData.forEach(ch => {
    const sec = document.createElement("section");
    sec.className = "scene";
    sec.id = ch.id;
    sec.setAttribute("aria-label", ch.label);
    sec.innerHTML = ch.html;
    stage.appendChild(sec);
  });

  let cur = 0, tl = null, paused = false;
  const app = $("#app"), mainEl = $("#main"), scenes = $$(".scene");
  const scrub = $("#scrub"), playBtn = $("#play"), toggleBtn = $("#toggleNotes");

  function fit() {
    const s = Math.min(mainEl.clientWidth / 1280, mainEl.clientHeight / 720);
    stage.style.transform = `translate(-50%,-50%) scale(${s})`;
  }

  function renderNotes(i) {
    const n = scenesData[i].notes;
    const plainHtml = n.plain
      ? `<h3>人话先看懂（术语 + 文件）</h3><div class="plain"><ul>${n.plain.map(p => `<li>${p}</li>`).join("")}</ul></div>`
      : "";
    $("#notes").innerHTML = `<h2>${i}. ${n.t}</h2>${plainHtml}<h3>验收怎么讲</h3><p>${n.say}</p><h3>代码在哪</h3><ul>${n.code.map(c => `<li>${c}</li>`).join("")}</ul><h3>助教可能问</h3>${n.qa.map(([q, a]) => `<details><summary>${q}</summary><p>${a}</p></details>`).join("")}`;
    $("#chapName").textContent = `${i + 1} / ${scenesData.length} · ${n.t}`;
  }

  function go(i) {
    cur = (i + scenes.length) % scenes.length;
    if (tl) { tl.revert(); tl.kill(); }
    scenes.forEach((s, k) => s.classList.toggle("on", k === cur));
    $$(".dots button").forEach((b, k) => b.classList.toggle("on", k === cur));
    renderNotes(cur);
    tl = scenesData[cur].build(scenes[cur]);
    tl.eventCallback("onUpdate", () => { scrub.value = Math.round(tl.progress() * 1000); });
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) tl.progress(1);
    paused = false;
    playBtn.textContent = "❚❚";
  }

  function setNotesOpen(open) {
    app.classList.toggle("no-notes", !open);
    toggleBtn.setAttribute("aria-expanded", String(open));
    setTimeout(fit, 0);
  }

  function toggleNotes() {
    setNotesOpen(app.classList.contains("no-notes"));
  }

  const dots = $("#dots");
  scenesData.forEach((ch, i) => {
    const b = document.createElement("button");
    b.title = ch.notes.t;
    b.setAttribute("aria-label", ch.notes.t);
    b.onclick = () => go(i);
    dots.appendChild(b);
  });

  $("#prev").onclick = () => go(cur - 1);
  $("#next").onclick = () => go(cur + 1);
  $("#replay").onclick = () => go(cur);
  playBtn.onclick = () => {
    if (!tl) return;
    if (tl.progress() === 1) { go(cur); return; }
    paused = !paused;
    tl.paused(paused);
    playBtn.textContent = paused ? "▶" : "❚❚";
  };
  scrub.oninput = () => {
    tl.pause();
    tl.progress(scrub.value / 1000);
    paused = true;
    playBtn.textContent = "▶";
  };
  toggleBtn.setAttribute("aria-expanded", String(!app.classList.contains("no-notes")));
  toggleBtn.onclick = toggleNotes;

  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowRight" || e.key === "PageDown") { go(cur + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft" || e.key === "PageUp") { go(cur - 1); e.preventDefault(); }
    else if (e.key === " ") { playBtn.click(); e.preventDefault(); }
    else if (e.key === "r" || e.key === "R") go(cur);
    else if (e.key === "n" || e.key === "N") toggleNotes();
  });

  if (typeof ResizeObserver !== "undefined") new ResizeObserver(fit).observe(mainEl);
  window.addEventListener("resize", fit);
  fit();
  go(0);

  return {
    go,
    next: () => go(cur + 1),
    prev: () => go(cur - 1),
    togglePlay: () => playBtn.click(),
    toggleNotes,
    setNotesOpen,
    fit,
  };
}
