import { $ } from "./dom.js";

export function initMobile(player) {
  const app = $("#app"), main = $("#main"), aside = $("#notes");
  if (!app || !main || !aside) return;

  const mqCompact = matchMedia("(max-width: 900px), (max-height: 520px) and (orientation: landscape)");
  const mqLandscape = matchMedia("(orientation: landscape)");

  const backdrop = document.createElement("div");
  backdrop.id = "notesBackdrop";
  backdrop.addEventListener("click", () => player.setNotesOpen(false));
  app.appendChild(backdrop);

  const syncOrientationNotes = () => {
    const shouldCollapse = mqCompact.matches && mqLandscape.matches;
    player.setNotesOpen(!shouldCollapse);
  };
  syncOrientationNotes();
  mqCompact.addEventListener?.("change", syncOrientationNotes);
  mqLandscape.addEventListener?.("change", syncOrientationNotes);

  // 移动端舞台手势：左右滑动换章，轻点暂停/继续
  let sx = 0, sy = 0, st = 0, down = false;
  main.addEventListener("pointerdown", e => {
    if (!mqCompact.matches) return;
    down = true;
    sx = e.clientX;
    sy = e.clientY;
    st = Date.now();
  });
  main.addEventListener("pointercancel", () => { down = false; });
  main.addEventListener("pointerup", e => {
    if (!down || !mqCompact.matches) return;
    down = false;
    const dx = e.clientX - sx, dy = e.clientY - sy, dt = Date.now() - st;
    if (Math.abs(dx) >= 50 && Math.abs(dx) > Math.abs(dy) * 1.5 && dt < 700) {
      if (dx < 0) player.next();
      else player.prev();
      return;
    }
    if (Math.abs(dx) < 12 && Math.abs(dy) < 12 && dt < 350) player.togglePlay();
  });

  // 横屏抽屉：向下滑或按 Escape 合上
  let dy0 = 0, tracking = false;
  aside.addEventListener("pointerdown", e => {
    if (!(mqCompact.matches && mqLandscape.matches)) return;
    dy0 = e.clientY;
    tracking = true;
  });
  aside.addEventListener("pointercancel", () => { tracking = false; });
  aside.addEventListener("pointerup", e => {
    if (!tracking) return;
    tracking = false;
    if (e.clientY - dy0 > 80 && aside.scrollTop <= 0) player.setNotesOpen(false);
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && mqCompact.matches && mqLandscape.matches && !app.classList.contains("no-notes")) {
      player.setNotesOpen(false);
    }
  });
}
