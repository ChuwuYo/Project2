import { IMG } from "../data/images.js";

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const A = (sc, k) => $(`[data-a="${k}"]`, sc);
export const png = i => "data:image/png;base64," + IMG[i];
const NS = "http://www.w3.org/2000/svg";

export function S(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

export function T(parent, x, y, txt, attrs = {}) {
  const t = S("text", { x, y, "font-size": 16, fill: "#5e5d59", ...attrs }, parent);
  t.textContent = txt;
  return t;
}

export function div(html, style = "", parent) {
  const d = document.createElement("div");
  d.innerHTML = html;
  d.style.cssText = style;
  if (parent) parent.appendChild(d);
  return d;
}

export const E = { ease: "power3.out" };
