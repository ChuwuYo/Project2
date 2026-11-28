import { $, $$, A, png, div, E } from "../core/dom.js";
import { CLS, TRAIN_COUNT } from "../data/metrics.js";

// 1 数据：像素串打出 → 图像逐行出现 → 划分条 → 类别分布
export default {
  id: "s1",
  label: "数据",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">一行 CSV = 一张脸</div>
    <div class="abs card mono" style="left:60px;top:110px;width:1160px;padding:14px 18px;font-size:17px;white-space:nowrap;overflow:hidden" data-a="csv">
      <span class="clay">3</span>, <span style="color:var(--sky)">Training</span>, <span id="pixtxt"></span>
    </div>
    <div class="abs lbl" style="left:60px;top:176px" data-a="csvlbl">emotion（0–6） · Usage（用途） · pixels（2304 个 0–255 的数）</div>
    <div class="abs" style="left:60px;top:230px;width:240px;height:240px" data-a="imgbox">
      <img id="img1" class="px" style="width:240px;height:240px" alt="">
      <div id="wipe1" class="abs" style="left:0;top:0;width:240px;height:240px;background:var(--bg)"></div>
    </div>
    <div class="abs" style="left:60px;top:484px" data-a="reshape"><span class="num" style="font-size:30px">2304 = 48 × 48</span><div class="lbl">reshape 成二维图像</div></div>
    <div class="abs" style="left:380px;top:230px;width:840px" id="split1">
      <div class="lbl" style="margin-bottom:10px">官方划分（不改动）</div>
    </div>
    <div class="abs" style="left:380px;top:430px;width:840px;height:230px" id="dist1"></div>
  `,
  notes: {
    "t": "数据：CSV 怎么变成图片",
    "plain": [
      "<b>CSV 还原成图（<code>reshape</code>）</b>：原始数据是一张表格，每行把 48×48 张黑白照片拉平存成 2304 个数字（0 黑 ~ 255 白）。<code>dataset.py</code> 把这串数折回二维矩阵，就变回照片了。",
      "<b>npz 缓存</b>：每次读大 CSV 要 1 分钟，<code>dataset.py</code> 第一次读完会打包存成 <code>data/fer2013.npz</code>，之后 1 秒加载。",
      "<b>三划分（官方固定）</b>：<code>Training</code>（28,709）= 平时刷题库；<code>PublicTest</code>（3,589）= 模拟考；<code>PrivateTest</code>（3,589）= 高考真题（考前封存）。",
      "<b>类别不平衡</b>：快乐有 7,215 张，厌恶只有 436 张（差 16 倍），所以不能只看总准确率，还要看每类召回率（<code>check_data.py</code> 负责体检）。"
    ],
    "say": "数据读取全部收敛在 dataset.py：逐行解析 CSV 的 emotion、Usage 和 2304 个像素并 reshape 成 48×48 灰度图，首次解析后缓存成 npz 提速。我们严格沿用官方三划分（28709 / 3589 / 3589），并用 check_data.py 做了数据检查，发现了类别极不平衡（厌恶仅 436 张）以及少量空白图和跨子集重复图。",
    "code": [
      "<code>dataset.py</code> 第 36–58 行（<code>_parse_csv</code>）：解析 CSV 并校验每行 2304 像素",
      "<code>dataset.py</code> 第 69–89 行（<code>load_all / load_split</code>）：npz 缓存与唯一取数入口",
      "<code>check_data.py</code>：数据体检脚本，报告输出在 <code>reports/data_check/</code>"
    ],
    "qa": [
      [
        "数据检查查了什么？",
        "<code>check_data.py</code> 核对了三个子集数量、标签范围 0–6、图像尺寸 48×48、均值方差，并查出 13 张几乎纯色的空白图和跨子集重复图；为了遵守官方协议，我们保留原划分不做篡改。"
      ],
      [
        "类别不平衡怎么处理？",
        "我们专门做了 A5 类别加权损失实验，并如实报告了 7 类各自的召回率；虽然厌恶样本最少，但最终召回有 72.7%，最难的其实是恐惧（51.1%）。"
      ]
    ]
  },
  build(sc) {
    $("#img1").src=png(3);
     const pix="70 80 82 72 58 58 60 63 54 58 60 48 89 115 121 119 115 110 98 91 84 84 90 99 110 126 143 153 158 171 169 172 169 165 129 110 113 107 95 79 66 62 56 57 61 52 43 41 65 61 58 57 56 69 75 70 65 56 54 105 146 154 151 151 155 155 150 147 …";
     const pt=$("#pixtxt"); pt.textContent="";
     const sp=$("#split1"); $$(".row",sp).forEach(e=>e.remove());
     const parts=[["Training",28709,"训练：更新模型参数","var(--sage)"],["PublicTest",3589,"验证：选模型、调参数","var(--sky)"],["PrivateTest",3589,"测试：只做最终评估","var(--clay)"]];
     parts.forEach(([n,c,u,col])=>div(`<div class="b" style="height:34px;width:${c/28709*640}px;background:${col};border-radius:8px"></div><div style="font-size:16px;white-space:nowrap"><b>${n}</b> <span class="num">${c.toLocaleString()}</span> <span class="lbl">· ${u}</span></div>`,"display:flex;align-items:center;gap:14px;margin-bottom:10px",sp).classList.add("row"));
     const d=$("#dist1"); d.innerHTML=""; div("Training 各类数量","position:absolute;left:0;top:-4px",d).className="lbl";
     TRAIN_COUNT.forEach((c,i)=>{ const x=i*118, h=c/7215*150;
       div("", `left:${x}px;width:70px;height:${h}px;bottom:44px;background:${i===1?"var(--clay)":"#cfc9bb"}`,d).className="bar";
       div(`<span class="num">${c}</span>`,`position:absolute;left:${x}px;width:70px;text-align:center;bottom:${h+50}px;font-size:14px`,d).className="cnt";
       div(CLS[i].split(" ")[0],`position:absolute;left:${x}px;width:70px;text-align:center;bottom:16px;font-size:14px;color:var(--ink2)`,d); });
     const o={n:0}, tl=gsap.timeline({defaults:E});
     tl.from(A(sc,"csv"),{opacity:0,y:20,duration:.5})
       .to(o,{n:pix.length,duration:2,ease:"none",onUpdate:()=>pt.textContent=pix.slice(0,o.n|0)})
       .from(A(sc,"csvlbl"),{opacity:0,duration:.4},"-=.6")
       .fromTo("#wipe1",{height:240,top:0},{height:0,top:240,duration:1.6,ease:"steps(24)"})
       .from(A(sc,"reshape"),{opacity:0,y:10,duration:.5})
       .from($$(".row .b",sp),{width:0,stagger:.25,duration:.8},"+=.2")
       .from($$(".row div:last-child",sp),{opacity:0,stagger:.25,duration:.4},"<")
       .from($$(".bar",d),{height:0,stagger:.08,duration:.7},"+=.3")
       .from($$(".cnt",d),{opacity:0,stagger:.08,duration:.3},"<.3");
     return tl;
  },
};
