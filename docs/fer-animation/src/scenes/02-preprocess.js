import { $, $$, A, png, div, E } from "../core/dom.js";

// 2 预处理：流程 → 计算 → 增强 4 种 → 共用函数
export default {
  id: "s2",
  label: "预处理",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">喂给模型之前：标准化 + 数据增强</div>
    <div class="abs" style="left:60px;top:120px;display:flex;align-items:center;gap:14px" id="pipe2"></div>
    <div class="abs card" style="left:60px;top:220px;width:520px;padding:20px" data-a="calc">
      <div class="lbl">举例：一个像素值 128</div>
      <div class="num" style="font-size:30px;margin-top:10px">128 ÷ 255 = <span class="clay">0.502</span></div>
      <div class="num" style="font-size:30px;margin-top:6px">(0.502 − 0.5077) ÷ 0.2550 = <span class="clay">−0.023</span></div>
      <div class="lbl" style="margin-top:10px">均值 0.5077、标准差 0.2550 只用 Training 统计</div>
    </div>
    <div class="abs" style="left:640px;top:200px" data-a="auglbl"><span class="tag">只在训练时</span> <span class="lbl">每次随机变一下，等于凭空多出很多数据</span></div>
    <div class="abs" style="left:640px;top:250px;display:flex;gap:22px" id="aug2"></div>
    <div class="abs" style="left:60px;top:520px;width:1160px" id="shared2">
      <div class="lbl" style="margin-bottom:12px">同一个函数，三处共用 → 不可能不一致</div>
      <div style="display:flex;align-items:center;gap:18px">
        <span class="chip" data-a="u1">训练 train.py</span><span class="chip" data-a="u2">评估 evaluate.py</span><span class="chip" data-a="u3">前端 app.py</span>
        <svg width="60" height="24" data-a="arr"><path d="M2 12h50m-10-8 10 8-10 8" stroke="var(--ink2)" stroke-width="2" fill="none"/></svg>
        <span class="chip hot mono" data-a="fn">dataset.build_transform()</span>
      </div>
    </div>
  `,
  notes: {
    "t": "预处理与数据增强",
    "plain": [
      "<b>标准化（<code>Normalize</code>）</b>：像素先除以 255 变到 0~1，再减均值 <code>0.5077</code>、除标准差 <code>0.2550</code>，把所有图片的明暗统一到 0 附近，模型学起来更稳。这两个数<b>只用 Training 算</b>，绝不偷看测试集。",
      "<b>数据增强（<code>augment</code>）</b>：只在平时刷题（训练）时，把同一张脸随机<b>左右镜像（HorizontalFlip）、歪头 ±10°（旋转）、挪动 10%（平移）、缩放 0.9–1.1</b>，逼模型学真正表情而不是背原图。",
      "<b>三处共用防走样</b>：<code>train.py</code>、<code>evaluate.py</code>、<code>app.py</code> 全部调用 <code>dataset.py</code> 里的同一个 <code>build_transform()</code>，确保训练、测试、网页看到的图标准完全一致。"
    ],
    "say": "预处理都在 dataset.py 的 build_transform 里：先除以 255，再用仅从训练集统计的均值 0.5077 和标准差 0.2550 做标准化。只在训练时开启随机水平翻转和仿射变换（旋转、平移、缩放）做数据增强；验证、测试和网页演示全部只做标准化，三处共用同一函数，不会出现不一致。",
    "code": [
      "<code>dataset.py</code> 第 32–33 行：<code>MEAN = 0.5077</code>、<code>STD = 0.2550</code>",
      "<code>dataset.py</code> 第 92–105 行（<code>build_transform</code>）：<code>train=True</code> 加增强，<code>False</code> 只标准化",
      "<code>dataset.py</code> 第 123–127 行（<code>preprocess_face</code>）：供前端 <code>app.py</code> 复用"
    ],
    "qa": [
      [
        "为什么要做标准化？",
        "把输入数值统一缩放到 0 附近、方差接近 1，梯度更新不会忽大忽小，训练更稳、收敛更快。"
      ],
      [
        "为什么验证和测试不做增强？",
        "验证和测试要客观反映真实水平，随机旋转缩放会让结果每次不一样；而后面用的“翻转 TTA”是确定的原图+镜像取平均，不是随机增强。"
      ]
    ]
  },
  build(sc) {
    const p=$("#pipe2"); p.innerHTML="";
     ["像素 0–255","÷ 255","− 均值 ÷ 标准差","张量 1×48×48"].forEach((s,i)=>{ if(i) div("→","font-size:22px;color:var(--ink3)",p).className="ar"; div(s,"",p).className="chip"+(i===3?" hot":""); });
     const a=$("#aug2"); a.innerHTML="";
     const augs=[["原图",""],["水平翻转","scaleX(-1)"],["旋转 ±10°","rotate(10deg)"],["平移 10%","translate(12px,-10px)"],["缩放 0.9–1.1","scale(.9)"]];
     augs.forEach(([n])=>div(`<div style="width:100px;height:100px;overflow:hidden;border-radius:8px;background:#ddd"><img class="px im" src="${png(3)}" style="width:100px;height:100px;display:block;border-radius:0" alt=""></div><div class="lbl" style="text-align:center;margin-top:8px;font-size:14px">${n}</div>`,"",a));
     const ims=$$(".im",a), tl=gsap.timeline({defaults:E});
     tl.from(p.children,{opacity:0,x:-20,stagger:.15,duration:.4})
       .from(A(sc,"calc"),{opacity:0,y:20,duration:.6})
       .from(A(sc,"auglbl"),{opacity:0,duration:.4})
       .from(a.children,{opacity:0,y:30,stagger:.12,duration:.5})
       .to(ims[1],{scaleX:-1,duration:.6,ease:"power2.inOut"})
       .to(ims[2],{rotation:10,duration:.6,ease:"power2.inOut"},"<.15")
       .to(ims[3],{x:10,y:-10,duration:.6,ease:"power2.inOut"},"<.15")
       .to(ims[4],{scale:.9,duration:.6,ease:"power2.inOut"},"<.15")
       .from(["u1","u2","u3"].map(k=>A(sc,k)),{opacity:0,y:16,stagger:.15,duration:.4},"+=.3")
       .from(A(sc,"arr"),{opacity:0,x:-10,duration:.3})
       .from(A(sc,"fn"),{opacity:0,scale:.8,duration:.5,ease:"back.out(2)"});
     return tl;
  },
};
