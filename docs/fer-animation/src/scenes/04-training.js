import { $, $$, A, S, T, div, E } from "../core/dom.js";
import { CLS } from "../data/metrics.js";

// 4 训练循环
export default {
  id: "s4",
  label: "训练",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">训练 = 不停地猜、算错多少、往回改</div>
    <svg class="abs" style="left:40px;top:100px" width="620" height="560" id="loop4"></svg>
    <div class="abs card" style="left:700px;top:110px;width:520px;padding:20px" data-a="ce">
      <div style="font-weight:600">损失函数：交叉熵 CrossEntropy</div>
      <div class="lbl" style="margin:6px 0 12px">softmax 把 7 个分数变成概率，loss = −ln(正确类的概率)</div>
      <div id="sm4"></div>
      <div class="num" style="font-size:24px;margin-top:12px">loss = −ln(<span class="clay">0.70</span>) = <span class="clay">0.36</span> <span class="lbl">（示意）</span></div>
      <div class="lbl" style="margin-top:4px">猜得越准，正确类概率越接近 1，loss 越接近 0</div>
    </div>
    <div class="abs" style="left:700px;top:440px;width:520px" id="cfg4"></div>
  `,
  notes: {
    "t": "训练过程与损失函数",
    "plain": [
      "<b>batch 128 与 epoch（一轮）</b>：每次取 128 张图算一次错、改一次参数；把 28,709 张练习题全刷完一遍叫 <b>1 个 epoch</b>（核心循环在 <code>train.py</code>）。",
      "<b>softmax + 交叉熵（<code>CrossEntropyLoss</code>）</b>：模型先输出 7 个原始分数（logits），<code>softmax</code> 把它们变成百分比概率；<b>交叉熵就是扣分规则</b>——正确类别的概率越低扣分越狠，猜对越接近 1，loss 越接近 0。",
      "<b>反向传播 + SGD</b>：算完扣分后往回找每个参数该怎么调，用 <b>SGD</b>（带 Nesterov 动量 0.9 和权重衰减 weight decay 5e-4）更新参数。",
      "<b>学习率衰减 + 早停 + 混合精度</b>：学习率（lr 0.01）是改参数的步子大小；模拟考（PublicTest）连续 5 轮没涨就把步子减半（<code>ReduceLROnPlateau</code>），连续 20 轮没涨就提前交卷（<b>早停</b>）；GPU 上开混合精度（AMP）加速训练。"
    ],
    "say": "训练全在 train.py 里完成：每批 128 张图，前向输出 7 类分数，用交叉熵损失 CrossEntropyLoss 算误差，反向传播后用 SGD（lr 0.01、Nesterov 动量 0.9、weight decay 5e-4）更新参数。每轮都在 PublicTest 上验证，连续 5 轮不升就把学习率减半，连续 20 轮不升早停，最多 100 轮，并把每轮指标记录在 log.csv 里。",
    "code": [
      "<code>train.py</code> 第 60–92 行：<code>train_one_epoch</code>（单轮训练）与 <code>validate</code>（验证）",
      "<code>train.py</code> 第 122–128 行：损失函数、SGD 优化器、学习率调度器、混合精度",
      "<code>train.py</code> 第 145–169 行：训练循环、早停、保存 <code>best.pt</code> 与断点续训 <code>last.pt</code>"
    ],
    "qa": [
      [
        "交叉熵为什么适合分类？",
        "它直接衡量预测概率分布和真实标签的差距，重点惩罚“正确类概率给低了”的情况，和提高分类准确率方向完全一致。"
      ],
      [
        "标签平滑（Label Smoothing）是什么？",
        "把正确类的目标从绝对的 1 改成 0.9，其余 6 类分一点小数，防止模型过度自信（对应 A2 实验）。"
      ],
      [
        "best.pt 按什么标准保存？",
        "只按 PublicTest（验证集）准确率最高的那一轮保存，绝不看 PrivateTest。"
      ]
    ]
  },
  build(sc) {
    const g=$("#loop4"); g.innerHTML="";
     const cx=300, cy=270, R=200;
     const nodes=["取一批 128 张图","前向：得到 7 个分数","交叉熵算 loss","反向传播求梯度","SGD 更新参数"];
     const ring=S("circle",{cx,cy,r:R,fill:"none",stroke:"#e6e2d8","stroke-width":3},g);
     const dot=S("circle",{cx:cx,cy:cy-R,r:10,fill:"#c96442"},g);
     const ns=nodes.map((n,i)=>{ const a=-Math.PI/2+i*2*Math.PI/5, x=cx+R*Math.cos(a), y=cy+R*Math.sin(a);
       const grp=S("g",{},g); const w=n.length*17+28;
       S("rect",{x:x-w/2,y:y-22,width:w,height:44,rx:22,fill:i===2?"#d97757":"#fff",stroke:i===2?"#d97757":"#e6e2d8"},grp);
       T(grp,x,y+6,n,{"text-anchor":"middle","font-size":16,fill:i===2?"#fff":"#1f1e1d"}); return grp; });
     T(g,cx,cy-6,"一轮 = 把 28709 张",{"text-anchor":"middle","font-size":17,fill:"#1f1e1d"});
     T(g,cx,cy+20,"全部过一遍",{"text-anchor":"middle","font-size":17,fill:"#1f1e1d"});
     // softmax 示意条
     const sm=$("#sm4"); sm.innerHTML="";
     const pr=[.06,.01,.05,.70,.05,.06,.07];
     pr.forEach((p,i)=>div(`<span style="width:110px;font-size:14px">${CLS[i].split(" ")[0]}</span><span style="flex:1;height:10px;background:#ece8de;border-radius:5px;overflow:hidden"><i class="pb" style="display:block;height:100%;width:${p*100}%;background:${i===3?"var(--clay)":"#b9b2a2"}"></i></span><span class="num" style="width:44px;text-align:right;font-size:14px">${p.toFixed(2)}</span>`,"display:flex;align-items:center;gap:10px;margin:5px 0",sm));
     const c=$("#cfg4"); c.innerHTML="";
     ["学习率 0.01，SGD + Nesterov 动量 0.9","5 轮验证准确率不升 → 学习率减半","20 轮不升 → 早停，最多 100 轮","weight decay 5e-4 · batch 128 · GPU 混合精度"].forEach(s=>div(s,"margin-bottom:10px;font-size:16px;padding-left:14px;border-left:3px solid var(--clay)",c));
     const tl=gsap.timeline({defaults:E}), o={a:0};
     tl.from(ring,{opacity:0,duration:.4}).from(ns,{opacity:0,scale:.7,transformOrigin:"center",stagger:.15,duration:.4,ease:"back.out(2)"})
       .from(A(sc,"ce"),{opacity:0,x:30,duration:.6},"<.3")
       .from($$(".pb",sm),{width:0,stagger:.06,duration:.6},"<.3")
       .to(o,{a:Math.PI*4,duration:5,ease:"none",onUpdate:()=>{gsap.set(dot,{attr:{cx:cx+R*Math.sin(o.a),cy:cy-R*Math.cos(o.a)}})}},"<")
       .from(c.children,{opacity:0,x:20,stagger:.15,duration:.4},"<1");
     return tl;
  },
};
