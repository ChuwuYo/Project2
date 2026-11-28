import { $, $$, S, T, div, E } from "../core/dom.js";
import { CLS, RECALL } from "../data/metrics.js";

// 8 召回率 + 主要混淆
export default {
  id: "s8",
  label: "结果分析",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">哪类认得好，哪类容易混</div>
    <svg class="abs" style="left:40px;top:110px" width="700" height="520" id="rec8"></svg>
    <div class="abs" style="left:780px;top:130px;width:440px" id="conf8"></div>
  `,
  notes: {
    "t": "结果分析",
    "plain": [
      "<b>召回率（Recall）是什么</b>：就是“某一类真实表情里，模型抓对了多少”。比如 879 张笑脸抓对了 802 张，快乐的召回率就是 <b>91.2%</b>。",
      "<b>混淆矩阵（Confusion Matrix）是什么</b>：就是一张“错题流向表”，看模型最爱把哪种表情误认成哪种。比如 <b>恐惧</b> 召回最低（51.1%），528 张里有 94 张被认成悲伤、68 张被认成愤怒；悲伤有 112 张被认成中性。",
      "<b>为什么会混</b>：48×48 灰度图分辨率很低，而且恐惧/悲伤/愤怒在皱眉瞪眼时本身就很接近，连人眼看都会犹豫。"
    ],
    "say": "从 PrivateTest 的分类报告和混淆矩阵看，快乐（91.2%）和惊讶（83.4%）表情特征最明显、识别最好；恐惧召回最低（51.1%），最常被误认成悲伤（94 张）和愤怒（68 张），悲伤也常被认成中性（112 张）。主要受限于 48×48 灰度图分辨率低以及这几类表情本身相似度高。",
    "code": [
      "<code>reports/PrivateTest_ensemble9/classification_report.txt</code>：各类精确率、召回率、F1",
      "<code>reports/PrivateTest_ensemble9/confusion_matrix.png</code>：混淆矩阵热力图"
    ],
    "qa": [
      [
        "召回率和总准确率有什么区别？",
        "总准确率是全部 3589 张图里一共猜对的比例（73.50%）；召回率是单看某一个表情类别里猜对的比例，能暴露少数类或难类的短板。"
      ],
      [
        "怎么改进恐惧类？",
        "我们试过 A5 类别加权损失，单模型没有超过 A1；根本瓶颈在于 48×48 像素太低且数据集标签本身有歧义，进一步提升需要更高清的数据或更强模型。"
      ]
    ]
  },
  build(sc) {
    const g=$("#rec8"); g.innerHTML=""; T(g,20,22,"PrivateTest 各类召回率（认对的比例）",{"font-size":16,fill:"#1f1e1d","font-weight":600});
     const order=RECALL.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]);
     const X0=150, W=460;
     const bars=order.map(([v,i],k)=>{ const y=50+k*62, low=i===2;
       T(g,X0-14,y+26,CLS[i].split(" ")[0],{"text-anchor":"end","font-size":17,fill:"#1f1e1d"});
       S("rect",{x:X0,y,width:W,height:36,rx:6,fill:"#f0ece2"},g);
       const r=S("rect",{x:X0,y,width:W*v/100,height:36,rx:6,fill:low?"#d97757":i===3?"#7a8f6a":"#b9b2a2"},g);
       T(g,X0+W*v/100+10,y+25,v.toFixed(1)+"%",{"font-size":16,fill:low?"#c96442":"#5e5d59","font-weight":low?700:400}); return r; });
     const c=$("#conf8"); c.innerHTML="";
     div("最常见的认错（混淆矩阵）","font-weight:600;margin-bottom:12px",c);
     [["悲伤","中性",112,594],["恐惧","悲伤",94,528],["恐惧","愤怒",68,528],["愤怒","悲伤",62,491]].forEach(([a,b,n,tot])=>div(`<span style="font-size:18px">${a} <span class="clay">→</span> ${b}</span><span class="num" style="margin-left:auto;font-size:22px">${n}<span class="lbl" style="font-size:14px"> / ${tot}</span></span>`,"display:flex;align-items:center;padding:14px 18px;margin-bottom:10px",c).className="card cf");
     div("这几种表情在 48×48 小图里本来就很像；厌恶训练样本只有 436 张，但最难的是恐惧。","margin-top:8px;line-height:1.7",c).className="lbl";
     const tl=gsap.timeline({defaults:E});
     tl.from(g.querySelectorAll("text"),{opacity:0,duration:.3,stagger:.02}).from(bars,{attr:{width:0},stagger:.12,duration:.7},"<")
       .from($$(".cf",c),{opacity:0,x:30,stagger:.2,duration:.5}).from(c.lastChild,{opacity:0,duration:.5});
     return tl;
  },
};
