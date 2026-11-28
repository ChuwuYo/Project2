import { $, S, T, E } from "../core/dom.js";
import { CURVES } from "../data/metrics.js";

// 5 过拟合：真实曲线
export default {
  id: "s5",
  label: "过拟合",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">没有增强：背答案；有增强：学规律</div>
    <svg class="abs" style="left:40px;top:110px" width="1200" height="520" id="chart5"></svg>
    <div class="abs lbl" style="left:60px;top:650px">真实训练记录：runs/A0_s0/log.csv、runs/A1_s0/log.csv（Colab T4）</div>
  `,
  notes: {
    "t": "过拟合与数据增强",
    "plain": [
      "<b>什么是过拟合（Overfitting）</b>：就是<b>“死记硬背练习题”</b>——练习题（Training）考 99.8%，一换模拟考（PublicTest）只考 58%，中间差了 40 个百分点。",
      "<b>对比曲线怎么看</b>：灰线是无增强的 <code>A0_s0</code>，第 17 轮后模拟考就卡死不动了；橙线是加了数据增强的 <code>A1_s0</code>，虽然练习题变难了（准确率爬得慢），但模拟考一路涨到 <b>68–69%</b>，真正学到了表情规律。",
      "<b>数据来源</b>：图里的线不是示意图，是直接从真实训练日志 <code>runs/A0_s0/log.csv</code> 和 <code>runs/A1_s0/log.csv</code> 画出来的。"
    ],
    "say": "这张图是真实训练日志画出的曲线。不加数据增强的 A0，训练集冲到 99.8%，但验证集停在 58% 左右，差 40 个百分点，属于严重过拟合；加上数据增强（A1）后，训练题变难了，但验证集涨到了 68–69%，训练和验证的差距大幅缩小。",
    "code": [
      "<code>runs/A0_s0/log.csv</code>、<code>runs/A1_s0/log.csv</code>：每轮真实训练与验证记录",
      "<code>utils.py</code> 第 134–149 行（<code>plot_curves</code>）：自动把 log.csv 画成 <code>curves.png</code>"
    ],
    "qa": [
      [
        "怎么判断模型过拟合了？",
        "看曲线：训练集准确率很高（接近 100%），但验证集准确率远远低于训练集且不再上升甚至开始下降。"
      ],
      [
        "除了数据增强，还用了哪些防过拟合手段？",
        "权重衰减 weight decay 5e-4、早停（patience 20），并在 A3 实验中对比了 Dropout 0.4 和 RandomErasing。"
      ],
      [
        "为什么 A1 的训练集准确率反而比 A0 低？",
        "因为 A1 的训练准确率是在随机翻转、旋转、缩放后的“变难版图片”上算的，这正好说明模型没法死背原图了。"
      ]
    ]
  },
  build(sc) {
    const g=$("#chart5"); g.innerHTML="";
     const X0=70, Y0=460, W=680, H=420, xm=70, ymin=20, ymax=100;
     const px=e=>X0+(e/xm)*W, py=v=>Y0-(v-ymin)/(ymax-ymin)*H, ax=S("g",{},g);
     for(let v=20; v<=100; v+=20){ S("line",{x1:X0,x2:X0+W,y1:py(v),y2:py(v),stroke:"#ece8de"},ax); T(ax,X0-12,py(v)+5,v+"%",{"text-anchor":"end","font-size":14}); }
     for(let e=0;e<=70;e+=10) T(ax,px(e),Y0+24,e,{"text-anchor":"middle","font-size":14});
     T(ax,X0+W/2,Y0+50,"训练轮数 epoch",{"text-anchor":"middle","font-size":14});
     const lines=[["A0_s0",0,"#8f8c84","A0 训练集","6 3"],["A0_s0",1,"#5e5d59","A0 验证集",""],["A1_s0",0,"#e8a58c","A1 训练集","6 3"],["A1_s0",1,"#c96442","A1 验证集",""]];
     const paths=lines.map(([r,j,col,lab,dash])=>{ const pts=CURVES[r].map((v,i)=>`${px(i+1)},${py(v[j])}`).join(" ");
       const p=S("polyline",{points:pts,fill:"none",stroke:col,"stroke-width":j?3.5:2.5},g);
       const last=CURVES[r][CURVES[r].length-1]; const len=p.getTotalLength(); p.style.strokeDasharray=len; p.style.strokeDashoffset=len;
       const l=T(g,px(CURVES[r].length)+8,py(last[j])+(r==="A1_s0"&&j===0?-6:5),`${lab} ${last[j]}%`,{"font-size":14,fill:col,"font-weight":600});
       return {p,l,len,dash}; });
     // 右侧说明
     const k=S("g",{},g);
     S("rect",{x:900,y:70,width:290,height:150,rx:14,fill:"#fff",stroke:"#e6e2d8"},k);
     T(k,920,104,"A0 无增强",{"font-size":18,fill:"#1f1e1d","font-weight":600});
     T(k,920,136,"训练 99.8% vs 验证 58%",{"font-size":16});
     T(k,920,166,"差 40 个点 → 严重过拟合",{"font-size":16,fill:"#c96442"});
     T(k,920,196,"第 17 轮后验证集不再上升",{"font-size":14});
     const k2=S("g",{},g);
     S("rect",{x:900,y:250,width:290,height:150,rx:14,fill:"#fff",stroke:"#e6e2d8"},k2);
     T(k2,920,284,"A1 加数据增强",{"font-size":18,fill:"#1f1e1d","font-weight":600});
     T(k2,920,316,"验证集到 68–69%",{"font-size":16});
     T(k2,920,346,"训练/验证差距明显缩小",{"font-size":16,fill:"#c96442"});
     T(k2,920,376,"训练准确率在变换后的图上算",{"font-size":14});
     const tl=gsap.timeline({defaults:E});
     tl.from(ax.children,{opacity:0,duration:.4,stagger:.01})
       .to(paths[0].p,{strokeDashoffset:0,duration:1.4,ease:"power1.inOut"}).to(paths[1].p,{strokeDashoffset:0,duration:1.4,ease:"power1.inOut"},"<")
            .from([paths[0].l,paths[1].l],{opacity:0,duration:.4})
       .from(k,{opacity:0,x:30,duration:.6})
       .to(paths[2].p,{strokeDashoffset:0,duration:2,ease:"power1.inOut"},"+=.4").to(paths[3].p,{strokeDashoffset:0,duration:2,ease:"power1.inOut"},"<")
            .from([paths[2].l,paths[3].l],{opacity:0,duration:.4})
       .from(k2,{opacity:0,x:30,duration:.6});
     return tl;
  },
};
