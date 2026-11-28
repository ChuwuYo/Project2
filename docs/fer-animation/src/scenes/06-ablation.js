import { $, A, S, T, E } from "../core/dom.js";
import { ABL } from "../data/metrics.js";

// 6 消融：均值 ± 标准差
export default {
  id: "s6",
  label: "消融实验",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">每次只加一个改动，每组 3 个随机种子</div>
    <svg class="abs" style="left:40px;top:110px" width="1200" height="480" id="chart6"></svg>
    <div class="abs card" style="left:60px;top:600px;width:1160px;padding:16px 20px" data-a="concl">
      <span style="font-weight:600">结论：</span><span class="lbl">数据增强提升约 <b class="clay">10.3 个百分点</b>，最显著；标签平滑、Dropout 擦除、类别加权均在波动内；ResNet18 高于 VGG 约 1 个点；超参数试了 lr/batch，batch 64 单次达 71.05%。</span>
    </div>
  `,
  notes: {
    "t": "消融实验与结论",
    "plain": [
      "<b>消融实验（Ablation）</b>：就是<b>控制变量做对照</b>——每次只加一个改动，其余条件完全不变，才能分清到底是谁在提分（配置定义在 <code>train.py</code> 的 <code>EXPERIMENTS</code>）。",
      "<b>为什么要跑 3 个随机种子（均值 ± 标准差）</b>：神经网络初始化和打乱数据有随机性，只跑 1 次可能是撞大运；每组跑 3 个种子看波动范围（标准差约 0.3–0.6%），结论才站得住脚。",
      "<b>六组对比结果</b>：<code>A0→A1</code> 加数据增强涨了 <b>+10.3 个百分点</b>（最硬核的提升）；<code>A2</code> 标签平滑、<code>A3</code> Dropout+随机擦除、<code>A5</code> 类别加权损失单模型都在波动内；<code>V1</code> 换成没有残差连接的 8 层 VGG 低了约 1 个点。另外在单种子上试了学习率和 batch size，<code>batch 64</code> 最高（71.05%）。"
    ],
    "say": "我们做了严格控制变量的消融实验，且每组跑 3 个随机种子报告均值和标准差。数据增强（A1）把验证集从 58.87% 提到 69.19%，远超随机波动；标签平滑（A2）、Dropout+随机擦除（A3）、类别加权（A5）单看都在波动范围内；换成 VGG（V1）比 ResNet18 低约 1 个百分点。超参数也对比了学习率和 batch size，batch 64 单次最好（71.05%）。",
    "code": [
      "<code>train.py</code> 第 36–41 行（<code>EXPERIMENTS</code>）：消融配置定义",
      "<code>reports/model_selection.md</code>：A0–A3、A5、V1 与超参数的完整 PublicTest 记录"
    ],
    "qa": [
      [
        "为什么要跑 3 个随机种子？",
        "单次训练有随机波动，项目要求任何改进结论都要有充分实验支撑，不能靠一次运气宣称某方法更好。"
      ],
      [
        "为什么还试了 VGG 和类别加权？",
        "对应 PPT 的改进方向：验证残差结构（ResNet18 vs VGG）和处理类别不平衡（A5 加权损失）是否有效；实验证明 ResNet18 优于 VGG，A5 单模型没超过 A1 但给集成提供了多样性。"
      ]
    ]
  },
  build(sc) {
    const g=$("#chart6"); g.innerHTML="";
     const X0=90, Y0=400, H=320, ymin=50, ymax=75, py=v=>Y0-(v-ymin)/(ymax-ymin)*H, ax=S("g",{},g);
     for(let v=50; v<=70; v+=5){ S("line",{x1:X0,x2:1150,y1:py(v),y2:py(v),stroke:"#ece8de"},ax); T(ax,X0-12,py(v)+5,v+"%",{"text-anchor":"end","font-size":14}); }
     const bars=[], errs=[], labs=[];
     ABL.forEach((a,i)=>{ const x=X0+25+i*174, w=112, col=i===0?"#b9b2a2":i===1?"#d97757":i===5?"#6a8caf":"#e3b6a3";
       bars.push(S("rect",{x,y:py(a.m),width:w,height:Y0-py(a.m),rx:8,fill:col},g));
       const eg=S("g",{},g); S("line",{x1:x+w/2,x2:x+w/2,y1:py(a.m+a.s),y2:py(a.m-a.s),stroke:"#1f1e1d","stroke-width":2},eg);
       [a.m+a.s,a.m-a.s].forEach(v=>S("line",{x1:x+w/2-10,x2:x+w/2+10,y1:py(v),y2:py(v),stroke:"#1f1e1d","stroke-width":2},eg)); errs.push(eg);
       const lg=S("g",{},g); T(lg,x+w/2,py(a.m+a.s)-12,`${a.m.toFixed(2)}±${a.s.toFixed(2)}`,{"text-anchor":"middle","font-size":14,fill:"#1f1e1d","font-weight":600});
       T(lg,x+w/2,Y0+28,a.k,{"text-anchor":"middle","font-size":17,fill:"#1f1e1d","font-weight":600}); T(lg,x+w/2,Y0+50,a.d,{"text-anchor":"middle","font-size":14}); labs.push(lg); });
     const br=S("g",{},g); const x1=X0+25+56, x2=X0+25+174+56, yb=py(74.2);
     S("path",{d:`M${x1} ${py(60.5)} V${yb} H${x2} V${py(72.4)}`,fill:"none",stroke:"#c96442","stroke-width":2.5,"stroke-dasharray":"6 4"},br);
     T(br,(x1+x2)/2,yb-10,"+10.3 个百分点",{"text-anchor":"middle","font-size":18,fill:"#c96442","font-weight":600});
     const tl=gsap.timeline({defaults:E});
     tl.from(ax.children,{opacity:0,duration:.3,stagger:.01})
       .from(bars,{attr:{y:Y0,height:0},stagger:.35,duration:.8})
       .from(errs,{opacity:0,stagger:.2,duration:.3},"-=.6").from(labs,{opacity:0,stagger:.2,duration:.3},"<")
       .from(br,{opacity:0,y:10,duration:.6})
       .from(A(sc,"concl"),{opacity:0,y:20,duration:.6});
     return tl;
  },
};
