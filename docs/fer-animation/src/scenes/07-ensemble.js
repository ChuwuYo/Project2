import { $, $$, A, png, S, T, div, E } from "../core/dom.js";
import { SEL } from "../data/metrics.js";

// 7 TTA + 集成 + 方案选择
export default {
  id: "s7",
  label: "集成",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">多看一眼、多问几个模型，再取平均</div>
    <div class="abs" style="left:60px;top:120px;width:560px" id="tta7"></div>
    <div class="abs" style="left:660px;top:120px;width:560px" id="ens7"></div>
    <svg class="abs" style="left:40px;top:300px" width="760" height="300" id="sel7"></svg>
    <div class="abs card" style="left:840px;top:330px;width:380px;padding:22px;text-align:center" data-a="final">
      <div class="lbl">方案定死后，PrivateTest 只跑一次</div>
      <div class="num clay" style="font-size:70px;font-weight:600">73.50%</div>
      <div class="lbl">evaluate.py · 3589 张（最佳单模型 71.97%）</div>
    </div>
  `,
  notes: {
    "t": "集成与翻转 TTA",
    "plain": [
      "<b>翻转 TTA（Test-Time Augmentation）</b>：考试时，把同一张脸<b>正着看一眼 + 左右镜像再看一眼</b>，把两次算出的概率取平均（写在 <code>utils.py</code> 的 <code>model_probs</code>）。每个模型开不开 TTA，由它自己在模拟考 PublicTest 上决定，结果存在 <code>summary.json</code>。",
      "<b>9 模型集成（Ensemble）</b>：就是<b>9 个学霸集体投票</b>——把 A1、A2、A5 各 3 个种子共 9 个模型的预测概率取平均（写在 <code>evaluate.py</code>）。不同模型犯错的地方不一样，一平均就把错互相抵消了。",
      "<b>先定方案，再考一次</b>：在 PublicTest 上比较后，9 模型 <b>71.69%</b> 超过 6 模型 71.52% 和单模型 71.05%；定死方案后，才在高考真题 PrivateTest 上跑一次，拿到 <b>73.50%</b>。"
    ],
    "say": "推理阶段结合了水平翻转 TTA 和多模型集成：TTA 是原图与翻转图各预测一次取概率平均，每个模型是否启用由它在 PublicTest 上的表现决定；集成是把 A1、A2、A5 共 9 个模型的概率再取平均。PublicTest 上九模型最高（71.69%），确定方案后由 evaluate.py 在 PrivateTest 上只跑了一次，得到 73.50%。",
    "code": [
      "<code>utils.py</code> 第 79–84 行（<code>model_probs</code>）：水平翻转 TTA 推理，评估和前端共用",
      "<code>evaluate.py</code> 第 60–72 行：加载 9 个模型做概率平均，<b>唯一</b>读取 PrivateTest 的脚本",
      "<code>reports/PrivateTest_ensemble9/</code>：最终 9 模型报告（<code>PrivateTest/</code> 为最佳单模型）"
    ],
    "qa": [
      [
        "为什么 PrivateTest（73.50%）比 PublicTest（71.69%）还高？",
        "两个子集图片不同，本来就有自然差异；因为我们全程没在 PrivateTest 上做过任何选择，这是真实的一次性泛化结果。"
      ],
      [
        "九模型比六模型在验证集只高 0.17%，为什么换？",
        "事先定好了换方案规则：只要新集成在 PublicTest 超过 71.52% 就换，遵守事先约定不事后改标准；最终 PrivateTest 也从 73.11% 提到了 73.50%。"
      ]
    ]
  },
  build(sc) {
    const t=$("#tta7"); t.innerHTML="";
     div(`<div style="font-weight:600;margin-bottom:10px">翻转 TTA</div><div style="display:flex;align-items:center;gap:14px"><img class="px" src="${png(5)}" style="width:84px;height:84px" alt=""><span class="lbl">+</span><img class="px fl" src="${png(5)}" style="width:84px;height:84px;transform:scaleX(-1)" alt=""><span class="lbl">→ 两次概率取平均</span></div>`,"",t);
     const e=$("#ens7"); e.innerHTML="";
     div(`<div style="font-weight:600;margin-bottom:8px">9 模型集成（A1、A2、A5 各 3 个种子）</div><div class="ms" style="display:flex;gap:6px;flex-wrap:wrap"></div><div class="avg lbl" style="margin-top:8px">→ 9 份概率取平均 → 最终预测</div>`,"",e);
     ["A1_s0","A1_s1","A1_s2","A2_s0","A2_s1","A2_s2","A5_s0","A5_s1","A5_s2"].forEach(n=>div(n,"padding:6px 12px;font-size:14px",$(".ms",e)).className="chip mono");
     const g=$("#sel7"); g.innerHTML=""; T(g,20,22,"PublicTest 上比较四种方案（用于选择）",{"font-size":16,fill:"#1f1e1d","font-weight":600});
     const X0=175, W=485, vmin=69.5, vmax=72.2, pw=v=>(v-vmin)/(vmax-vmin)*W;
     const bars=SEL.map(([n,v],i)=>{ const y=50+i*56, best=i===3;
       T(g,X0-14,y+24,n,{"text-anchor":"end","font-size":16,fill:"#1f1e1d"});
       const r=S("rect",{x:X0,y,width:pw(v),height:34,rx:6,fill:best?"#d97757":"#cfc9bb"},g);
       T(g,X0+pw(v)+10,y+23,v.toFixed(2)+"%",{"font-size":16,fill:best?"#c96442":"#5e5d59","font-weight":best?700:400});
       return r; });
     T(g,X0,290,"横轴从 69.5% 开始，放大了差距",{"font-size":13,fill:"#8f8c84"});
     const tl=gsap.timeline({defaults:E});
     tl.from(t,{opacity:0,y:20,duration:.5}).from($(".fl",t),{scaleX:1,duration:.7,ease:"power2.inOut"})
       .from(e,{opacity:0,y:20,duration:.5}).from($$(".chip",e),{opacity:0,scale:.6,stagger:.1,duration:.35,ease:"back.out(2)"}).from($(".avg",e),{opacity:0,duration:.4})
       .from(g.children,{opacity:0,duration:.3,stagger:.02}).from(bars,{attr:{width:0},stagger:.2,duration:.7},"<")
       .from(A(sc,"final"),{opacity:0,scale:.85,duration:.7,ease:"back.out(1.6)"},"+=.4");
     return tl;
  },
};
