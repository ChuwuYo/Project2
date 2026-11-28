import { $, A, png, div, E } from "../core/dom.js";
import { IMG } from "../data/images.js";
import { CLS } from "../data/metrics.js";

// 0 封面：7 张真实样例脸依次出现，准确率计数
export default {
  id: "s0",
  label: "封面",
  html: `
    <div class="abs" style="left:0;right:0;top:120px;text-align:center">
      <div class="t-hero" data-a="title">让电脑看懂 7 种表情</div>
      <div class="t-sub" style="margin-top:14px" data-a="sub">FER2013 · 48×48 灰度人脸 · PyTorch 从头训练 ResNet18 · 9 模型集成</div>
    </div>
    <div class="abs" id="faces0" style="left:0;right:0;top:300px;display:flex;justify-content:center;gap:22px"></div>
    <div class="abs" style="left:0;right:0;top:500px;text-align:center" data-a="score">
      <div class="lbl">PrivateTest 准确率（3589 张，只评估一次）</div>
      <div class="num clay" style="font-size:92px;font-weight:600"><span id="score0">0.00</span>%</div>
    </div>
  `,
  notes: {
    "t": "项目概览",
    "plain": [
      "<b>任务定位</b>：这不是刷脸认“你是谁”（人脸识别），而是看图猜“你现在什么心情”（<b>表情分类</b>，共 7 类）。",
      "<b>从头训练 ResNet18</b>：不借用别人现成的大模型权重，而是用 PyTorch 把一个 18 层神经网络从零开始教到会认表情（模型写在 <code>model.py</code>）。",
      "<b>9 模型集成 = 9 个学霸投票</b>：把 9 个独立训练的模型给出的概率取平均（配置写在 <code>runs/final_runs.txt</code>），最终在高考真题 <b>PrivateTest</b> 上考出 <b>73.50%</b>。"
    ],
    "say": "我做的是 7 类人脸表情识别：输入一张 48×48 灰度人脸，用 PyTorch 从头训练改造后的 ResNet18。所有调参和选方案只看验证集 PublicTest，最后用 9 个模型（A1、A2、A5 各 3 个种子）做概率平均集成，在测试集 PrivateTest 上只评估一次，准确率 73.50%（最佳单模型 71.97%）。",
    "code": [
      "<code>README.md</code>：项目总览、实验汇总表与复现命令",
      "<code>runs/final_runs.txt</code>：最终 9 模型列表（<code>evaluate.py</code> 与 <code>app.py</code> 默认读取）",
      "<code>runs/best_single.txt</code>：最佳单模型 <code>hp_bs64/A1_s0</code>（71.97%）"
    ],
    "qa": [
      [
        "这是人脸识别吗？",
        "不是。人脸识别是认“这个人是谁”，我们做的是“人脸表情识别（FER）”，属于 7 分类图像分类任务。"
      ],
      [
        "73.50% 怎么来的？",
        "所有方案都在 PublicTest 上定死后，由 <code>evaluate.py</code> 在 PrivateTest（3589 张）上只跑了一次得出，没有拿测试集调过参。"
      ]
    ]
  },
  build(sc) {
    const box=$("#faces0"); box.innerHTML="";
     IMG.forEach((_,i)=>div(`<img class="px" src="${png(i)}" style="width:120px;height:120px" alt=""><div class="lbl" style="text-align:center;margin-top:6px">${CLS[i].split(" ")[0]}</div>`,"",box));
     const n={v:0}, tl=gsap.timeline({defaults:E});
     tl.from(A(sc,"title"),{y:30,opacity:0,duration:.8})
       .from(A(sc,"sub"),{y:20,opacity:0,duration:.6},"-=.4")
       .from(box.children,{y:60,opacity:0,scale:.6,stagger:.09,duration:.7,ease:"back.out(1.6)"},"-=.2")
       .from(A(sc,"score"),{opacity:0,duration:.4})
       .to(n,{v:73.50,duration:1.6,ease:"power2.out",onUpdate:()=>$("#score0").textContent=n.v.toFixed(2)},"<");
     return tl;
  },
};
