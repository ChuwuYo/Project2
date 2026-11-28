import { $, $$, A, png, div, E } from "../core/dom.js";
import { CLS } from "../data/metrics.js";

// 9 前端流程
export default {
  id: "s9",
  label: "网页演示",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">网页演示：任何照片都先变成“FER2013 的样子”</div>
    <div class="abs card" style="left:60px;top:130px;width:380px;height:300px;overflow:hidden;background:#e9e5da" data-a="cam">
      <img id="img9" class="abs" style="left:110px;top:60px;width:160px;height:160px;filter:blur(.4px);border-radius:12px" alt="">
      <div id="box9" class="abs" style="left:110px;top:60px;width:160px;height:160px;border:3px solid var(--clay);border-radius:6px"></div>
      <div class="abs lbl" style="left:14px;top:12px">摄像头 / 上传的照片</div>
    </div>
    <div class="abs" style="left:470px;top:150px;display:flex;flex-direction:column;gap:14px" id="steps9"></div>
    <div class="abs" style="left:900px;top:130px;text-align:center" data-a="crop">
      <img id="img9b" class="px" style="width:144px;height:144px" alt="">
      <div class="lbl" style="margin-top:6px">模型真正看到的 48×48</div>
    </div>
    <div class="abs" style="left:860px;top:340px;width:360px" id="prob9"></div>
    <div class="abs lbl" style="left:60px;top:470px;width:760px;line-height:1.8" data-a="note9">为什么一定要缩成 48×48 灰度？模型只在这种图上学过，直接喂高清图等于换了一种数据，准确率反而下降。所以前端和评估用的是<b class="clay">同一套预处理、同一组 9 个模型</b>。</div>
  `,
  notes: {
    "t": "网页演示",
    "plain": [
      "<b>为什么不能把高清自拍直接喂给模型</b>：模型只在 48×48 的黑白正脸图上学过，直接喂高清彩色大图等于换了考卷，准确率会暴跌。",
      "<b><code>app.py</code> 做的 4 步转换</b>：① 用 OpenCV 内置的 <b>YuNet</b> 框出人脸和双眼；② 按双眼连线角度把歪头转正（<code>align_crop</code>）；③ 裁一个 1.2 倍正方形并缩成 48×48 灰度图；④ 调用和评估完全一样的 <code>preprocess_face</code> + 9 模型 + TTA。",
      "<b>为什么实时摄像头不卡</b>：后端只在 320px 缩略图上检测，只把几百字节的人脸框坐标和概率发回浏览器，由前端 JS 在视频上叠一层 Canvas 画框，不来回传整张图片。"
    ],
    "say": "网页演示写在 app.py，支持实时摄像头和上传拍照。核心思路是把任意照片先处理成 FER2013 的样子：用 YuNet 找人脸并按双眼转正，裁成 48×48 灰度图，再走与 evaluate.py 完全一致的预处理、9 模型集成和 TTA。前端只做演示，不参与任何测试集打分。",
    "code": [
      "<code>app.py</code> 第 49–85 行（<code>detect_faces / align_crop</code>）：YuNet 检测与双眼转正裁剪",
      "<code>app.py</code> 第 100–149 行（<code>ensemble_probs / predict / detect_live</code>）：批量集成推理",
      "<code>dataset.py</code> 第 123–127 行（<code>preprocess_face</code>）：与评估共享的预处理入口"
    ],
    "qa": [
      [
        "为什么界面里“模型看到的人脸”那么糊？",
        "因为展示的就是模型真实收到的 48×48 灰度图；如果想直接吃高清图，必须换高分辨率数据集重新训练。"
      ],
      [
        "真实摄像头里准确率也能到 73.50% 吗？",
        "不一定。73.50% 是 FER2013 测试集上的成绩，真实摄像头的光线、角度和人脸分布都不同，表情明显一些会更准。"
      ]
    ]
  },
  build(sc) {
    $("#img9").src=png(3); $("#img9b").src=png(3);
     const s=$("#steps9"); s.innerHTML="";
     ["① YuNet 找到人脸","② 按双眼转正","③ 裁 1.2 倍正方形","④ 灰度、缩到 48×48","⑤ 同样的预处理 + 9 模型 + TTA"].forEach((t,i)=>div(t,"",s).className="chip"+(i===4?" hot":""));
     const p=$("#prob9"); p.innerHTML=""; div("7 类概率（示意）","margin-bottom:6px",p).className="lbl";
     [.03,.0,.02,.86,.02,.03,.04].forEach((v,i)=>div(`<span style="width:56px;font-size:14px">${CLS[i].split(" ")[0]}</span><span style="flex:1;height:10px;background:#ece8de;border-radius:5px;overflow:hidden"><i class="pb" style="display:block;height:100%;width:${v*100}%;background:${i===3?"var(--clay)":"#b9b2a2"}"></i></span>`,"display:flex;align-items:center;gap:10px;margin:5px 0",p));
     const tl=gsap.timeline({defaults:E});
     tl.from(A(sc,"cam"),{opacity:0,duration:.5})
       .fromTo("#box9",{scale:1.6,opacity:0},{scale:1,opacity:1,duration:.6,ease:"back.out(1.6)"})
       .from(s.children,{opacity:0,x:-20,stagger:.35,duration:.4})
       .from(A(sc,"crop"),{opacity:0,scale:.6,duration:.6,ease:"back.out(1.6)"},"-=.6")
       .from($$(".pb",p),{width:0,stagger:.06,duration:.6})
       .from(A(sc,"note9"),{opacity:0,y:20,duration:.6});
     return tl;
  },
};
