import { $, A, S, T, E } from "../core/dom.js";

// 3 模型：逐层出现，特征图越来越小、通道越来越多；残差块
export default {
  id: "s3",
  label: "模型",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">ResNet18，为 48×48 小图改了三处</div>
    <svg class="abs" style="left:0;top:110px" width="1280" height="360" id="net3"></svg>
    <div class="abs card" style="left:60px;top:490px;width:560px;padding:18px" data-a="why">
      <div style="font-weight:600;margin-bottom:6px">为什么改？</div>
      <div class="lbl" style="line-height:1.7">原版第一层 7×7、步长 2，再接 maxpool，<b class="clay">48 一上来就缩到 12</b>，细节丢光。改成 3×3、步长 1、去掉 maxpool 后，最后还剩 6×6。</div>
    </div>
    <svg class="abs" style="left:660px;top:480px" width="560" height="200" id="res3"></svg>
  `,
  notes: {
    "t": "模型结构",
    "plain": [
      "<b>ResNet18 是什么</b>：一个 18 层的卷积神经网络（CNN），浅层找边缘线条，中层找眉眼嘴角，深层组合成表情判断。全部代码就在 <code>model.py</code>（只有 19 行）。",
      "<b>为什么要改三处</b>：原版 ResNet18 是给 224×224 彩色大图用的，一上来用大扫把（<code>7×7/步长2</code>）加暴力压缩（<code>maxpool</code>），会把 48×48 小脸瞬间压成 12×12、细节全糊。我们改成细扫的 <b>1 通道 3×3/步长 1</b>，<b>去掉 maxpool</b>，让尺寸慢慢变小：<b>48 → 24 → 12 → 6</b>。",
      "<b>残差连接（<code>F(x) + x</code>）</b>：18 层太深，信息传到后面容易衰减；残差就是修一条“直通高架桥（+x）”把输入抄近道加到输出上，反向传播时梯度也能直接传回去。",
      "<b>Dropout + FC</b>：最后全局平均池化成 512 个数字，加一层 <code>Dropout</code>（训练时随机关掉一部分神经元防偷懒），再接全连接层（<code>Linear</code>）输出 7 类分数。"
    ],
    "say": "模型定义在 model.py 里，是基于 PyTorch 的 ResNet18 从头训练，专门针对 48×48 灰度小图改了三处：第一层 conv1 改成单通道 3×3 步长 1；去掉 maxpool 保留空间分辨率，让 4 个残差阶段的特征图依次为 48、24、12、6；最后在全连接层前加了可配置的 Dropout 输出 7 类。残差连接 F(x)+x 让深层网络梯度能顺畅回传。",
    "code": [
      "<code>model.py</code> 第 7–18 行（<code>build_model</code>）：全文件就 19 行，第 15–17 行就是全部三处改造"
    ],
    "qa": [
      [
        "为什么不用预训练权重？",
        "ImageNet 预训练是 224×224 彩色自然物体，和 48×48 灰度人脸差别很大；而且改造首层后从头训练也能超过 70%，结构更干净、更好解释。"
      ],
      [
        "残差连接有什么用？",
        "反向传播时梯度可以沿着 +x 这条捷径直接往回传，避免深层网络梯度消失或退化。"
      ],
      [
        "参数量大约多少？",
        "约 1100 万（标准 ResNet18 量级）。"
      ]
    ]
  },
  build(sc) {
    const g=$("#net3"); g.innerHTML="";
     const L=[["输入","1×48×48",48,1,"#cfc9bb"],["conv1 3×3","64×48×48",48,2,"var(--clay)"],["layer1","64×48×48",48,2,"#b9b2a2"],["layer2","128×24×24",24,3,"#a69e8c"],["layer3","256×12×12",12,4,"#8f8676"],["layer4","512×6×6",6,5,"#6f675a"],["全局池化","512",3,5,"#5e5d59"],["Dropout+FC","7 类",3,2,"var(--clay)"]];
     const blocks=[]; let x=40;
     L.forEach(([n,sh,s,w,c],i)=>{ const h=Math.max(30,s*4.6), bw=26+w*10, y=170-h/2;
       const grp=S("g",{},g); S("rect",{x,y,width:bw,height:h,rx:8,fill:c},grp);
       T(grp,x+bw/2,y-14,n,{"text-anchor":"middle","font-size":16,fill:"#1f1e1d","font-weight":600});
       T(grp,x+bw/2,y+h+26,sh,{"text-anchor":"middle","font-size":14});
       if(i<L.length-1) S("path",{d:`M${x+bw+6} 170h${118-bw-14}`,stroke:"#8f8c84","stroke-width":2,"marker-end":"url(#ah)"},grp);
       blocks.push(grp); x+=150; });
     const defs=S("defs",{},g), m=S("marker",{id:"ah",markerWidth:8,markerHeight:8,refX:6,refY:4,orient:"auto"},defs); S("path",{d:"M0 0 8 4 0 8z",fill:"#8f8c84"},m);
     const dot=S("circle",{cx:40,cy:170,r:9,fill:"var(--clay)"},g);
     const out=T(g,1180,175,"",{"font-size":18,fill:"#c96442","font-weight":600});
     // 残差块示意
     const r=$("#res3"); r.innerHTML="";
     T(r,10,24,"残差块：输出 = F(x) + x",{"font-size":17,fill:"#1f1e1d","font-weight":600});
     T(r,20,112,"x",{"font-size":20,fill:"#1f1e1d"});
     S("path",{d:"M40 106h40",stroke:"#5e5d59","stroke-width":2},r);
     S("rect",{x:80,y:84,width:240,height:44,rx:8,fill:"#ece8de"},r); T(r,200,112,"卷积 → BN → ReLU → 卷积 → BN",{"text-anchor":"middle","font-size":14});
     S("path",{d:"M320 106h60",stroke:"#5e5d59","stroke-width":2},r);
     S("circle",{cx:396,cy:106,r:16,fill:"#fff",stroke:"#5e5d59","stroke-width":2},r); T(r,396,112,"+",{"text-anchor":"middle","font-size":20,fill:"#1f1e1d"});
     S("path",{d:"M412 106h50",stroke:"#5e5d59","stroke-width":2},r); T(r,470,112,"ReLU",{"font-size":15});
     const skip=S("path",{d:"M50 106 C50 170 396 170 396 122",stroke:"#d97757","stroke-width":3,fill:"none"},r);
     T(r,200,186,"捷径：梯度可以直接往回传",{"text-anchor":"middle","font-size":14,fill:"#c96442"});
     const len=skip.getTotalLength(); skip.style.strokeDasharray=len;
     const tl=gsap.timeline({defaults:E});
     tl.from(blocks,{opacity:0,y:20,stagger:.22,duration:.5})
       .to(dot,{attr:{cx:x-150+30},duration:2.2,ease:"power1.inOut"})
       .call(()=>out.textContent="→ 快乐")
       .from(A(sc,"why"),{opacity:0,y:20,duration:.6})
       .from(r.children,{opacity:0,stagger:.04,duration:.3},"<")
       .fromTo(skip,{strokeDashoffset:len},{strokeDashoffset:0,duration:1.2,ease:"power2.inOut"});
     return tl;
  },
};
