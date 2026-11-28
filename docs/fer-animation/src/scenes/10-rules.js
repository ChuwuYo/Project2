import { $, A, div, E } from "../core/dom.js";

// 10 数据用途规则
export default {
  id: "s10",
  label: "规则",
  html: `
    <div class="abs t-title" style="left:60px;top:40px">三份数据，三种用途，绝不混用</div>
    <div class="abs" style="left:60px;top:120px;display:flex;gap:24px" id="rule10"></div>
    <div class="abs card" style="left:60px;top:340px;width:1160px;padding:20px 24px" data-a="tests">
      <div style="font-weight:600;margin-bottom:8px">自动测试保证（pytest，9 项全部通过）</div>
      <div class="lbl" style="line-height:1.9" id="tests10"></div>
    </div>
  `,
  notes: {
    "t": "数据用途与自查",
    "plain": [
      "<b>防数据泄漏的红线</b>：平时只用 <code>Training</code> 练，所有挑模型、调超参数、定 TTA 和集成全只用 <code>PublicTest</code>，<code>PrivateTest</code> 只在全部定案后由 <code>evaluate.py</code> 跑一次。",
      "<b>怎么用代码自证清白</b>：不是靠嘴说，而是写了 <b>9 项 <code>pytest</code> 自动测试</b>（<code>tests/test_pipeline.py</code>），自动检查训练过程绝不读 PrivateTest、源码里只有极少数文件出现 PrivateTest、前端和评估误差小于 <code>1e-5</code>、且不会误覆盖已有模型。",
      "<b>一键复核脚本</b>：验收现场双击 <code>scripts/verify_test.bat</code> 就能直接复现 <b>73.50%</b>（9 模型）或 <b>71.97%</b>（最佳单模型）。"
    ],
    "say": "整个项目严格遵守三划分协议：Training 训练，PublicTest 做一切选择，PrivateTest 只做最终评估。我们用 tests/test_pipeline.py 里的 9 项自动测试做硬性校验，确保训练绝不读取 PrivateTest、前端和评估预处理完全一致，现场可以直接用 verify_test.bat 复核 73.50%。",
    "code": [
      "<code>tests/test_pipeline.py</code>：9 项自动测试（防泄漏、防覆盖、一致性检查）",
      "<code>scripts/verify_test.bat</code>：Windows 一键复核 73.50% / 71.97%",
      "<code>references/references.yaml</code>：参考项目与开源模型说明"
    ],
    "qa": [
      [
        "怎么证明没有用测试集调参？",
        "<code>train.py</code> 只读 Training 和 PublicTest，<code>tests/test_pipeline.py</code> 会自动监听并静态扫描源码，确认训练相关文件里绝不出现 PrivateTest。"
      ],
      [
        "现场怎么复现？",
        "双击 <code>scripts/verify_test.bat</code>（或运行 <code>python evaluate.py</code>）选 1 得 73.50%，选 2（<code>--single</code>）得最佳单模型 71.97%；重训用 <code>python train.py --exp A1 --seed 0 --out runs_local</code>。"
      ],
      [
        "参考来源和代码是怎么完成的？",
        "思路参考了 GitHub 上 LetheSec 的项目（ResNet 首层改 3×3、去 maxpool、数据增强等），但修正了原仓库拿 PrivateTest 选模型的问题，未使用其权重；整体方案和实验思路是我自己选定的，代码实现借助 AI 辅助编写完成；网页演示中的 YuNet（OpenCV Zoo）仅用于框人脸，不参与任何评估。"
      ]
    ]
  },
  build(sc) {
    const r=$("#rule10"); r.innerHTML="";
     [["Training","28,709","只用来训练（更新参数）","var(--sage)"],["PublicTest","3,589","选模型、选 TTA、选集成方案","var(--sky)"],["PrivateTest","3,589","方案确定后评估一次：73.50%","var(--clay)"]].forEach(([n,c,u,col])=>div(`<div style="height:8px;background:${col};border-radius:4px;margin-bottom:14px"></div><div style="font-size:22px;font-weight:600">${n}</div><div class="num" style="font-size:18px;color:var(--ink2);margin:4px 0 10px">${c} 张</div><div style="font-size:17px;line-height:1.6">${u}</div>`,"width:370px;padding:22px",r).className="card");
     $("#tests10").innerHTML="✓ 三个子集数量与标签范围 &nbsp; ✓ 均值/标准差只来自 Training &nbsp; ✓ 模型输出形状<br>✓ 前端预处理 = 评估预处理 &nbsp; ✓ 训练过程从不读取 PrivateTest，且能断点续训<br>✓ 源码里只有少数文件出现 PrivateTest &nbsp; ✓ 前端推理 = 评估推理（误差 &lt; 1e-5）<br>✓ 训练不会覆盖已有模型 &nbsp; ✓ 最终 9 模型与最佳单模型权重齐备";
     const tl=gsap.timeline({defaults:E});
     tl.from(r.children,{opacity:0,y:30,stagger:.25,duration:.6}).from(A(sc,"tests"),{opacity:0,y:20,duration:.6},"+=.2");
     return tl;
  },
};
