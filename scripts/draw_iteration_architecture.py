#!/usr/bin/env python
"""Reproducible vector architecture figure; render PNG with CairoSVG."""
from pathlib import Path
from html import escape
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures'
W,H=2100,1190
items=[]
ORANGE='#ffe1b4'; BLUE='#d7eaff'; GRAY='#edf0f4'; INK='#18293e'; GREEN='#e2f2e9'

def rect(x,y,w,h,fill=GRAY,stroke='#8193a8',r=14):
    items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
def text(x,y,lines,size=24,color=INK,anchor='middle',bold=False):
    if isinstance(lines,str):lines=[lines]
    parts=''.join(f'<tspan x="{x}" dy="{0 if i==0 else size*1.42}">{escape(line)}</tspan>' for i,line in enumerate(lines))
    items.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}" font-weight="{700 if bold else 400}">{parts}</text>')
def arrow(points,color='#48627e',dashed=False):
    d='M '+' L '.join(f'{x},{y}' for x,y in points)
    items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#arrow)"'+(' stroke-dasharray="9 6"' if dashed else '')+'/>')
def box(x,y,w,h,lines,fill=GRAY,size=23):
    rect(x,y,w,h,fill);n=1 if isinstance(lines,str) else len(lines)
    text(x+w/2,y+h/2-(n-1)*size*.71+size*.34,lines,size)

rect(0,0,W,H,'#ffffff','#ffffff',0)
text(W/2,55,'LTE-WAM：DINO 条件 + 迭代感知的共享能量模型',36,bold=True)
text(W/2,99,'新版代码 / 待训练验证    ·    保留 Qwen 语言主干，移除 Qwen ViT、visual merger 与 DeepStack',22,color='#52657c')
# Current condition
rect(35,145,505,440,'#f7fbff')
text(287,185,'条件 C：先生成 token，不提前过语言主干',24,bold=True)
box(55,215,125,75,'指令文本',GRAY,22)
box(220,215,295,75,['Qwen text embedding','保留、可训练'],ORANGE,22);arrow([(180,252),(220,252)])
box(55,320,125,85,['当前 RGB','图像'],GRAY,22)
box(220,312,295,44,'DINOv3 ViT-S/16（冻结）',BLUE,20)
box(220,366,295,44,'current projector（训练）',ORANGE,20)
arrow([(180,340),(220,340)])
arrow([(367,356),(367,366)])
box(55,435,125,80,['机器人状态','sin/cos 32维'],GRAY,21)
box(220,435,295,80,['state projector（训练）','→ 状态 token'],ORANGE,22);arrow([(180,475),(220,475)])
text(287,553,'C = [T_text, T_current-DINO, T_state]',22,bold=True)
# Iteration
rect(585,145,835,220,'#fcf9f3')
text(1002,185,'新增：迭代条件（不新增序列位置）',25,bold=True)
box(605,215,180,105,['s = k / K','固定连续 sin/cos PE','256 维，scale=1000'],GRAY,19)
box(845,210,245,60,'Action MLP → e_A(s)',ORANGE,21)
box(1135,285,245,60,'Future MLP → e_Z(s)',ORANGE,21)
arrow([(785,240),(845,240)])
arrow([(785,295),(810,295),(810,315),(1135,315)])
# Two independent MLP nodes are explicitly labelled above.
# candidates
rect(35,625,505,190,'#f6faf9')
text(287,661,'候选未来 Z（连续特征，内层修改）',24,bold=True)
text(287,700,['3 帧 × 1 相机 × 196 patch × 384维','标准化 DINO 特征 → future projector','+ 帧 / 空间 / 相机 / 类型编码'],21)
rect(35,850,505,145,'#f6faf9')
text(287,889,'候选动作 A（连续动作，内层修改）',24,bold=True)
text(287,929,['16 步 × 12维 → action projector','+ 动作位置 / 类型编码'],21)
# sequence
rect(640,405,370,395,'#f4f7fc')
text(825,446,'X_k = [C, T_Z, T_A, e_E]',24,bold=True)
box(675,470,300,55,'C：文本 + DINO + 状态',GRAY,21)
box(675,550,300,65,'T_Z = Z 投影与编码 + e_Z(s)',ORANGE,20)
box(675,640,300,65,'T_A = A 投影与编码 + e_A(s)',ORANGE,20)
box(675,730,300,50,'e_E：可训练能量 token',ORANGE,21)
arrow([(540,550),(585,550),(585,497),(675,497)])
arrow([(540,720),(610,720),(610,582),(675,582)])
arrow([(540,922),(615,922),(615,672),(675,672)])
arrow([(967,270),(1030,270),(1030,630),(825,630),(825,640)])
arrow([(1257,345),(1257,385),(600,385),(600,535),(825,535),(825,550)])
# transformer + head
box(1080,455,300,300,['Qwen 语言 Transformer','36 层 / hidden 2560','可训练，同一套共享权重','','一次能量计算过一次主干','因果顺序：[C, Z, A, E]'],ORANGE,23)
arrow([(1010,607),(1080,607)])
box(1450,455,295,300,['新增 Energy Head','取 e_E 位置的 hidden','LayerNorm(2560)','Linear 2560→256','GELU → Linear 256→1','可训练；不用 lm_head'],ORANGE,22)
arrow([(1380,607),(1450,607)])
box(1810,530,235,145,['共享标量能量 E','每样本一个数'],ORANGE,26)
arrow([(1745,607),(1810,607)])
# solver
rect(700,860,1345,195,GREEN)
text(1372,902,'内层：只修改连续候选；K=2 时 s=0、1/2',26,bold=True)
text(1372,944,['A_next = A − α_A · ∂E/∂A       Z_next = Z − α_Z · ∂E/∂Z',
                   'α_A、α_Z：两个独立的正值可学习标量，每次外层更新后改变（支持固定对照）',
                   '初值 0.05 / 200；限制在初值的 0.1–10 倍；两方向按条件分别优化'],22)
arrow([(1927,675),(1927,860)])
arrow([(700,990),(580,990),(580,1030),(287,1030),(287,995)],dashed=True)
arrow([(580,990),(565,990),(565,805),(540,805)],dashed=True)
text(1372,1093,'Stage 1：给真实 Z 预测 A；给真实 A 预测 Z。两方向依次反向，合并梯度后只更新一次权重。',22,bold=True)
text(1372,1132,'完整展开求导 → L_action + L_future → 更新主干、投影、Energy Head、迭代 MLP 与可学习步长',21)
text(40,1100,'橙色：可训练模块 / 参数',21,anchor='start');text(40,1135,'蓝色：冻结 DINO 视觉编码器',21,anchor='start');text(40,1170,'灰色：输入 / 固定编码；虚线：候选反馈',21,anchor='start')
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#48627e"/></marker></defs><g font-family="Noto Sans CJK SC, sans-serif">{''.join(items)}</g></svg>'''
OUT.mkdir(parents=True,exist_ok=True)
p=OUT/'LTE_WAM_ITERATION_AWARE_ARCHITECTURE_CN.svg';p.write_text(svg)
if __name__=='__main__':
    # Optional rendering dependency: CairoSVG and a Noto Sans CJK SC font.
    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(p.with_suffix('.png')),output_width=2520,output_height=1428)
    # Outline glyphs so the exported SVG is portable without installed CJK fonts.
    cairosvg.svg2svg(bytestring=svg.encode(),write_to=str(p))
    print(p)
