"""Assemble the review from its maintained Markdown sources and render a reading PDF.
Run with the bundled Python (reportlab). No network or device work.
"""

from pathlib import Path
import html
import os
import re
from urllib.parse import urlparse, urljoin
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
OUT = ROOT / "output/pdf/runtime-mods-review.pdf"
MD = R / "report.md"

intro = """# vLLM-HUST 运行时 MOD 研究总报告

Fletcher · 共同阅读版 · 2026-10-27

## 1. 当前结论与证据地图

本报告汇集本仓库已有评审意见、源码研究与独立实验，不新增实验、不替作者发布结论。目录盘点覆盖24项；不是24项都完成了源码审计或设备验收。BetterScale与BidKV按Fletcher要求跳过。

**vSpec当前状态不合格。** 当前对照同时改变gamma预算、refill调度、capture buckets及动态宽度runtime，不能据此认定自适应策略相对宿主原生MTP2的独立收益。整套配置的观测耗时比保留在第3章，不作为策略收益认定；亦未证明额外graph/runtime支持造成提速。280题得分比原生下降3.57个百分点、比MTP2下降3.21个百分点，验收不接受这一准确率下降。下降的因果归因仍受截断与原生重复波动限制；不合格是当前验收判断，不是已证明算法必然损失精度。

**DiffSpec性能收益不合格，需要重新评估项目可行性。** 有效数据是统一B16/C04的三臂56题首轮，无抢占；默认检索计数为0。无有效的完整三臂280题结果。Fletcher已决定停止追加实验，不能用坏掉的原生Eagle链路充当算法反证。

| 方向 | 当前判断 | 证据性质与边界 |
|---|---|---|
| vSpec / 35B MTP | 当前不合格：收益归因未控制变量，准确率下降 | 280题质量与三轮56题性能；整套配置对照，非策略消融 |
| DiffSpec / 27B Eagle | 性能收益不合格，需重新评估项目可行性 | 我们的56题首轮；检索未生效，完整280题未完成 |
| OPset | 融合不合格：比ACLNN增时延48.37%–82.31% | 两卡算子实测；去清零另作有用小补丁 |
| Prefix Router | 有需求，值得做；优先联接分布式KV存储 | 方向判断与源码研究，不是生产验收 |
| StateAxis异步KV传输 | 原理正确，需适配35B-A3B及缓存压力验证 | 方向认可；35B适配与新workload尚未验证 |
| StateAxis自动前缀缓存 | 有潜力；需高效异步recurrent state checkpointing | 特定数据集前缀复用潜力，不能干扰计算 |
| StateAxis流水权重加载 | 实用，建议进入系统主线并默认启用 | 采用建议；历史收益不全归因于流水 |
| 其余有损KV、同步搬运、重算与调度路线 | 按具体实现作不采用或低优先级判断 | 见评审原文；不扩大成所有未来实现的否定 |

### 阅读顺序

- 第2章以两列表格汇总各MOD的当前正式评价，历史对话记录不收入报告。
- 第3章仅报告vSpec的35B主实验，以质量下滑、性能比较未控制变量这两项问题作评审。历史近似验收材料不收入本报告，仍保存在仓库。
- 第4章给出DiffSpec有效短对照与失败边界；第5章是OPset独立算子复现。
- 第6章合并StateAxis三条数据路径；第7章保留24项官网目录索引。

### 证据口径

**官网声明、源码观察、作者历史实验、我们的评审判断、我们的独立实测**是不同层级。本报告保留原始限定词，不把“measured / validated”自动解释为收益或生产可用。章节中的原始日期、历史“未重测”等表述均对应当时状态；当前状态以本章及第3、4章为准。

完整源码调用链仍保留在各专题principles.md；本报告通过可点击链接连接它们，不将全部日志和逐题输出灌入正文。本地证据链接需与本仓库一起使用，外部来源链接可独立打开。
"""


def include(rel, title, cut=None):
    p = ROOT / rel
    s = p.read_text()
    if cut:
        s = s.split(cut)[0]
    s = "\n".join(s.splitlines()[1:]).lstrip()
    # Promote each source's subheadings beneath one report chapter.
    s = re.sub(r"^(#{2,5}) ", lambda m: "#" + m.group(1) + " ", s, flags=re.M)

    def link(m):
        label, url = m.groups()
        if not urlparse(url).scheme and not url.startswith("#"):
            target, sep, anchor = url.partition("#")
            import os

            url = os.path.relpath(p.parent / target, R) + (sep + anchor if sep else "")
        return f"[{label}]({url})"

    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, s)
    return (
        f"\n\n## {title}\n\n来源：[专题原文]({rel.removeprefix('research/') if rel.startswith('research/') else '../' + rel})。\n\n"
        + s
    )


parts = [
    intro,
    include("research/comments.md", "2. MOD正式评审意见"),
    include(
        "research/speculative-mmlu-pro/results-qwen35.md",
        "3. vSpec：35B主实验",
        cut="## 证据入口",
    ),
    include(
        "research/speculative-mmlu-pro/results-diffspec.md",
        "4. DiffSpec：有效证据与停止边界",
    ),
    include("research/opset/reproduction/README.md", "5. OPset：两卡独立算子复现"),
    include(
        "research/stateaxis-data-paths/principles.md", "6. StateAxis：三条数据路径"
    ),
    include(
        "research/runtime-mods.md", "7. 官网24项目录索引", cut="## 接下来的研究问题"
    ),
]
text = "\n".join(parts)
# Keep infrastructure identity in the source evidence, not the reader-facing report.
text = text.replace(
    "`/root/lumi-opset-study-20261026`", "任务私有工作目录（精确路径见专题原文）"
)
MD.write_text(text)

pdfmetrics.registerFont(
    TTFont("CJK", "/System/Library/Fonts/STHeiti Light.ttc", subfontIndex=0)
)
pdfmetrics.registerFont(
    TTFont("CJK-Bold", "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0)
)
pdfmetrics.registerFontFamily(
    "CJK", normal="CJK", bold="CJK-Bold", italic="CJK", boldItalic="CJK-Bold"
)
INK = colors.HexColor("#203345")
TEAL = colors.HexColor("#137E83")
GRAY = colors.HexColor("#667887")
styles = {
    "body": ParagraphStyle(
        "body",
        fontName="CJK",
        fontSize=9.9,
        leading=15.4,
        textColor=INK,
        wordWrap="CJK",
        spaceAfter=6,
        splitLongWords=True,
    ),
    "h1": ParagraphStyle(
        "h1", fontName="CJK-Bold", fontSize=27, leading=37, textColor=INK, spaceAfter=24
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName="CJK-Bold",
        fontSize=19,
        leading=27,
        textColor=TEAL,
        spaceAfter=16,
        keepWithNext=True,
    ),
    "h3": ParagraphStyle(
        "h3",
        fontName="CJK-Bold",
        fontSize=12.5,
        leading=20,
        textColor=INK,
        spaceBefore=10,
        spaceAfter=7,
        keepWithNext=True,
    ),
    "cell": ParagraphStyle(
        "cell",
        fontName="CJK",
        fontSize=8.1,
        leading=12.5,
        textColor=INK,
        wordWrap="CJK",
        splitLongWords=True,
    ),
    "reject": ParagraphStyle(
        "reject",
        fontName="CJK-Bold",
        fontSize=19,
        leading=27,
        textColor=colors.HexColor("#CC1825"),
        spaceBefore=10,
        spaceAfter=14,
        wordWrap="CJK",
    ),
    "reviewcell": ParagraphStyle(
        "reviewcell",
        fontName="CJK",
        fontSize=9.2,
        leading=14,
        textColor=INK,
        wordWrap="CJK",
    ),
    "badcell": ParagraphStyle(
        "badcell",
        fontName="CJK-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#CC1825"),
        wordWrap="CJK",
    ),
    "code": ParagraphStyle(
        "code",
        fontName="CJK",
        fontSize=8.3,
        leading=12,
        textColor=INK,
        wordWrap="CJK",
        backColor=colors.HexColor("#F0F4F6"),
        spaceAfter=8,
    ),
}


def inline(s):
    s = s.replace("–", "-").replace("—", "-").replace("\u2011", "-")
    tokens = {}

    def protect(m):
        key = f"ZZLINK{len(tokens)}ZZ"
        label, url = m.groups()
        if not urlparse(url).scheme and not url.startswith("#"):
            target, sep, anchor = url.partition("#")
            base = os.environ.get("REPORT_LINK_BASE")
            url = (
                urljoin(base, url)
                if base
                else (R / target).resolve().as_uri() + (sep + anchor if sep else "")
            )
        tokens[key] = (
            f'<a href="{html.escape(url, quote=True)}" color="#137E83">{html.escape(label)}</a>'
        )
        return key

    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", protect, s)
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r'<font color="#476477">\1</font>', s)
    for k, v in tokens.items():
        s = s.replace(k, v)
    return s


def paragraph(s, kind="body"):
    if kind == "body" and s.startswith(("**不合格：", "**性能收益不合格", "**评语：")):
        kind = "reject"
    if kind == "cell" and re.fullmatch(r"\+\d+\.\d+%", s):
        kind = "badcell"
    return Paragraph(inline(s), styles[kind])


class Report(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if (
            self.page > 1
            and isinstance(flowable, Paragraph)
            and flowable.style.name in {"h2", "h3"}
        ):
            title = flowable.getPlainText()
            key = f"section-{self.page}-{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(
                title, key, level=0 if flowable.style.name == "h2" else 1, closed=False
            )


W, H = A4
story = []
lines = text.split("\n")
i = 0
ordered_index = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line and not re.match(r"^\d+\. ", line):
        ordered_index = 0
    if line.startswith("```"):
        buf = []
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            buf.append(lines[i])
            i += 1
        story.append(
            Paragraph("<br/>".join(html.escape(x) for x in buf), styles["code"])
        )
        i += 1
        continue
    if line.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = lines[i].strip().strip("|").split("|")
            if not all(re.fullmatch(r"\s*:?-+:?\s*", c) for c in cells):
                rows.append([paragraph(c.strip(), "cell") for c in cells])
            i += 1
        n = len(rows[0])
        avail = W - 108
        review_table = n == 2 and rows[0][1].getPlainText() == "正式评价"
        if review_table:
            for row in rows[1:]:
                row[1] = Paragraph(row[1].text, styles["reviewcell"])
        weights = (
            [1, 3.3]
            if n == 2
            else [1.6, 2.2, 2.8]
            if n == 3
            else [1.7, 1.3, 1.0, 1.3]
            if n == 4
            else [2.2] + [1] * (n - 1)
        )
        if n == 4 and "MOD /" in rows[0][0].getPlainText():
            weights = [2.1, 1.7, 1.65, 1]
        if n == 4 and rows[0][0].getPlainText() == "系统项":
            weights = [0.85, 1.2, 1.6, 2.15]
        widths = [avail * w / sum(weights) for w in weights]
        chunks = [rows] if len(rows) < 20 else [rows[:13], [rows[0]] + rows[13:]]
        for chunk in chunks:
            table = Table(chunk, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEDEE")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#F4F7F9")],
                        ),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            story.extend([table, Spacer(1, 12)])
        continue
    m = re.match(r"^(#{1,6}) (.*)", line)
    if m:
        level = len(m[1])
        title = m[2]
        if level == 2:
            story.append(PageBreak())
        story.append(
            paragraph(title, "h1" if level == 1 else "h2" if level == 2 else "h3")
        )
        i += 1
        continue
    if line.startswith("- ") or re.match(r"^\d+\. ", line):
        if line.startswith("- "):
            line = "• " + line[2:]
        else:
            ordered_index += 1
            line = re.sub(r"^\d+\. ", f"{ordered_index}. ", line)
        story.append(paragraph(line))
        i += 1
        continue
    buf = [line]
    i += 1
    while (
        i < len(lines)
        and lines[i].strip()
        and not re.match(r"^(#|\||-|```|\d+\. )", lines[i].strip())
    ):
        buf.append(lines[i].strip())
        i += 1
    story.append(paragraph(" ".join(buf)))


def page(c, doc):
    c.saveState()
    c.setFont("CJK", 8)
    c.setFillColor(GRAY)
    if doc.page > 1:
        c.drawString(48, H - 30, "vLLM-HUST · 运行时 MOD 研究总报告")
        c.setStrokeColor(colors.HexColor("#D8E2E8"))
        c.line(48, H - 39, W - 48, H - 39)
    c.drawString(48, 27, "Fletcher  |  共同阅读版  |  2026-10-27")
    c.drawRightString(W - 48, 27, str(doc.page))
    c.restoreState()


# Intentional compact cover instead of a nearly empty opening page.
story.insert(2, Spacer(1, 35))
story.insert(3, paragraph("源码审阅 · 采用判断 · 独立复现", "h3"))
story.insert(
    4,
    paragraph(
        "本轮实验已收口。保留正结果、失败证据与未完成范围，不以原先的立场代替实测。"
    ),
)
story.insert(5, Spacer(1, 24))
story.insert(
    6,
    paragraph(
        "重点阅读：第3章 vSpec 35B主实验；第4章 DiffSpec有效短对照；第5章 OPset独立算子结果。"
    ),
)
OUT.parent.mkdir(parents=True, exist_ok=True)
Report(
    str(OUT),
    pagesize=A4,
    rightMargin=48,
    leftMargin=48,
    topMargin=58,
    bottomMargin=50,
    title="vLLM-HUST 运行时 MOD 研究总报告",
    author="Fletcher",
).build(story, onFirstPage=page, onLaterPages=page)
print(MD)
print(OUT)
