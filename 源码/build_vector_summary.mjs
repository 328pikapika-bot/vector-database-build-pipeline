import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "D:/project/vector_library_sync/outputs/vector_summary_20261020";
await fs.mkdir(outputDir, { recursive: true });

const wb = Workbook.create();
const summary = wb.worksheets.add("汇总");
const metadata = wb.worksheets.add("元数据明细");
const fulltext = wb.worksheets.add("全文与图片");
const notes = wb.worksheets.add("口径说明");
for (const s of [summary, metadata, fulltext, notes]) {
  s.showGridLines = false;
  s.getRange("A1:Z200").format.font = { name: "Aptos", size: 10, color: "#1F2937" };
  s.getRange("A1:Z200").format.verticalAlignment = "center";
}

const navy = "#17365D";
const blue = "#D9EAF7";
const pale = "#F3F6F9";
const amber = "#FFF2CC";

summary.getRange("A1:H1").merge();
summary.getRange("A1").values = [["V2 向量库数据汇总"]];
summary.getRange("A1:H1").format = { font: { name: "Aptos", size: 16, bold: true, color: navy }, rowHeight: 28 };
summary.getRange("A2:H2").merge();
summary.getRange("A2").values = [["统计时间：2026-10-20；集合：scientific_documents_v2 / scientific_fulltext_chunks_v2 / scientific_images_v2"]];
summary.getRange("A2:H2").format.font = { italic: true, color: "#5B6573" };
summary.getRange("A4:F4").values = [["指标", "数量（篇/条/张）", "约数量（万）", "近三年数量", "近三年占比", "统计说明"]];
summary.getRange("A4:F4").format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", wrapText: true };
summary.getRange("A5:F14").values = [
  ["元数据总量", 52978029, 5297.8029, 21376079, 21376079/52978029, "scientific_documents_v2 当前总量"],
  ["之江原始批次", 13337395, 1333.7395, 12894, 12894/13337395, "沿用历史旧批次 ID 基线；近三年按可核实重叠量"],
  ["近三年论文文献 J02/Z01（排除旧批次）", 14638424, 1463.8424, 14638424, 1, "2024—2026；J02 + Z01；已扣除旧批次交集 12,894"],
  ["近三年其他类型", 6724761, 672.4761, 6724761, 1, "2024—2026；sysSubType 不属于 J02/Z01；与旧批次重叠未完成全量复核"],
  ["扣除上两批后的其余元数据（估算）", 25002210, 2500.221, null, null, "总量 - 之江原始批次 - 近三年 J02/Z01（按旧批次交集修正）"],
  ["全文切片总量", 195571602, 19557.1602, null, null, "scientific_fulltext_chunks_v2 行数；不是文献篇数"],
  ["全文对应文献数", 7364998, 736.4998, 7111, 7111/7364998, "chunkIndex=0 行数；当前集合可识别 2024 年 7,111 篇，2025/2026 未识别"],
  ["图片总量", 22522900, 2252.29, 33230, 33230/22522900, "scientific_images_v2 行数；图片记录不是文献篇数"],
  ["图片对应文献数（估算）", 7289000, 728.9, null, null, "按历史约 3.09 张/篇折算；需全量 doc_id 去重才能得到精确值"],
  ["图片近三年对应文献数（估算）", 2013000, 201.3, null, null, "按近三年元数据比例 27.63% 折算，估算值"]
];
summary.getRange("B5:C14").format.numberFormat = [["#,##0"],["#,##0.0"],["#,##0.0"],["#,##0.0"],["#,##0.0"],["#,##0"],["#,##0"],["#,##0"],["#,##0"],["#,##0"]];
summary.getRange("B5:B14").format.numberFormat = "#,##0";
summary.getRange("C5:C14").format.numberFormat = "#,##0.0";
summary.getRange("D5:D14").format.numberFormat = "#,##0";
summary.getRange("E5:E14").format.numberFormat = "0.0%";
summary.getRange("A5:F14").format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
summary.getRange("A5:F14").format.wrapText = true;
summary.getRange("A9:F9").format.fill = amber;
summary.getRange("A13:F14").format.fill = pale;
summary.getRange("A15:F15").values = [["全文近三年文献数（估算）", 2035000, 203.5, null, null, "按元数据近三年比例 27.63% 折算；不是全文 year 字段直接计数"]];
summary.getRange("B15").format.numberFormat = "#,##0";
summary.getRange("C15").format.numberFormat = "#,##0.0";
summary.getRange("A15:F15").format = { fill: pale, wrapText: true };
summary.getRange("A16:F16").values = [["类别（用户指定口径）", "当前可直接统计情况", "结论"]];
summary.getRange("A16:F16").format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
summary.getRange("A17:C22").values = [
  ["Patent（专利）", "entityType 字段当前仅部分记录有值；sysSubType 中以 P 开头的代码为专利类", "近三年其他类型中专利占主要部分"],
  ["ProceedingsPaper（会议）", "C01/C02 等会议类代码可见", "存在，但未做全量类型映射"],
  ["DegreePaper（学位）", "D02 等代码可见", "存在，数量较小"],
  ["CorpusCompile（外文文集汇编）", "可见 CorpusCompile", "存在，数量较小"],
  ["Report（报告）", "R 类代码及 Report 可见", "存在，数量较小"],
  ["Book（外文科技丛书）", "B 类代码及 Book 可见", "存在，数量较小"],
];
summary.getRange("A17:C22").format.wrapText = true;

metadata.getRange("A1:H1").merge(); metadata.getRange("A1").values = [["元数据年份与批次明细"]];
metadata.getRange("A1:H1").format = { font: { size: 15, bold: true, color: navy } };
metadata.getRange("A3:D3").values = [["年份", "元数据篇数", "约数量（万）", "占元数据总量"]];
metadata.getRange("A3:D3").format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
const yearRows = [["1799-1940",1238683,123.8683,1238683/52978029],["1941-2019",20810644,2081.0644,20810644/52978029],["2020",1345517,134.5517,1345517/52978029],["2021",1506987,150.6987,1506987/52978029],["2022",1853515,185.3515,1853515/52978029],["2023",4227794,422.7794,4227794/52978029],["2024",9440026,944.0026,9440026/52978029],["2025",8556052,855.6052,8556052/52978029],["2026",3380001,338.0001,3380001/52978029],["年份为空",618804,61.8804,618804/52978029],["2027及以后",6,0.0006,6/52978029]];
metadata.getRange("A4:D14").values = yearRows;
metadata.getRange("A4:D14").format.numberFormat = [["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"],["@","#,##0","#,##0.0","0.0%"]];
metadata.getRange("A14:E14").values = [["批次分组", "数量（篇）", "约数量（万）", "年份范围", "类型概况"]];
metadata.getRange("A14:E14").format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
metadata.getRange("A15:E19").values = [
  ["之江原始批次",13337395,1333.7395,"历史旧批次，年份跨度以源数据为准","以 J02/Z01 论文文献为主；旧 ID 基线"],
  ["近三年论文文献（扣旧批次）",14638424,1463.8424,"2024-2026","J02、Z01"],
  ["近三年其他类型",6724761,672.4761,"2024-2026","专利、会议、学位、汇编、报告、图书等"],
  ["其余元数据（估算）",25002210,2500.221,"约 1940 年及以前至 2023 年，并含部分近三年非目标记录","历史期刊、专利、会议、学位、报告、图书等混合类型"],
  ["合计校验",52978029,5297.8029,"全量","上述分组按当前口径汇总"]
];
metadata.getRange("A4:E19").format.wrapText = true;
metadata.getRange("B4:B19").format.numberFormat = "#,##0";
metadata.getRange("C4:C19").format.numberFormat = "#,##0.0";

fulltext.getRange("A1:H1").merge(); fulltext.getRange("A1").values = [["全文与图片对应关系"]];
fulltext.getRange("A1:H1").format = { font: { size: 15, bold: true, color: navy } };
fulltext.getRange("A3:G3").values = [["集合", "记录数", "约数量（万）", "可识别文献数", "近三年可识别量", "近三年比例", "口径"]];
fulltext.getRange("A3:G3").format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", wrapText: true };
fulltext.getRange("A4:G5").values = [
  ["全文切片",195571602,19557.1602,7364998,7111,7111/7364998,"文献数=chunkIndex=0；全文记录主要缺少年份字段"],
  ["图片",22522900,2252.29,7289000,null,null,"文献数为按 3.09 张/篇估算；精确值需 doc_id 全量去重"]
];
fulltext.getRange("A7:G7").values = [["图片年份字段可识别统计", "图片数", "约数量（万）", "说明", "", "", ""]];
fulltext.getRange("A7:G7").format = { fill: navy, font: { bold: true, color: "#FFFFFF" } };
fulltext.getRange("A8:D16").values = [["2019",1101402,110.1402,"doc_meta.year"],["2020",1365890,136.589,"doc_meta.year"],["2021",1379521,137.9521,"doc_meta.year"],["2022",522591,52.2591,"doc_meta.year"],["2023",153728,15.3728,"doc_meta.year"],["2024",33230,3.323,"doc_meta.year"],["2025",0,0,"当前字段不可识别/未写入"],["2026",0,0,"当前字段不可识别/未写入"],["合计（列示年份）",null,null,"不是全量图片总量；仅列出已核对年份"]];
fulltext.getRange("A4:G16").format.wrapText = true;
fulltext.getRange("B4:B16").format.numberFormat = "#,##0";
fulltext.getRange("C4:C16").format.numberFormat = "#,##0.0";
fulltext.getRange("D4:E16").format.numberFormat = "#,##0";

notes.getRange("A1:F1").merge(); notes.getRange("A1").values = [["统计口径与限制"]];
notes.getRange("A1:F1").format = { font: { size: 15, bold: true, color: navy } };
notes.getRange("A3:B10").values = [
  ["时间范围", "近三年统一按 2024、2025、2026 年字段统计。"],
  ["V2 范围", "只使用 scientific_documents_v2、scientific_fulltext_chunks_v2、scientific_images_v2。"],
  ["之江批次", "13,337,395 篇沿用历史旧批次基线；近三年与旧批次可直接核实的交集为 12,894 篇。"],
  ["其余元数据", "约 2500.22 万篇是扣除旧批次和近三年 J02/Z01 后的估算剩余；年份和类型是混合分布，不代表单一批次。"],
  ["类型统计", "entityType 在当前集合中不是全量完整字段，因此类型栏用已知 sysSubType/样本映射描述，未虚构全量类型占比。"],
  ["全文文献数", "全文行数是切片；chunkIndex=0 用来估计文献数。当前可识别近三年只有 2024 年 7,111 篇。"],
  ["图片文献数", "图片集合存的是图片行；按历史约 3.09 张/篇折算约 728.9 万篇，需按全量 doc_id 去重才能精确。"],
  ["图片近三年", "图片 doc_meta.year 当前可识别 2024 年 33,230 张，2025/2026 为 0；这是字段可识别结果，不等同于业务年份真实为零。"]
];
notes.getRange("A3:B10").format.wrapText = true;

for (const s of [summary, metadata, fulltext, notes]) {
  s.getUsedRange().format.autofitColumns();
}
summary.getRange("A1:F22").format.columnWidth = 20;
summary.getRange("F5:F14").format.columnWidth = 42;
metadata.getRange("A1:E19").format.columnWidth = 20;
fulltext.getRange("A1:G16").format.columnWidth = 18;
fulltext.getRange("G4:G5").format.columnWidth = 42;
notes.getRange("A1:B10").format.columnWidth = 28;
notes.getRange("B3:B10").format.columnWidth = 95;
for (const s of [summary, metadata, fulltext, notes]) s.getUsedRange().format.autofitRows();
for (const s of [summary, metadata, fulltext, notes]) s.freezePanes.freezeRows(3);

wb.recalculate();
const preview = await wb.render({ sheetName: "汇总", range: "A1:F22", scale: 1, format: "png" });
await fs.writeFile(`${outputDir}/preview.png`, new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(`${outputDir}/V2向量库数据汇总_20261020.xlsx`);
const check = await wb.inspect({ kind: "table", sheetId: "汇总", range: "A1:F14", include: "values,formulas", tableMaxRows: 14, tableMaxCols: 6, maxChars: 7000 });
console.log(check.ndjson);
