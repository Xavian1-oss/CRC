# CRC 审稿复验 — 任务文档（Task）

> **文档状态**：Draft v0.3  
> **依赖**：`review_spec/requirements.md`, `review_spec/design.md`  
> **Table R1**：**[x] 完成** — 后续以 **rebuttal 文稿 + P1 实验** 为主

---

## 图例

- `[ ]` 未开始 · `[x]` 已完成 · `[—]` 跳过

---

## Phase 0：Table R1 & gnqC【已完成】

| ID | 任务 | 状态 |
|----|------|------|
| T0-1 | Table R1 填入 G0–G5 | [x] |
| T0-2 | G4 B1–B3 | [x] |
| T0-3 | G2 实现 + B1–B3 | [x] |
| T0-4 | `rebuttal_gnqC.md` 定稿 | [x] |

---

## Phase R-gnqC：Reviewer gnqC

| ID | 任务 | 状态 |
|----|------|------|
| RG-1 | 粘贴 OpenReview rebuttal | [ ] |

> 正文：`review_spec/rebuttal_gnqC.md`

---

## Phase R-vaUX：Reviewer vaUX

| ID | 任务 | 状态 | 验收 |
|----|------|------|------|
| RV-1 | 撰写 `rebuttal_vaUX.md` | [x] | G0/G1/G5 三选一叙事 |
| RV-2 | 回应校准集 / 背景 / Table1 NDR | [ ] | revision 承诺 |
| RV-3 | OpenReview 提交 | [ ] | |

---

## Phase R-14dv：Reviewer 14dv

| ID | 任务 | 状态 | 验收 |
|----|------|------|------|
| R14-1 | 撰写 `rebuttal_14dv.md` | [x] | 承诺 seed + PIR 检验 |
| R14-2 | **R14-E1** multi-seed（B1 G5+G1 最小） | [x] | `rebuttal_results.md` §Seeds |
| R14-3 | **R14-E2** PIR 显著性脚本/表 | [x] | `r14_e2_pir_crc_significance.md` |
| R14-4 | OpenReview 提交 | [ ] | |

---

## Phase R-MkCF：Reviewer MkCF

| ID | 任务 | 状态 |
|----|------|------|
| RM-1 | 撰写 `rebuttal_MkCF.md` | [x] |
| RM-2 | **RMK-E1** top_k sweep B1 | [x] |
| RM-3 | OpenReview 提交 | [ ] |

---

## Phase R-AC：Area Chair

| ID | 任务 | 状态 |
|----|------|------|
| RAC-1 | 撰写 `rebuttal_AC.md`（AC-Q1/Q2 + Meta） | [x] |
| RAC-2 | 粘贴至 OpenReview（Author-AC 或 discussion） | [ ] |

---

## Phase REV：Revision backlog（会后）

| ID | 任务 | 状态 |
|----|------|------|
| REV-T1 | 主表 appendix G0/G1/G5 扩展 | [ ] |
| REV-T2 | Table 1 NDR 列 | [ ] |
| REV-T3 | NDR–ΔMAE 图 | [ ] |
| REV-T4 | blend zero_rate | [ ] |
| REV-T5 | Related work TAFAS/SOLID/PETSA/FAC/FFORMA | [ ] |
| REV-T6 | Appendix A + 校准集 Methods | [ ] |

---

## 执行顺序（rebuttal 截止前）

```
1. [x] Table R1
2. [ ] rebuttal_vaUX.md（快，复用 Table R1）
3. [ ] rebuttal_MkCF.md
4. [ ] rebuttal_AC.md（最长，新颖性）
5. [ ] rebuttal_14dv.md + 尽量跑 R14-E1 preview
6. [ ] 五条 OpenReview 粘贴
```

---

## 阻塞项

| 日期 | ID | 描述 | resolution |
|------|-----|------|------------|
| — | — | — | — |

---

## 附录：Reviewer → 文件

| Reviewer | Rebuttal 文件 | 关键实验 |
|----------|---------------|----------|
| gnqC | `rebuttal_gnqC.md` | Table R1 |
| vaUX | `rebuttal_vaUX.md` | Table R1 G0/G1/G5 |
| 14dv | `rebuttal_14dv.md` | R14-E1/E2 |
| MkCF | `rebuttal_MkCF.md` | RMK-E1 |
| AC | `rebuttal_AC.md` | 文案 + Table R1 |
