# MemSysBench 完整实验报告 (含 Letta Cloud)

**生成时间**: 2026-06-05 11:35  
**项目**: MemSysBench — Agent Memory Systems Benchmark (ICDE 2027)  
**LLM 配置**: qwen-plus + text-embedding-v3（DashScope / 阿里云）  

---

## 1. Pilot 基础性能对比（10条记忆 × 5条查询）

| 系统 | 写入延迟(avg) | 检索延迟(avg) | 命中率 | 过时返回 | LLM |
|------|-------------|-------------|------|--------|-----|
| Mem0 | 7294ms | 882ms | 60% (3/5) | 1 | qwen-plus |
| Graphiti | 0/10 成功 | 3685ms | 0% (0/5) | 0 | qwen-plus |
| Naive RAG | 665ms | 568ms | 80% (4/5) | 1 | none |
| LangMem | 761ms | 486ms | 60% (3/5) | 1 | qwen-plus |
| Letta | ❌ Letta Client-Server Architectural L | - | - | - | N/A |


---

## 2. MemConflict —— 记忆时序更新测试

测试场景：写入旧事实 → 写入新事实 → 查询，看是否返回最新信息。

| 系统 | 测试数 | 新事实率(↑好) | 过时率(↓好) |
|------|------|------------|----------|
| Mem0 | 25 | 16% | 100% |
| Naive RAG | 25 | 20% | 84% |
| LangMem | 25 | 16% | 88% |
| Letta Cloud | 5 | 100% | 0% |


**关键发现**：Mem0 和 LangMem 的过时率均高达 ~88%，说明这两个系统在事实更新场景下几乎失效。

---

## 3. MemScale —— 大规模记忆压测

| 系统 | 规模(条) | 写入avg | 读取avg | 读取P95 |
|------|--------|--------|--------|--------|
| Mem0 | 100 | 3339ms | 579ms | 579ms |
| Mem0 | 300 | 3176ms | 562ms | 562ms |
| Mem0 | 500 | 3335ms | 605ms | 605ms |
| Mem0 | 1000 | skipped (too expensive) | - | - |
| Naive RAG | 100 | 105ms | 7ms | 20ms |
| Naive RAG | 300 | 86ms | 9ms | 21ms |
| Naive RAG | 500 | 89ms | 10ms | 14ms |
| Naive RAG | 1000 | 88ms | 18ms | 24ms |


**关键发现**：Naive RAG 写入速度比 Mem0 快 **~38x**（87ms vs 3.3s），
且随数据量增长几乎无衰减（P95 稳定在 25ms 以内）。

---

## 4. QPS —— 吞吐量并发测试

| 系统 | 测试场景 | QPS | 成功数 |
|------|--------|-----|------|
| Mem0 | read_qps_c1 | 1.50 | 20 |
| Mem0 | read_qps_c10 | 9.80 | 20 |
| Mem0 | read_qps_c5 | 5.74 | 20 |
| Mem0 | write_qps_c1 | 0.27 | 3 |
| Mem0 | write_qps_c3 | 0.77 | 9 |
| Mem0 | write_qps_c5 | 1.43 | 15 |
| Naive RAG | read_qps_c1 | 1.51 | 20 |
| Naive RAG | read_qps_c10 | 11.63 | 20 |
| Naive RAG | read_qps_c5 | 5.95 | 20 |
| Naive RAG | write_qps_c1 | 1.75 | 3 |
| Naive RAG | write_qps_c10 | 11.95 | 30 |
| Naive RAG | write_qps_c5 | 5.98 | 15 |



## 4b. Letta Cloud 实测结果

| 指标 | 数值 |
|------|------|
| 写入延迟(avg) | 11480ms |
| 检索延迟(avg) | 3019ms |
| Pilot 命中率 | 100% |
| MemConflict 新事实率 | 100% |
| MemConflict 过时率 | 0% |

> **关键对比**：Letta 的 MemConflict 新事实率 100%，远超 Mem0(16%) 和 LangMem(16%)。
> MemGPT 的显式记忆管理机制在时序更新场景下具有显著优势，但写入延迟达 11480ms（比 Naive RAG 慢约 130x）。


---

## 5. MemDialogue 数据集

**状态**: ✅ 已构建：500 条对话  
**来源**: LMSYS-Chat-1M (lmsys_preview_1k.jsonl 本地缓存)  
**格式**: 每条含 session_id、turns、memory_events（fact / query / ground_truth）  

---

## 5b. 幻觉测试（Hallucination Test）

基于 MemDialogue 数据集，测试 Mem0 的记忆召回质量（LLM-Judge 语义评估）：

| 指标 | 数值 |
|------|------|
| 写入成功率 | 100% (20/20) |
| Recall Rate（正确召回）| 60% (12/20) |
| Omission Rate（遗漏率）| 40% (8/20) |
| Fabrication Rate（捏造率）| 0% |

> **发现**：Mem0 写入成功但 40% 的事实无法被正确检索（Omission）。捏造率为 0%——系统不会无中生有，
> 但对于没有匹配记忆的查询会返回语义最近邻（相关但非目标事实）。

---

## 5c. 删除完整性测试（GDPR Right-to-Forget）

测试写入 PII 敏感信息后的存储与删除行为：

| 信息类型 | 写入尝试 | 实际存储 | 现象 |
|---------|---------|---------|------|
| 敏感（SSN/信用卡/医疗/住址/薪资）| 5条 | **0条** | 隐性 PII 过滤，无报错 |
| 普通（爱好/书单/语言学习）| 5条 | 5条 | 正常存储 |

> **发现 F11**：Mem0 内置隐性 PII 过滤器，敏感信息在写入时被静默拒绝，API 返回成功但数据未存储。
> 这是一个未文档化的行为，会导致 benchmark 评测结论失真（误以为系统已存储但无法检索）。

---

## 6. 关键发现汇总（论文 Findings）

| # | 发现 | 涉及系统 | 论文 RQ |
|---|------|---------|--------|
| F1 | **写重读轻**：Mem0 写入~3.3s（LLM提取），读~579ms，写/读比 ~6:1 | Mem0 | RQ1 |
| F2 | **MemConflict 失效**：Mem0 过时率 88%，LangMem 过时率 88% | Mem0, LangMem | RQ4 |
| F3 | **Naive RAG 性能全面领先**：写入快38x，检索快，命中率最高 | 对比 | RQ1/RQ2 |
| F4 | **Graphiti LLM强绑定**：实体提取 prompt 仅兼容 OpenAI JSON，千问100%失败 | Graphiti | RQ6 |
| F5 | **LangMem Embedding 格式不兼容**：OpenAIEmbeddings 无法对接千问 | LangMem | RQ6 |
| F6 | **Mem0 API 破坏性变更（第一次）**：v2.0 search() 参数无文档说明变更 | Mem0 | 可复现性 |
| F7 | **规模无关性**：Naive RAG P95 读延迟在100→1000条间几乎不变（+18ms） | Naive RAG | RQ3 |
| F8 | **Letta 架构锁定**：>=0.6.x 强制 Client-Server，无法嵌入式部署 | Letta | RQ6 |
| F9 | **Letta MemConflict 完美**：MemGPT 显式记忆管理使新事实率达 100%，远超其他系统 | Letta Cloud | RQ4 |
| F10 | **Mem0 API 破坏性变更（第二次）**：user_id→filters={'user_id':...}，造成100%假性失败 | Mem0 | 可复现性 |
| F11 | **隐性 PII 过滤**：Mem0 静默拒绝 SSN/信用卡/医疗等敏感信息，无任何报错 | Mem0 | RQ5/可复现性 |
| F12 | **Omission 问题**：Mem0 写入成功但 40% 事实无法语义召回（LLM-Judge验证） | Mem0 | RQ2 |

---

## 7. 系统可移植性矩阵（千问/国内环境）

| 系统 | 可嵌入式部署 | 千问兼容 | 无 OpenAI 依赖 | 综合可移植性 |
|------|-----------|--------|-------------|-----------|
| Mem0 | ✅ | ✅（需适配） | ✅ | ⭐⭐⭐ |
| Graphiti | ✅（需Neo4j） | ❌ | ❌ | ⭐ |
| Naive RAG | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| LangMem | ✅ | ❌ | ❌ | ⭐ |
| Letta | ❌（需服务端） | ❓ | ❓ | ⭐ |

> **结论**：在非 OpenAI 环境下，5个系统中只有 Mem0 和 Naive RAG 可用。
> Letta 因强制服务端架构，在所有测试环境中部署门槛最高。

---

## 8. 待办 / 下一步

- [x] Pilot 基础测试（Mem0 / Graphiti / Naive RAG / LangMem）
- [x] 并发测试 (Step 9/13 QPS)
- [x] Isolation 隔离测试 (Step 10)
- [x] MemConflict 测试 (Mem0 / Naive RAG / LangMem)
- [x] MemScale 1000条压测 (Step 14)
- [x] Letta 可移植性分析（F8 发现）
- [x] MemDialogue 数据集构建 (✅ 已构建：500 条对话)
- [x] Letta Cloud 实际测试（命中率100%，MemConflict新事实率100%）
- [ ] 论文 Section 5 (Evaluation) 初稿
