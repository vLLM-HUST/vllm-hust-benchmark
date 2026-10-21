# 数据格式修改说明

## 🔧 主要修改

### 1. 新增 `config_type` 字段（顶层）

**目的**：明确区分三种配置类型

```json
{
  "config_type": "single_gpu" | "multi_gpu" | "multi_node"
}
```

| 配置类型     | `config_type` 值 | `cluster` 字段   | `hardware.chip_count` |
| ------------ | ---------------- | ---------------- | --------------------- |
| **单机单卡** | `"single_gpu"`   | `null`           | `1`                   |
| **单机多卡** | `"multi_gpu"`    | `null`           | `> 1`                 |
| **多机多卡** | `"multi_node"`   | `{...}` (非null) | `> 1`                 |

______________________________________________________________________

### 2. `hardware` 字段细化

#### ❌ 旧字段（有歧义）

```json
{
  "chip_count": 8,         // 总芯片数？每节点芯片数？
  "interconnect": "HCCS"   // 节点内还是节点间？
}
```

#### ✅ 新字段（明确语义）

```json
{
  "chip_count": 8,                      // 总芯片数（所有节点）
  "chips_per_node": 4,                  // 每节点芯片数
  "intra_node_interconnect": "HCCS"    // 节点内卡间互联
}
```

**新增字段**：

| 字段名                    | 类型    | 必填        | 说明           | 示例                                     |
| ------------------------- | ------- | ----------- | -------------- | ---------------------------------------- |
| `chips_per_node`          | integer | ⚠️ 多机必填 | 每节点芯片数   | `4`, `8`                                 |
| `intra_node_interconnect` | string  | ✅          | 节点内卡间互联 | `"NVLink"`, `"HCCS"`, `"PCIe"`, `"None"` |

**废弃字段**：

- ~~`interconnect`~~ → 拆分为 `intra_node_interconnect` + `cluster.inter_node_network`

______________________________________________________________________

### 3. `cluster` 字段细化（多机配置）

#### ❌ 旧字段（混淆网络和拓扑）

```json
{
  "topology_type": "Multi-Node-InfiniBand"  // 混合了网络类型和拓扑
}
```

#### ✅ 新字段（分离网络和拓扑）

```json
{
  "inter_node_network": "InfiniBand",       // 节点间网络类型
  "network_bandwidth_gbps": 200,            // 网络带宽
  "topology_type": "Ring"                   // 物理拓扑
}
```

**新增/修改字段**：

| 字段名                   | 类型    | 必填 | 说明             | 示例                                                      |
| ------------------------ | ------- | ---- | ---------------- | --------------------------------------------------------- |
| `inter_node_network`     | string  | ✅   | 节点间网络类型   | `"InfiniBand"`, `"Ethernet"`, `"RoCE"`, `"NVLink-Switch"` |
| `network_bandwidth_gbps` | integer | ⬜   | 网络带宽（Gbps） | `100`, `200`, `400`                                       |
| `topology_type`          | string  | ✅   | 物理拓扑         | `"Ring"`, `"Tree"`, `"Mesh"`, `"All-to-All"`              |

**说明**：

- `inter_node_network` - **网卡类型**（这是你问题2关心的）
- `topology_type` - **连接方式**（如何组网）

______________________________________________________________________

### 4. `workload` 字段补全

#### ❌ 旧数据（缺失字段）

```json
{
  "dataset": "AlpacaEval"  // 只有数据集名称
}
```

#### ✅ 完整数据

```json
{
  "input_length": 2048,
  "output_length": 512,
  "batch_size": 4,
  "concurrent_requests": 8,
  "dataset": "AlpacaEval"
}
```

______________________________________________________________________

### 5. `metadata` 字段补全

#### ❌ 旧数据（缺失字段）

```json
{
  "submitted_at": "...",
  "submitter": "...",
  "git_commit": "...",
  "release_date": "...",
  "notes": "...",
  "verified": true
}
```

#### ✅ 完整数据（必需字段）

```json
{
  "submitted_at": "2026-01-28T11:00:00Z",
  "submitter": "HUST",
  "data_source": "automated-benchmark",        // ← 新增
  "reproducible_cmd": "vllm bench serve ...", // ← 新增（重要！）
  "git_commit": "a1b2c3d4...",
  "release_date": "2026-01-27",
  "changelog_url": "https://...",               // ← 新增
  "notes": "...",
  "verified": true
}
```

______________________________________________________________________

## 📋 完整示例对比

### 示例 1：单机单卡

```json
{
  "config_type": "single_gpu",
  "hardware": {
    "chip_count": 1,
    "chips_per_node": 1,
    "intra_node_interconnect": "None"
  },
  "cluster": null
}
```

### 示例 2：单机多卡

```json
{
  "config_type": "multi_gpu",
  "hardware": {
    "chip_count": 8,
    "chips_per_node": 8,
    "intra_node_interconnect": "NVLink"
  },
  "cluster": null
}
```

### 示例 3：多机多卡（你的场景）

```json
{
  "config_type": "multi_node",
  "hardware": {
    "chip_count": 8,              // 总卡数 = 2 节点 × 4 卡/节点
    "chips_per_node": 4,          // 每节点 4 卡
    "intra_node_interconnect": "HCCS"  // 节点内：HCCS
  },
  "cluster": {
    "node_count": 2,
    "inter_node_network": "InfiniBand",  // 节点间：InfiniBand 网卡
    "network_bandwidth_gbps": 200,       // 200Gbps
    "topology_type": "Ring"              // Ring 拓扑
  }
}
```

______________________________________________________________________

## 🐛 你的原始数据的问题

### 问题 1：缺少 `config_type`

- **影响**：前端无法快速区分配置类型
- **修复**：添加 `"config_type": "multi_node"`

### 问题 2：`interconnect` 语义混淆

- **问题**：`"interconnect": "HCCS"` 是节点内还是节点间？
- **修复**：
  - `"intra_node_interconnect": "HCCS"` （节点内）
  - `"inter_node_network": "InfiniBand"` （节点间）

### 问题 3：网卡类型不明确

- **问题**：`"topology_type": "Multi-Node-InfiniBand"` 混合了网络和拓扑
- **修复**：
  - `"inter_node_network": "InfiniBand"` （网卡类型）
  - `"topology_type": "Ring"` （拓扑类型）
  - `"network_bandwidth_gbps": 200` （网络带宽）

### 问题 4：缺少 `chips_per_node`

- **问题**：`chip_count=8, node_count=2`，但每节点几卡不明确
- **修复**：添加 `"chips_per_node": 4`

### 问题 5：`workload` 不完整

- **问题**：只有 `dataset`，缺少 `input_length`, `output_length` 等
- **修复**：补全所有必需字段

### 问题 6：`metadata` 不完整

- **问题**：缺少 `data_source`, `reproducible_cmd`, `changelog_url`
- **修复**：补全可复现性必需字段

______________________________________________________________________

## 🎯 网卡类型字段总结（回答你的问题2）

**✅ 是的，现在有明确的网卡类型字段了**：

```json
{
  "cluster": {
    "inter_node_network": "InfiniBand",  // ← 这就是网卡类型
    "network_bandwidth_gbps": 200        // ← 网络带宽
  }
}
```

**支持的网卡类型**：

- `"InfiniBand"` - 高性能互联（100-400 Gbps）
- `"Ethernet"` - 以太网（10-100 Gbps）
- `"RoCE"` - RDMA over Converged Ethernet
- `"NVLink-Switch"` - NVIDIA NVSwitch（节点间）

______________________________________________________________________

## 🔍 验证清单

使用修改后的数据时，请检查：

- [ ] ✅ 有 `config_type` 字段
- [ ] ✅ `hardware.intra_node_interconnect` 正确（节点内）
- [ ] ✅ `cluster.inter_node_network` 正确（节点间网卡）
- [ ] ✅ `cluster.topology_type` 只描述拓扑（不包含网络类型）
- [ ] ✅ `hardware.chips_per_node` 与 `chip_count/node_count` 一致
- [ ] ✅ `workload` 包含所有必需字段
- [ ] ✅ `metadata.reproducible_cmd` 存在（可复现性）
- [ ] ✅ `environment` 字段完整
