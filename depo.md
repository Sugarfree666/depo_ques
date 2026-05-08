# Method

**总览：**

> 复杂问题
> ↓
> 识别实体 / 类型变量
> ↓
> 依存句法分析
> ↓
> AST 构建
> ↓
> 基于图的一跳关系生成原子子问题

**Q1：Which university did the CEO of the artificial intelligence company that developed AlphaGo graduate from and in which city is this university located?**

**Q2：Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?**

## 识别实体 / 类型 / 关系词 

需要识别三类东西：实体，类型变量（不是具体实体，是问题中出现的类别，角色，概念等）

|      | 实体                                                   | 类型变量                                                     |
| ---- | ------------------------------------------------------ | ------------------------------------------------------------ |
| Q1   | AlphaGo                                                | the artificial intelligence company<br/>CEO<br/>university<br/>city |
| Q2   | Ten9Eight: Shoot For The Moon<br/>Sabotage (1936 Film) | director，nationality                                        |

## 依存句法分析

由于传统依存句法分析是 token-level 的，在我们的例子中，"the artificial intelligence company"会被解析成四个独立的节点。但是我们在这一步想要的是实体/类型变量之间的依赖关系，所以我们根据第一步提取出的实体和变量，将原句中对应的实体和变量进行占位符替换。然后再对处理后的句子进行句法分析。

占位符替换，要注意最好设计的更像自然语言词，否则解析器可能会把它们当成奇怪的专有名词，导致parser质量下降。

> 使用大驼峰命名，并且使用类型+希腊字母的命名格式。[Type：company，the artificial intelligence company]->CompanyAlpha [person，director]->PersonAlpha等等。

这里使用CoreNLP Enhanced++ 解析，得到增强依存图。

## AST生成

我们先将问题结构分为两大类：串行结构和并行结构

- 前面生成的这个依存图中可能存在噪音。
- 这里我们把依存句法树看成带有权重的无向图，我们之前提取出来的实体和变量成为锚点，找到连接锚点的最小生成树（MST），我们这里只保留锚点，如果两个锚点之间有其他节点，直接删除，让锚点代替。
- 让LLM根据这个MST还有原问题在算子集合中选择一个算子，加入到共享节点
- 把占位符映射回去，得到最终的AST

## 原子子问题

拿到AST，我们可以从实体出发找相邻节点进行组合，$\{e_1, e_2\}$，要求LLM只用这两个实体/变量根据原问题生成一个子问题。答案就是中间实体 $X$，我们再用 $X$ 和图中的下一个节点进行组合得出原子问题，然后继续向外推演。

**例子：**Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?

1. 提取实体和变量：

   实体 1: FilmAlpha = film Ten9Eight: Shoot For The Moon

   实体 2: FilmBeta= film Sabotage (1936 Film)

   变量 1: PersonAlpha = director (第一个)

   变量 2: PersonBeta= director (第二个)

   变量 3: NationalityAlpha = nationality

2. "Do PersonAlpha of FilmAlpha and PersonBeta of FilmBeta share the same NationalityAlpha?"

   利用Stanford CoreNLP生成依存图

   > share
   > ├── nsubj: PersonAlpha
   > ├── nsubj: PersonBeta
   > ├── obj: NationalityAlpha
   > └── modifier: same
   >
   > PersonAlpha --of--> FilmAlpha
   > PersonBeta --of--> FilmBeta

3. 我们只关心锚点，找到连接这些锚点的最短路径得到

   > FilmAlpha ---- PersonAlpha
   >                                                  \
   >                                               share ---- NationalityAlpha
   >                                                  /
   > FilmBeta ---- PersonBeta

4. 如果两个锚点之间有其他的节点，我们可以直接将“非锚点节点”删去，让锚点节点代替

   > FilmAlpha ---- PersonAlpha
   >                                                  \
   >                                               NationalityAlpha
   >                                                  /
   > FilmBeta ---- PersonBeta

5. 让LLM根据这个图还有原问题选择一个算子，连接共享节点。再把实体还原，这个就是最终AST

   > film Ten9Eight: Shoot For The Moon ---- director
   >                                                				  				\
   >                                             								nationality-----COMPARE_SAME
   >                                                					  			/
   > film Sabotage (1936 Film) --------------director

6. 现在就可以根据这个图生成原子子问题了：

   节点（film Ten9Eight: Shoot For The Moon，director）可以生成

   q1: Who is the director of Ten9Eight: Shoot For The Moon?，答案是X1

   （X1，nationality）可以生成

   q2: What is the nationality of X1?

   同理：

   q3: Who is the director of Sabotage (1936 film)?
   q4: What is the nationality of X2?

   最后利用算子生成。

   q5: Are X1_nationality and X2_nationality the same?









> 请实现一个 Python 项目，严格按照 depo.md 中的 Method 实现“基于实体/类型变量关系图的一跳原子子问题拆解”。不要把方法改成普通的端到端子问题生成，也不要跳过依存图、MST/AST、邻接一跳生成这些步骤。
>
> 你必须先阅读并遵守仓库根目录下的 depo.md。不要自主修改方法，不要把它实现成普通的端到端问题拆解。核心流程必须严格是：
>
> 复杂问题
> → 识别实体 / 类型变量
> → 占位符替换
> → CoreNLP Enhanced++ 依存句法分析
> → 基于锚点构建 MST
> → 构建最终 AST
> → 基于 AST 的一跳相邻关系生成原子子问题
>
> 任务目标：
> 给定一个复杂问题，程序需要在控制台直接输出以下内容，要求易读、分段清晰，不要默认输出复杂 JSON 文件：
>
> 1. 原始问题
> 2. LLM 识别出的实体变量和类型变量
> 3. 占位符替换后的问题
> 4. 依存句法图，也就是 CoreNLP Enhanced++ dependency graph
> 5. 锚点 MST / anchor graph
> 6. 最终 AST
> 7. 基于 AST 一跳边生成的原子子问题
>
> 请使用 gpt-4o-mini 作为 LLM。
> API 配置必须支持 api_key 和 base_url：
> - 优先读取环境变量：
>   OPENAI_API_KEY
>   OPENAI_BASE_URL
> - 同时支持命令行参数：
>   --api-key
>   --base-url
> - 使用 openai Python SDK：
>   from openai import OpenAI
>   client = OpenAI(api_key=..., base_url=...)
>
> 输入要求：
> 1. 默认读取 questions.json 中的问题。
> 2. 同时支持命令行直接输入单个问题，例如：
>    python main.py --question "Which university did the CEO of the artificial intelligence company that developed AlphaGo graduate from and in which city is this university located?"
> 3. questions.json 需要兼容两种格式：
>    - ["question1", "question2"]
>    - [{"id": "q1", "question": "..."}, ...]
>
> 控制台输出格式要求：
> 输出必须是人类易读的，不要直接打印复杂嵌套 JSON。请使用类似下面的格式：
>
> ============================================================
> Question 1
> ============================================================
>
> [Original Question]
> Which university did the CEO of the artificial intelligence company that developed AlphaGo graduate from and in which city is this university located?
>
> [1. Entities and Type Variables]
> Entities:
>   - EntityAlpha: AlphaGo
>
> Type Variables:
>   - CompanyAlpha: the artificial intelligence company
>   - PersonAlpha: CEO
>   - UniversityAlpha: university
>   - CityAlpha: city
>
> [2. Placeholder Question]
> Which UniversityAlpha did the PersonAlpha of the CompanyAlpha that developed EntityAlpha graduate from and in which CityAlpha is this UniversityAlpha located?
>
> [3. Dependency Graph: Enhanced++]
> Edges:
>   - EntityAlpha --acl/developed--> CompanyAlpha
>   - CompanyAlpha --nmod/of--> PersonAlpha
>   - PersonAlpha --nmod/from--> UniversityAlpha
>   - UniversityAlpha --nmod/in--> CityAlpha
>
> 注意：这里的边格式可以根据 CoreNLP 实际输出调整，但必须能看出 source、dependency relation、target。
>
> [4. Anchor MST / Anchor Graph]
>   EntityAlpha ---- CompanyAlpha ---- PersonAlpha ---- UniversityAlpha ---- CityAlpha
>
> [5. Final AST]
>   AlphaGo ---- the artificial intelligence company ---- CEO ---- university ---- city
>
> Operators:
>   - NONE 或 BRIDGE
>
> [6. Atomic Subquestions]
>   q1: Which artificial intelligence company developed AlphaGo?
>       answer: X1
>
>   q2: Who is the CEO of X1?
>       answer: X2
>
>   q3: Which university did X2 graduate from?
>       answer: X3
>
>   q4: In which city is X3 located?
>       answer: X4
>
> 对于并行比较问题，例如：
> Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?
>
> 控制台输出应该类似：
>
> [Final AST]
>   Ten9Eight: Shoot For The Moon ---- director ---- nationality ---- COMPARE_SAME
>   Sabotage (1936 Film) ------------ director ---- nationality ---- COMPARE_SAME
>
> [Atomic Subquestions]
>   q1: Who is the director of Ten9Eight: Shoot For The Moon?
>       answer: X1
>
>   q2: What is the nationality of X1?
>       answer: X1_nationality
>
>   q3: Who is the director of Sabotage (1936 Film)?
>       answer: X2
>
>   q4: What is the nationality of X2?
>       answer: X2_nationality
>
>   q5: Are X1_nationality and X2_nationality the same?
>
> 实现细节：
>
> Step 1：LLM 识别实体和类型变量
> - 调用 gpt-4o-mini。
> - 识别两类核心节点：
>   1. entity：具体实体，例如 AlphaGo、Ten9Eight: Shoot For The Moon、Sabotage (1936 Film)
>   2. type_variable：问题中的类别、角色、概念，例如 the artificial intelligence company、CEO、university、city、director、nationality
> - 不要在这一步生成子问题。
> - 每个实体/类型变量都要分配一个占位符。
> - 占位符格式必须是自然语言风格的大驼峰命名，使用“语义类型 + 希腊字母序号”：
>   CompanyAlpha
>   PersonAlpha
>   FilmAlpha
>   FilmBeta
>   NationalityAlpha
>   UniversityAlpha
>   CityAlpha
> - 对重复类型变量必须保留多个节点，不能合并。
>   例如两个 director 应该是 PersonAlpha 和 PersonBeta。
>
> Step 2：占位符替换
> - 将原问题中的实体和类型变量替换成占位符。
> - 替换时要根据 span 从后往前替换，避免字符串位置错乱。
> - 替换后的问题要尽量保持自然语言结构，方便 CoreNLP parser 解析。
> - 保存 placeholder 到原始文本的映射，后面生成最终 AST 时要映射回来。
> - 控制台要输出：
>   - 占位符问题
>   - placeholder mapping
>
> Step 3：CoreNLP Enhanced++ 依存句法分析
> - 使用 Stanford CoreNLP server。
> - 默认地址：
>   http://localhost:9000
> - 支持命令行参数：
>   --corenlp-url
> - 使用 enhancedPlusPlusDependencies。
> - 依存图节点可以是 token，也可以保留 CoreNLP 返回的 token index。
> - 边至少要包含：
>   source
>   target
>   dependency relation
> - 控制台输出依存图时，不要输出难读的大 JSON，而是输出边列表：
>   source --relation--> target
> - 如果 CoreNLP 服务不可用，不要静默失败。请给出清晰错误提示，例如：
>   "CoreNLP server is not available. Please start Stanford CoreNLP server at http://localhost:9000."
>
> Step 4：构建锚点 MST / Anchor Graph
> - 锚点是所有实体和类型变量的 placeholder。
> - 将 dependency graph 看成带权无向图。
> - 边权默认 1。
> - 计算所有锚点之间的最短路径。
> - 在锚点 metric closure 上构建 MST。
> - MST 的边代表锚点之间的主要结构关系。
> - 如果两个锚点之间存在非锚点节点，则删除非锚点节点，让两个锚点直接相连。
> - 控制台输出 MST 时，使用可读 graph 形式，例如：
>   EntityAlpha ---- CompanyAlpha ---- PersonAlpha ---- UniversityAlpha ---- CityAlpha
> - 如果是并行结构，要保留分支形态，例如：
>   FilmAlpha ---- PersonAlpha
>                          \
>                           NationalityAlpha
>                          /
>   FilmBeta  ---- PersonBeta
>
> Step 5：生成最终 AST
> - 基于 anchor graph / MST 和原问题，让 LLM 只做“算子选择”和“共享节点连接”。
> - LLM 不能直接重写整个图。
> - LLM 不能直接生成所有子问题。
> - LLM 只能从固定算子集合中选择：
>   COMPARE_SAME：判断两个分支结果是否相同
>   COMPARE_DIFFERENT：判断两个分支结果是否不同
>   AND：并列约束
>   OR：选择关系
>   FILTER：限定或筛选
>   COUNT：计数
>   BRIDGE：串行桥接
>   NONE：不需要额外算子
> - 最终 AST 必须将 placeholder 映射回原始实体/类型变量文本。
> - 控制台输出最终 AST 时要用可读图结构，而不是复杂 JSON。
> - 示例：
>   AlphaGo ---- the artificial intelligence company ---- CEO ---- university ---- city
>
> Step 6：基于 AST 一跳关系生成原子子问题
> - 必须从 AST 的相邻一跳边生成子问题。
> - 每次只允许使用两个相邻节点和原问题。
> - 禁止一次使用三跳或多跳信息。
> - LLM 可以用于把一跳节点对改写成自然语言子问题，但提示词中必须限制它：
>   “只根据这两个节点和原问题生成一个原子子问题，不要引入其他节点的信息。”
> - 串行结构中，前一个子问题的答案变量 X1、X2、X3 会继续作为下一跳输入。
> - 例如：
>   AlphaGo ---- the artificial intelligence company ---- CEO ---- university ---- city
>   应生成：
>   q1: Which artificial intelligence company developed AlphaGo? answer X1
>   q2: Who is the CEO of X1? answer X2
>   q3: Which university did X2 graduate from? answer X3
>   q4: In which city is X3 located? answer X4
>
> - 并行比较结构中，先分别生成每个分支的原子子问题，再根据 operator 生成最终比较问题。
> - 例如 COMPARE_SAME：
>   q5: Are X1_nationality and X2_nationality the same?
>
> 代码结构建议：
> - main.py
>   命令行入口，读取问题，串联完整 pipeline，打印控制台结果。
> - llm_client.py
>   封装 gpt-4o-mini 调用，支持 api_key 和 base_url。
> - prompts.py
>   存放 LLM prompts，包括实体/变量提取、operator 选择、一跳子问题生成。
> - entity_extractor.py
>   实体/类型变量提取。
> - placeholder.py
>   占位符生成与替换。
> - corenlp_parser.py
>   CoreNLP Enhanced++ 解析。
> - graph_builder.py
>   dependency graph、shortest path、MST、anchor graph 构建。
> - ast_builder.py
>   AST 构建、operator 注入、placeholder 还原。
> - subquestion_generator.py
>   基于 AST 一跳边生成原子子问题。
> - io_utils.py
>   读取 questions.json。
> - requirements.txt
> - README.md
>   写清楚如何安装依赖、启动 CoreNLP、运行示例。
>
> 依赖：
> openai
> requests
> networkx
> tqdm
> pydantic 或 dataclasses
>
> LLM 调用要求：
> - 所有 LLM 中间输出可以使用 JSON 方便程序解析，但不要把复杂 JSON 作为最终控制台展示格式。
> - 如果 LLM 输出不是合法 JSON，要进行最多 3 次重试。
> - 每个 LLM prompt 都必须强调：
>   1. 不要直接做端到端子问题拆解
>   2. 只完成当前步骤
>   3. 不要引入原问题中没有的实体或类型变量
>   4. 不要合并重复出现但语义角色不同的变量
>   5. 输出必须可解析
>
> 重要约束：
> - 不要省略 CoreNLP Enhanced++ 依存句法分析。
> - 不要省略 MST / anchor graph。
> - 不要省略最终 AST。
> - 不要直接让 LLM 一次性生成所有子问题。
> - 原子子问题必须来自 AST 上的一跳相邻节点。
> - 每个子问题都要能对应到 AST 中的一条边或一个 operator。
> - 控制台输出必须清晰，便于我人工检查论文实验结果。
> - 不要默认保存复杂 JSON 文件。
> - 如果需要 debug，可以提供可选参数：
>   --debug
>   打印更详细的中间结构；
>   但默认输出必须简洁易读。
>
> 请直接生成完整可运行代码，不要只写伪代码。
> 实现完成后，请给出运行方式示例，包括：
> 1. 如何安装依赖
> 2. 如何启动 Stanford CoreNLP server
> 3. 如何运行 questions.json
> 4. 如何运行单个手动输入问题