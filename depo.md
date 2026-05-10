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

为了方便后续依存图生成，我们使用大驼峰命名来给识别出来的实体/变量来取"别名"，并且使用类型+希腊字母的命名格式。[Type：company，the artificial intelligence company]->CompanyAlpha [person，director]->PersonAlpha等等。

这里使用CoreNLP Enhanced++ 解析，得到增强依存图。将原问题直接交给CoreNLP Enhanced++，生成依存图结构。这个图结构可能很复杂，比如一个实体Ten9Eight: Shoot For The Moon会被拆的乱七八糟，我们可以进行节点折叠，找到film节点，将下面一串合并成一个超级节点，用FilmAlpha这一个词来代替。同理将其他对应的实体/变量都替换掉就会得到我们想要的结构。

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

   > ROOT: share (VERB)
   >  ├── aux: Do (AUX)
   >  │
   >  ├── nsubj: director (NOUN)       
   >  │    ├── nmod:of: film          
   >  │    │    └── appos/dep: Ten9Eight
   >  │    │         ├── punct: :
   >  │    │         └── dep: Shoot
   >  │    │              └── nmod:for: Moon
   >  │    │                   └── det: The
   >  │    │
   >  │    └── conj:and: director     
   >  │
   >  ├── cc: and (CCONJ)
   >  │
   >  ├── nsubj: director (NOUN)      
   >  │    └── nmod:of: film
   >  │         └── appos: Sabotage
   >  │              ├── punct: (
   >  │              ├── appos: Film
   >  │              │    └── nummod: 1936
   >  │              └── punct: )
   >  │
   >  ├── obj: nationality (NOUN)
   >  │    ├── det: the
   >  │    └── amod: same
   >  │
   >  └── punct: ? (PUNCT)

   节点折叠得到

   > ROOT: share (VERB)
   >  ├── aux: Do (AUX)
   >  │
   >  ├── nsubj: PersonAlpha             <--- 替换为第一个导演变量
   >  │    ├── nmod:of: FilmAlpha        <--- 零碎的电影名 1 已折叠
   >  │    │
   >  │    └── conj:and: PersonBeta      
   >  │
   >  ├── cc: and (CCONJ)
   >  │
   >  ├── nsubj: [PersonBeta]              <--- 替换为第二个导演变量
   >  │    └── nmod:of: FilmBeta         <--- 零碎的电影名 2 已折叠
   >  │
   >  ├── obj: NationalityAlpha          <--- 替换为国籍变量
   >  │    ├── det: the
   >  │    └── amod: same
   >  │
   >  └── punct: ? (PUNCT)

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


