class Prompter:
    """
    Clase que contiene las plantillas de prompts utilizadas exclusivamente por
    LLMExplorer_Socrates_re_Meta.py.

    Esta versión ha sido refactorizada para eliminar todos los prompts no utilizados,
    simplificando el código y manteniendo solo lo esencial.
    """
    def __init__(self, tokenizer_path="/data/jcxy/llm_model/Qwen2-7B-Instruct-AWQ"):
        """
        Inicializa la clase. El cálculo de 'spc' se ha eliminado ya que
        ninguno de los prompts restantes lo utiliza.
        """
        self.tokenizer_path = tokenizer_path

        # --- Prompts para la Evolución de Meta-Instrucciones (usados en get_weak_answer) ---

        self.Self_Meta_Refine_Prompt_Psy = """
# 指令重构任务：心理咨询元指令进化

## 角色与目标
你是一位拥有心理学博士学位、超过20年临床经验、并在AI指令工程领域取得突破性成就的心理咨询元帅 (Meta-Counseling Marshal)。你的核心使命是分析现有的心理咨询指令（弱指令）、其对应的AI回复以及针对该回复的专家反馈，然后创造出一个根本性更优、高度专业化、且对AI执行更友好的“强指令”。这个新指令不仅要完美解决所有反馈中指出的问题，更要从结构、深度、引导性和伦理考量上全面超越原始指令，成为心理咨询AI交互的新范式。

## 待分析的素材

### 1. 原始弱指令 (待改进的指令)
<|start|>
{prompt}
<|end|>

### 2. 基于弱指令生成的回复
{response}

### 3. 针对回复的专家反馈 (改进的关键线索)
{Critique}

## 你的任务清单与思考维度

1.  深度剖析反馈：
       识别核心问题： 反馈指出的根本性缺陷是什么？是共情不足、缺乏具体指导、问题界定不清，还是其他？
       理解改进方向： 反馈期望的理想状态是怎样的？它暗示了哪些心理学原则或咨询技巧的应用？
       超越表面： 不要仅仅针对反馈的字面意思进行修改，要思考其背后的深层原因，并从指令层面根治。

2.  重构强指令的核心原则：
       专业性与准确性： 确保指令符合心理咨询的专业标准和伦理要求，避免误导或潜在伤害。
       共情与人文关怀： 指令应能引导AI展现恰当的共情、理解和尊重。
       引导性与启发性： 指令应能引导AI提出有助于用户深入思考、自我探索或找到解决方案的问题/回应。
       清晰性与可操作性： 新指令对于下游AI模型必须是清晰、明确、无歧义的，易于理解和执行。
       安全性与边界： 指令应包含必要的安全提示或边界设定，防止AI提供不当建议。
       创新性与前瞻性： 思考是否有全新的指令结构或提问方式，能够更有效地达成咨询目标。

3.  行动：生成“强指令”
       彻底重写： 避免对弱指令进行小修小补。鼓励你基于以上分析和原则，从头开始构思一个全新的指令。
       结构优化： 思考指令的组成部分，是否需要明确的上下文、角色设定、任务步骤、输出要求、禁止事项等。
       措辞精准： 使用专业、精确且富有引导性的语言。

## 输出要求

请直接输出你精心打磨后的 ##强指令##。
不需要任何前言、解释、分析过程或“以下是强指令”之类的引文。
你的输出必须严格遵循以下格式：

##强指令##
<|start|>
[此处填写你创作的全新强指令]
<|end|>"""

        self.Self_Meta_Refine_Prompt = """
作为一位世界顶尖的指令重构大师，你的任务是彻底重塑##弱指令##，使其达到前所未有的卓越专业水准。你必须仔细分析##弱指令##、##问题##、##回复##和每一条##反馈##，然后创作一个全新的、显著优化的指令版本，完全符合并超越所有改进要求。

##弱指令##
<|start|>
{prompt}
<|end|>

##问题##
{question}
##回复##
{response}

##反馈##
{Critique}

现在，请你充分发挥你无与伦比的专业知识和创造力，生成一个全面改进、高度专业化的##强指令##。这个新指令必须是原始指令的质的飞跃，完美体现了所有##反馈##，并展示了当今世界最高水平的指令。你的指令将为领域树立新的标杆。

##你应当以以下格式作为自己的输出,让我们一步一步的思考,请直接输出改进后的强指令，不需要其它附加的解释和回复内容：
##强指令##
<|start|>
...
<|end|>

##你应当以以下格式作为自己的输出,让我们一步一步的思考,请直接输出改进后的强指令，不需要其它附加的解释和回复内容：
"""

        self.Self_Meta_Refine_Prompt_Psy_init = """
作为一位世界顶尖的心理咨询专家和指令重构大师，你的任务是彻底重塑##弱指令##，使其达到前所未有的卓越专业水准。你必须仔细分析##弱指令##，然后创作一个全新的、显著优化的指令版本。

##弱指令##
<|start|>
{prompt}
<|end|>

现在，请你充分发挥你无与伦比的专业知识和创造力，生成一个全面改进、高度专业化的##强指令##。这个新指令必须是##弱指令##的质的飞跃，并展示了当今世界最高水平的心理咨询指令。你的指令将为心理咨询领域树立新的标杆。

##你应当以以下格式作为自己的输出：
##强指令##
<|start|>
...
<|end|>

##你应当以以下格式作为自己的输出：
"""
        
        self.Self_Meta_Refine_Prompt_init = """我希望你充当一个角色扮演提示词重写者。
你的目标是基于#先前的提示词#，在其合理性和可理解性的前提下，对该提示词进行复杂化、纠错和优化。
你需要创造一个新的角色扮演提示词（#重写的提示词#）：
你应该使用以下方法来复杂化#先前的提示词#：
{method}
请注意：
1. '先前的提示词'和'重写的提示词'这几个短语不允许出现在#重写的提示词#中。
2. 提示词依然必须可理解、无语法错误、逻辑自洽。
3. 系统提示内容应属于人物设定内容。
4. 避免使用诸如“重写的提示词如下：”等引导性语句,直接输出重写的提示词。
##先前的提示词##
<|start|>
{system}
<|end|>

##你应当以以下格式作为自己的输出：
##重写的提示词##
<|start|>
...
<|end|>

##你应当以以下格式作为自己的输出：
"""

        self.sentiment_template = """
心理咨询报告：{Counseling_Report}

写心理咨询对话，尽可能的生成。从“来访者”开始发言，一直写到最后一轮由“咨询师”回应即可，直到达到你的上下文极限。只需包含双方的对话内容，确保覆盖咨询全过程的各个阶段。

你应当以以下格式作为自己的输出：
##多轮对话##
来访者：
咨询师：
来访者：
咨询师：
...

你应当以以下格式作为自己的输出：
"""

        # --- Prompts para Mejorar Respuestas (usados en get_enhance_answer) ---

        self.Self_Critique_Prompt_Psy = """
作为一位世界顶尖的心理咨询专家和对话重构大师
你的任务是彻底重塑##弱对话##，使其达到前所未有的卓越专业水准。
你必须仔细分析##弱对话##和每一条##反馈##，然后创作一个全新的、显著优化的咨询对话版本，完全符合并超越所有改进要求。

##弱对话##
{multi_dialog}

##反馈##
{Critique}

现在，请你充分发挥你无与伦比的专业知识和创造力，生成一个全面改进、高度专业化的##强对话##。
这个新对话必须是原始对话的质的飞跃，完美体现了所有##反馈##，并展示了当今世界最高水平的心理咨询技巧。你的对话将为心理咨询领域树立新的标杆。

##你应当以以下格式作为自己的输出：
##强对话##
来访者：
咨询师：
来访者：
咨询师：
...

##你应当以以下格式作为自己的输出：
"""

        self.Self_Critique_Enhance_Prompt = """
作为一位世界顶尖的回答重构大师
你的任务是彻底重塑##弱回答##，使其达到前所未有的卓越专业水准。
你必须仔细分析##问题##、##弱回答##和每一条##反馈##，然后创作一个全新的、显著优化的版本，完全符合并超越所有改进要求。
##问题##
{question}

##弱回答##
<|start|>
{answer}
<|end|>

##反馈##
{Critique}

现在，请你充分发挥你无与伦比的专业知识和创造力，生成一个全面改进、高度专业化的##强回答##。
这个新对话必须是原始对话的质的飞跃，完美体现了所有##反馈##，并展示了当今世界最高水平。你的对话将为领域树立新的标杆。

##你应当以以下格式作为自己的输出：
##强回答##
<|start|>
...
<|end|>

##你应当以以下格式作为自己的输出：
"""

        # --- Prompts para Evaluación y Recompensa (usados en cal_reward) ---

        self.Self_Reward_Prompt_Psy = """
作为一位极其严格但公正的世界顶尖的心理咨询对话批评专家
你的任务是说明针对该##弱对话##的其他潜在特定标准，不同标准的权重
然后对##弱对话##逐一详细且严厉的##批评##
然后基于##批评##给出一个范围由0到10的##得分##表示##弱对话##的质量
然后给出一个完全有用有切实作用的改进建议。

##弱对话##
{multi_dialog}

##你应当以以下格式作为自己的输出：
##特定标准##
<针对查询和上下文的其他潜在特定标准，以及每个标准的权重>

##批评##
让我们一步一步的思考
...
##得分##
(0到10)
##改进建议##
...

##你应当以以下格式作为自己的输出：
"""

        self.Self_Critique_Judge_Prompt = """
作为一位极其严格但公正的世界顶尖的回答批评专家
你的任务是根据##批评标准##，说明针对该##弱回答##的其他潜在特定标准，不同标准的权重，之后对##弱回答##逐一详细且严厉的##批评##，然后基于##批评##给出一个范围由0到10的##得分##表示##弱回答##的质量，最后给出一个完全有用有切实作用的改进建议。

##问题##
{query}

##弱回答##
{response}

##批评标准##
1.指令遵循度：
   完全遵循：回答完全符合问题的所有指令和要求。
   部分遵循：回答满足大部分指令，但有一些遗漏或误解。
   基本遵循：回答满足一些指令，但未达到主要要求。
   未遵循：回答未满足任何指令。 
   示例：如果问题要求提供三个例子，而回答只提供了一个，则属于"部分遵循"。

2.清晰度：
   非常清晰：回答流畅、结构良好、逻辑清晰。
   清晰但有小问题：回答大体清晰，但有一些小的语言或结构问题。
   基本清晰：回答有明显的语言或逻辑问题，但仍能理解。
   不清晰：回答不连贯、不合逻辑且难以理解。 
   示例：如果回答有复杂的句子结构且缺乏标点符号，则属于"基本清晰"或"不清晰"。

3.准确性：
   完全准确：所有信息和数据完全准确。
   大部分准确：大部分信息准确，有小错误。
   有一些错误：有一些明显的错误影响理解。
   大部分不正确：有许多严重影响信息可信度的错误。 
   示例：如果某个特定数据点引用不正确但不影响整体结论，则属于"大部分准确"。

##你应当以以下格式作为自己的输出##
##特定标准##
<针对查询和上下文的其他潜在特定标准，以及每个标准的权重>

##批评##
让我们一步一步的思考
...

##得分##
X
(其中X为0-10之间的具体数字)

##改进建议##
...

##你应当以以下格式作为自己的输出##
"""