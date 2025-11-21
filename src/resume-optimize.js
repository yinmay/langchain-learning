import { ChatDeepSeek } from '@langchain/deepseek'
import 'dotenv/config'
import { StateGraph, START, END, Annotation } from '@langchain/langgraph'
import { ToolNode } from '@langchain/langgraph/prebuilt'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'

// 定义状态
const ResumeState = Annotation.Root({
  resumeText: Annotation({ reducer: (_, x) => x }),
  personalInfo: Annotation({ reducer: (_, x) => x }),
  skills: Annotation({ reducer: (_, x) => x }),
  projects: Annotation({ reducer: (_, x) => x }),
  yearsOfExperience: Annotation({ reducer: (_, x) => x }),
  skillsEvaluation: Annotation({ reducer: (_, x) => x }),
  projectsEvaluation: Annotation({ reducer: (_, x) => x }),
  suggestions: Annotation({ reducer: (_, x) => x }),
})

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Tool: 解析简历
const parseResumeTool = tool(
  async ({ resumeText }) => {
    return JSON.stringify({ parsed: true, text: resumeText })
  },
  {
    name: 'parse_resume',
    description: '解析简历文本，提取个人信息、专业技能和项目经验',
    schema: z.object({
      resumeText: z.string().describe('简历原文'),
    }),
  }
)

// Tool: 评估技能匹配度
const evaluateSkillsTool = tool(
  async ({ skills, yearsOfExperience }) => {
    return JSON.stringify({ skills, yearsOfExperience, evaluated: true })
  },
  {
    name: 'evaluate_skills',
    description: '评估专业技能的深度广度是否与工作年限匹配',
    schema: z.object({
      skills: z.string().describe('技能列表'),
      yearsOfExperience: z.number().describe('工作年限'),
    }),
  }
)

// Tool: 评估项目经验
const evaluateProjectsTool = tool(
  async ({ projects, yearsOfExperience }) => {
    return JSON.stringify({ projects, yearsOfExperience, evaluated: true })
  },
  {
    name: 'evaluate_projects',
    description: '评估项目经验的内容和难度是否与工作年限匹配',
    schema: z.object({
      projects: z.string().describe('项目经验'),
      yearsOfExperience: z.number().describe('工作年限'),
    }),
  }
)

const tools = [parseResumeTool, evaluateSkillsTool, evaluateProjectsTool]
const llmWithTools = llm.bindTools(tools)
const toolNode = new ToolNode(tools)

// 节点1: 解析简历
async function parseResumeNode(state) {
  const response = await llmWithTools.invoke([
    new SystemMessage(`你是一个专业的简历解析专家。请从简历中提取以下信息：
1. 个人信息（姓名、工作年限、联系方式等）
2. 专业技能（技术栈、工具、语言等）
3. 项目经验（项目名称、职责、技术、成果）

请调用 parse_resume 工具来处理简历。`),
    new HumanMessage(`请解析以下简历：\n\n${state.resumeText}`),
  ])

  // 如果有 tool_calls，执行它们
  if (response.tool_calls?.length > 0) {
    await toolNode.invoke({ messages: [response] })
  }

  // 让 LLM 直接提取结构化信息
  const extractResponse = await llm.invoke([
    new SystemMessage(`从简历中提取信息，直接返回 JSON 格式：
{
  "personalInfo": "个人信息摘要",
  "yearsOfExperience": 数字,
  "skills": "技能列表",
  "projects": "项目经验摘要"
}`),
    new HumanMessage(state.resumeText),
  ])

  try {
    const content = extractResponse.content
    const jsonMatch = content.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0])
      return {
        personalInfo: parsed.personalInfo || '',
        yearsOfExperience: parsed.yearsOfExperience || 0,
        skills: parsed.skills || '',
        projects: parsed.projects || '',
      }
    }
  } catch (e) {
    console.error('解析失败:', e)
  }
  return {}
}

// 节点2: 评估技能
async function evaluateSkillsNode(state) {
  const response = await llmWithTools.invoke([
    new SystemMessage(`你是技术面试专家。请评估程序员的技能是否与其工作年限匹配。

评估维度：
- 技能深度：是否有深入掌握的核心技术
- 技能广度：技术栈覆盖是否合理
- 年限匹配：技能水平是否符合工作年限预期

请调用 evaluate_skills 工具，然后给出评估结论。`),
    new HumanMessage(
      `工作年限: ${state.yearsOfExperience}年\n技能: ${state.skills}`
    ),
  ])

  if (response.tool_calls?.length > 0) {
    await toolNode.invoke({ messages: [response] })
  }

  const evalResponse = await llm.invoke([
    new SystemMessage(
      `作为技术面试专家，评估技能与工作年限的匹配度，直接给出评估结果（匹配/偏低/偏高）和具体分析。`
    ),
    new HumanMessage(
      `工作年限: ${state.yearsOfExperience}年\n技能: ${state.skills}`
    ),
  ])

  return { skillsEvaluation: evalResponse.content }
}

// 节点3: 评估项目
async function evaluateProjectsNode(state) {
  const response = await llmWithTools.invoke([
    new SystemMessage(`你是项目经验评估专家。请评估项目经验是否与工作年限匹配。

评估维度：
- 项目复杂度：是否承担过有难度的项目
- 职责范围：是否有核心职责或主导经历
- 技术深度：项目中的技术运用是否深入
- 成果量化：是否有可衡量的成果

请调用 evaluate_projects 工具，然后给出评估结论。`),
    new HumanMessage(
      `工作年限: ${state.yearsOfExperience}年\n项目经验: ${state.projects}`
    ),
  ])

  if (response.tool_calls?.length > 0) {
    await toolNode.invoke({ messages: [response] })
  }

  const evalResponse = await llm.invoke([
    new SystemMessage(
      `作为项目经验评估专家，评估项目经验与工作年限的匹配度，直接给出评估结果和具体分析。`
    ),
    new HumanMessage(
      `工作年限: ${state.yearsOfExperience}年\n项目经验: ${state.projects}`
    ),
  ])

  return { projectsEvaluation: evalResponse.content }
}

// 节点4: 生成建议
async function generateSuggestionsNode(state) {
  const response = await llm.invoke([
    new SystemMessage(`你是资深简历顾问。根据简历评估结果，给出具体、可操作的优化建议。

建议要包含：
1. 整体评价
2. 技能部分改进建议
3. 项目经验改进建议
4. 简历呈现方式建议`),
    new HumanMessage(`
个人信息: ${state.personalInfo}
工作年限: ${state.yearsOfExperience}年

技能评估结果:
${state.skillsEvaluation}

项目评估结果:
${state.projectsEvaluation}

请给出综合优化建议。`),
  ])

  return { suggestions: response.content }
}

// 构建工作流
const workflow = new StateGraph(ResumeState)
  .addNode('parseResume', parseResumeNode)
  .addNode('evaluateSkills', evaluateSkillsNode)
  .addNode('evaluateProjects', evaluateProjectsNode)
  .addNode('generateSuggestions', generateSuggestionsNode)
  .addEdge(START, 'parseResume')
  .addEdge('parseResume', 'evaluateSkills')
  .addEdge('evaluateSkills', 'evaluateProjects')
  .addEdge('evaluateProjects', 'generateSuggestions')
  .addEdge('generateSuggestions', END)

const app = workflow.compile()

// 测试用例
const testResume = `
张三
前端开发工程师 | 3年经验
邮箱: zhangsan@example.com

专业技能：
- 熟练掌握 HTML、CSS、JavaScript
- 熟悉 Vue.js 框架
- 了解 Node.js
- 使用过 Git

项目经验：
1. 公司官网开发
   - 负责页面切图和样式编写
   - 使用 jQuery 实现交互效果

2. 后台管理系统
   - 参与部分页面开发
   - 使用 Vue.js 开发表单组件
`

async function main() {
  console.log('=== 简历优化 Agent ===\n')
  console.log('输入简历:\n', testResume)
  console.log('\n正在分析...\n')

  const result = await app.invoke({ resumeText: testResume })

  console.log('=== 优化建议 ===\n')
  console.log(result.suggestions)
}

main().catch(console.error)
