import { ChatDeepSeek } from '@langchain/deepseek'
import { ToolNode } from '@langchain/langgraph/prebuilt'
import { tools } from './tools.js'

// LLM instance
export const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// LLM with tools bound
export const llmWithTools = llm.bindTools(tools)

// Tool execution node
export const toolNode = new ToolNode(tools)
