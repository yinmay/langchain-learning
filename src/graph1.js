import { tool } from '@langchain/core/tools'
import * as z from 'zod'
import { ChatDeepSeek } from '@langchain/deepseek'
import 'dotenv/config'
import { MessagesZodMeta, StateGraph, START, END, MemorySaver } from "@langchain/langgraph";
import { registry } from "@langchain/langgraph/zod";
import { BaseMessage } from "@langchain/core/messages";
import { SystemMessage, HumanMessage } from '@langchain/core/messages'
import { isAIMessage } from '@langchain/core/messages'
import { v4 as uuidv4 } from 'uuid'


const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Define tools
const add = tool(({ a, b }) => a + b, {
  name: 'add',
  description: 'Add two numbers',
  schema: z.object({
    a: z.number().describe('First number'),
    b: z.number().describe('Second number'),
  }),
})

const multiply = tool(({ a, b }) => a * b, {
  name: 'multiply',
  description: 'Multiply two numbers',
  schema: z.object({
    a: z.number().describe('First number'),
    b: z.number().describe('Second number'),
  }),
})

const divide = tool(({ a, b }) => a / b, {
  name: 'divide',
  description: 'Divide two numbers',
  schema: z.object({
    a: z.number().describe('First number'),
    b: z.number().describe('Second number'),
  }),
})

// Augment the LLM with tools
const toolsByName = {
  [add.name]: add,
  [multiply.name]: multiply,
  [divide.name]: divide,
}
const tools = Object.values(toolsByName)
const modelWithTools = llm.bindTools(tools)


// Define the state structure
const MessagesState = z.object({
  messages: z
    .array(z.custom((val) => val instanceof BaseMessage || typeof val === 'object'))
    .register(registry, MessagesZodMeta),
  llmCalls: z.number().optional(),
});

// add llmCall node
async function llmCall(state) {
  const newMessages = await modelWithTools.invoke([
    new SystemMessage(
      'You are a helpful assistant tasked with performing arithmetic on a set of inputs.'
    ),
    ...state.messages,
  ])
 
  const newLlmCalls = (state.llmCalls ?? 0) + 1
  return {
    messages: newMessages,
    llmCalls: newLlmCalls,
  }
}

// add toolNode node
// custom work flow and have to invoke tool manually
async function toolNode(state) {
  const lastMessage = state.messages.at(-1)

  if (lastMessage == null || !isAIMessage(lastMessage)) {
    return { messages: [] }
  }

  const result = []
  for (const toolCall of lastMessage.tool_calls ?? []) {
    const tool = toolsByName[toolCall.name]
    const observation = await tool.invoke(toolCall)
    result.push(observation)
  }

  return { messages: result }
}

// add shouldContinue condition
async function shouldContinue(state) {
  const lastMessage = state.messages.at(-1)
  if (lastMessage == null || !isAIMessage(lastMessage)) return END

  // If the LLM makes a tool call, then perform an action
  if (lastMessage.tool_calls?.length) {
    return 'toolNode'
  }

  // Otherwise, we stop (reply to the user)
  return END
}

// add workflow
const graph = new StateGraph(MessagesState)
  .addNode('llmCall', llmCall)
  .addNode('toolNode', toolNode)
  .addEdge(START, 'llmCall')
  .addConditionalEdges('llmCall', shouldContinue, ['toolNode', END])
  .addEdge('toolNode', 'llmCall')


// Add memory for conversation history
const memory = new MemorySaver()

// Compile with memory checkpointer
const agent = graph.compile({ checkpointer: memory })

// Config with thread_id to track conversation
const config = { configurable: { thread_id: uuidv4() } }

// Invoke - First message
console.log('=== Message 1: Add 3 and 4 ===\n')
const result1 = await agent.invoke(
  { messages: [new HumanMessage('Add 3 and 4.')] },
  config
)
for (const message of result1.messages) {
  console.log(`[${message.getType()}]: ${message.content}`)
}

// Invoke - Second message (uses memory from previous conversation)
console.log('\n=== Message 2: Multiply that result by 5 ===\n')
const result2 = await agent.invoke(
  { messages: [new HumanMessage('Now multiply that result by 5.')] },
  config
)
for (const message of result2.messages) {
  console.log(`[${message.getType()}]: ${message.content}`)
}

// Invoke - Third message (still remembers context)
console.log('\n=== Message 3: What was my first question? ===\n')
const result3 = await agent.invoke(
  { messages: [new HumanMessage('What was the first calculation I asked you to do?')] },
  config
)
for (const message of result3.messages) {
  console.log(`[${message.getType()}]: ${message.content}`)
}

console.log('\n=== Memory Demo Complete ===')