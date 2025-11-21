import { tool } from '@langchain/core/tools'
import * as z from 'zod'
import { ChatDeepSeek } from '@langchain/deepseek'
import 'dotenv/config'
import { MessagesZodMeta, StateGraph, START, END, MemorySaver ,task, entrypoint} from "@langchain/langgraph";
import { registry } from "@langchain/langgraph/zod";
import { BaseMessage } from "@langchain/core/messages";
import { SystemMessage, HumanMessage, isAIMessage } from '@langchain/core/messages'
import { v4 as uuidv4 } from 'uuid'

// Helper function to add messages to the conversation
const addMessages = (existingMessages, newMessages) => {
  return [...existingMessages, ...newMessages]
}


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


// define a task that calls the LLM and pass messages to it
const callModel = task({ name: 'callLlm' }, async (messages) => {
  return modelWithTools.invoke([
    new SystemMessage(
      'You are a helpful assistant tasked with performing arithmetic on a set of inputs.'
    ),
    ...messages,
  ])
})

// define a task that calls a tool
const callTool = task({ name: 'callTool' }, async (toolCall) => {
  const tool = toolsByName[toolCall.name]
  return tool.invoke(toolCall)
})

// ============================================
// Functional Style Agent with Memory
// ============================================

// Create agent factory function
const createAgent = (options = {}) => {
  const { systemPrompt = 'You are a helpful assistant tasked with performing arithmetic.' } = options

  // Create memory instance for this agent
  const memory = new MemorySaver()

  // Create the agent entrypoint with memory
  const agentEntrypoint = entrypoint(
    { name: 'agent', checkpointer: memory },
    async (messages) => {
      // First call the LLM
      let modelResponse = await callModel(messages)

      // Loop until no more tool calls
      while (true) {
        // Check if tool calls are needed
        if (!modelResponse.tool_calls?.length) {
          break
        }

        // Execute the tools
        const toolResults = await Promise.all(
          modelResponse.tool_calls.map((toolCall) => callTool(toolCall))
        )
        // Pass tool results back to the LLM
        messages = addMessages(messages, [modelResponse, ...toolResults])
        modelResponse = await callModel(messages)
      }

      return messages
    }
  )

  // Return agent interface
  return {
    // Create a new conversation
    createConversation: () => {
      const threadId = uuidv4()
      const config = { configurable: { thread_id: threadId } }

      return {
        threadId,
        // Send a message and get response
        send: async (message) => {
          const result = await agentEntrypoint.invoke(
            [new HumanMessage(message)],
            config
          )
          return result
        },
        // Get the last AI response content
        chat: async (message) => {
          const result = await agentEntrypoint.invoke(
            [new HumanMessage(message)],
            config
          )
          const lastMessage = result[result.length - 1]
          return lastMessage?.content || ''
        },
      }
    },
  }
}

// ============================================
// Usage Example
// ============================================

// Create an agent instance
const mathAgent = createAgent({
  systemPrompt: 'You are a helpful math assistant.',
})

// Create a new conversation with memory
const conversation = mathAgent.createConversation()

console.log(`Thread ID: ${conversation.threadId}\n`)

// Chat with the agent (functional style)
console.log('=== Message 1 ===')
const response1 = await conversation.chat('Add 3 and 4')
console.log(`Response: ${response1}\n`)

console.log('=== Message 2 ===')
const response2 = await conversation.chat('Multiply that result by 5')
console.log(`Response: ${response2}\n`)

console.log('=== Message 3 ===')
const response3 = await conversation.chat('What calculations did I ask you to do?')
console.log(`Response: ${response3}\n`)

console.log('=== Memory Demo Complete ===')