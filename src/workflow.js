import { ChatDeepSeek } from '@langchain/deepseek'
import 'dotenv/config'
import { trimMessages } from '@langchain/core/messages'

import {
  START,
  END,
  MessagesAnnotation,
  StateGraph,
  MemorySaver,
} from '@langchain/langgraph'
import { v4 as uuidv4 } from 'uuid'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Configure trimmer to keep the latest 10 messages
const trimmer = trimMessages({
  maxTokens: 10,
  strategy: 'last',
  tokenCounter: (msgs) => msgs.length,
  includeSystem: true,
  allowPartial: false,
  startOn: 'human',
})

// Define the function that calls the model
const callModel = async (state) => {
  console.log('Input messages length (before trim): ', state.messages.length)

  // Trim messages to keep only the latest 10
  const trimmedMessages = await trimmer.invoke(state.messages)

  console.log('Input messages length (after trim): ', trimmedMessages.length)

  const response = await llm.invoke(trimmedMessages)
  return { messages: response }
}

// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
  // Define the node and edge
  .addNode('model', callModel)
  .addEdge(START, 'model')
  .addEdge('model', END)

  // Add memory
const memory = new MemorySaver()
const app = workflow.compile({ checkpointer: memory })


const config = { configurable: { thread_id: uuidv4() } };

const input1 = [
  {
    role: 'user',
    content: 'Hi, I am melissa',
  },
]
const output1 = await app.invoke({ messages: input1 }, config)

// The output contains all messages in the state.
// This will log the last message in the conversation.
console.log('output1 ', output1.messages[output1.messages.length - 1])


const input2 = [
  {
    role: 'user',
    content: 'Who am I?',
  },
]
const output2 = await app.invoke({ messages: input2 }, config)

// The output contains all messages in the state.
// This will log the last message in the conversation.
console.log('output2 ', output2.messages[output2.messages.length - 1])

// Test with casual messages to demonstrate trimming
console.log('\n--- Testing message trimming with casual conversation ---')

const casualMessages = [
  "What's the weather like?",
  "Tell me a joke",
  "How do I make coffee?",
  "What's your favorite color?",
  "Can you help me with math?",
  "What time is it?",
  "Recommend a book",
  "Explain quantum physics briefly",
  "What should I eat for dinner?",
  "How's your day going?",
  "What's 2 + 2?",
  "Tell me about space",
  "What's the capital of Japan?",
  "How do I learn programming?",
  "What's artificial intelligence?",
]

for (let i = 0; i < casualMessages.length; i++) {
  const input = [
    {
      role: 'user',
      content: casualMessages[i],
    },
  ]
  const output = await app.invoke({ messages: input }, config)
  console.log(`\nMessage ${i + 3}: "${casualMessages[i]}"`)
  console.log(`Total messages in state = ${output.messages.length}`)
}

console.log('\n--- Trimming in action ---')
console.log('Notice: After message 12, the total stays at 10 (trimmer removes old messages)')
console.log('The trimmer keeps only the latest 10 messages to manage memory efficiently')
