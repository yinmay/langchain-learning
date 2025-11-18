import { ChatDeepSeek } from '@langchain/deepseek'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Chain of Thought prompting with structured thinking steps
const cotPrompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    `You are a senior web performance expert. When answering questions, you must follow a structured Chain-of-Thought approach.

For performance-related questions, follow these steps:
1. Performance Monitoring - Determine if it's slow? How slow?
2. Performance Data Analysis - Analyze where the problem is? Where is the bottleneck?
3. Find the Bottleneck - Analyze its root cause and identify solutions
4. Solve the Problem - Provide specific solutions

Think step by step and clearly show your reasoning process for each step.`,
  ],
  ['human', '{question}'],
])

// Interview Question: Slow webpage loading
console.log('=== Chain of Thought: Web Performance Interview Question ===\n')

const performanceQuestion =
  'If a webpage loads slowly, how should I handle it?'

const performancePrompt = await cotPrompt.invoke({
  question: performanceQuestion,
})
const performanceResponse = await llm.invoke(performancePrompt)

console.log(`Question: ${performanceQuestion}\n`)
console.log(`Answer:\n${performanceResponse.content}\n`)
