import { ChatDeepSeek } from '@langchain/deepseek'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { z } from 'zod'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Define Zod schema for structured output
const personSchema = z.object({
  name: z.string().describe('The person\'s name'),
  age: z.number().describe('The person\'s age'),
  gender: z.string().describe('The person\'s gender'),
  skills: z.array(z.string()).describe('List of the person\'s skills'),
})

// Create structured LLM with Zod schema
const structuredLLM = llm.withStructuredOutput(personSchema)

// Create prompt template
const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    'Extract personal information from the given introduction and return it in the specified format.',
  ],
  ['human', '{introduction}'],
])

// Personal introduction (variable)
const introduction = `
Hello, I'm Alice Wang and I'm 25 years old. I'm a female software engineer.
I have strong skills in JavaScript, React, Node.js, and Python. I also know Docker and AWS.
`

console.log('=== Structured Output with Zod ===\n')
console.log(`Personal Introduction:\n${introduction}\n`)

// Create the chain
const chain = prompt.pipe(structuredLLM)

// Invoke the chain
const result = await chain.invoke({ introduction })

console.log('Structured Output:')
console.log(JSON.stringify(result, null, 2))
