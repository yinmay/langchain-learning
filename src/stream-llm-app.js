import { ChatDeepSeek } from '@langchain/deepseek'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

const messages = [
  new SystemMessage('You are a helpful assistant.'),
  new HumanMessage('Write a short poem about artificial intelligence.'),
]

console.log('Streaming response:\n')

// Use stream method for streaming output
const stream = await llm.stream(messages)

for await (const chunk of stream) {
  process.stdout.write(chunk.content)
}

console.log('\n\nStream completed!')

// Define template
const systemTemplate = 'Translate the following from English into {language}'
const promptTemplate = ChatPromptTemplate.fromMessages([
  ['system', systemTemplate],
  ['user', '{text}'],
])

// Generate prompt value from template
const promptValue = await promptTemplate.invoke({
  language: 'french',
  text: 'hi, how are you?',
})

// Invoke LLM with prompt to generate AI result
const res = await llm.invoke(promptValue)
console.log(`${res.content}`)
