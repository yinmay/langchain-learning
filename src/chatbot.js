import { ChatDeepSeek } from '@langchain/deepseek'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

const res1 = await llm.invoke([{ role: 'user', content: 'Hello, I am melissa' }])
console.log('res1 ', res1)


const res2 = await llm.invoke([{ role: 'user', content: 'Who am I?' }])
console.log('res2 ', res2)

const res3 = await llm.invoke([
  { role: 'user', content: 'Hello, I am melissa' },
  { role: 'assistant', content: 'Hello, Melissa! What can I help you with today?' },
  { role: 'user', content: 'Who am I?' },
])
console.log('res3 ', res3)


