import { ChatDeepSeek } from '@langchain/deepseek'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { z } from 'zod'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Define Zod schema for sentiment analysis output
const sentimentSchema = z.object({
  sentiment: z
    .enum(['positive', 'negative', 'neutral'])
    .describe('The sentiment of the sentence: positive (褒义), negative (贬义), or neutral (中性)'),
  confidence: z
    .number()
    .min(0)
    .max(1)
    .describe('Confidence level of the sentiment analysis (0-1)'),
  reason: z.string().describe('Brief explanation for the sentiment classification'),
})

// Create structured LLM with Zod schema
const structuredLLM = llm.withStructuredOutput(sentimentSchema)

// Create prompt template
const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    'You are a sentiment analysis expert. Analyze the given sentence and classify it as positive (褒义), negative (贬义), or neutral (中性). Provide your confidence level and reasoning.',
  ],
  ['human', '{sentence}'],
])

// Create the chain
const chain = prompt.pipe(structuredLLM)

// Test sentences
const testSentences = [
  'This product is absolutely amazing and exceeded all my expectations!',
  'The service was terrible and the staff was very rude.',
  'The meeting is scheduled for 3 PM tomorrow.',
  '你真是太聪明了！',
  '这个电影简直是浪费时间。',
  '今天天气不错。',
]

console.log('=== Sentiment Analysis with AI ===\n')

// Analyze each sentence
for (const sentence of testSentences) {
  const result = await chain.invoke({ sentence })

  console.log(`Sentence: "${sentence}"`)
  console.log(`Sentiment: ${result.sentiment}`)
  console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`)
  console.log(`Reason: ${result.reason}`)
  console.log('---\n')
}
