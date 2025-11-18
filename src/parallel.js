import { ChatDeepSeek } from '@langchain/deepseek'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { RunnableParallel, RunnablePassthrough } from '@langchain/core/runnables'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

console.log('=== LangChain Parallel Chains Demo ===\n')

// Example 1: Simple Parallel Execution
console.log('1. Parallel Chain Execution:\n')

const jokePrompt = ChatPromptTemplate.fromTemplate('Tell me a joke about {topic}')
const poemPrompt = ChatPromptTemplate.fromTemplate('Write a short poem about {topic}')
const factPrompt = ChatPromptTemplate.fromTemplate('Tell me an interesting fact about {topic}')

const jokeChain = jokePrompt.pipe(llm).pipe(new StringOutputParser())
const poemChain = poemPrompt.pipe(llm).pipe(new StringOutputParser())
const factChain = factPrompt.pipe(llm).pipe(new StringOutputParser())

// Run all chains in parallel
const parallelChain = RunnableParallel.from({
  joke: jokeChain,
  poem: poemChain,
  fact: factChain,
})

const topic = 'cats'
console.log(`Topic: ${topic}\n`)

const results = await parallelChain.invoke({ topic })

console.log('Joke:', results.joke)
console.log('\n---\n')
console.log('Poem:', results.poem)
console.log('\n---\n')
console.log('Fact:', results.fact)
console.log('\n---\n')

// Example 2: Parallel with Passthrough
console.log('2. Parallel with Passthrough:\n')

const translationPrompt = ChatPromptTemplate.fromTemplate(
  'Translate to {language}: {text}'
)

const translationChain = translationPrompt.pipe(llm).pipe(new StringOutputParser())

// Create parallel chain that preserves original input and adds translations
const multiTranslationChain = RunnableParallel.from({
  original: new RunnablePassthrough(),
  french: translationChain,
  spanish: translationChain,
})

const text = 'Hello, how are you today?'

const frenchResult = await multiTranslationChain.invoke({
  text,
  language: 'French',
})

console.log('Original:', frenchResult.original)
console.log('French:', frenchResult.french)

const spanishResult = await translationChain.invoke({
  text,
  language: 'Spanish',
})
console.log('Spanish:', spanishResult)
console.log('\n---\n')

// Example 3: Complex Parallel Workflow
console.log('3. Complex Parallel Workflow:\n')

const summaryPrompt = ChatPromptTemplate.fromTemplate(
  'Summarize this text in one sentence: {text}'
)
const sentimentPrompt = ChatPromptTemplate.fromTemplate(
  'What is the sentiment of this text (positive/negative/neutral)? {text}'
)
const keywordsPrompt = ChatPromptTemplate.fromTemplate(
  'Extract 3 keywords from this text: {text}'
)

const analysisChain = RunnableParallel.from({
  summary: summaryPrompt.pipe(llm).pipe(new StringOutputParser()),
  sentiment: sentimentPrompt.pipe(llm).pipe(new StringOutputParser()),
  keywords: keywordsPrompt.pipe(llm).pipe(new StringOutputParser()),
})

const article = `
Artificial intelligence is transforming the way we work and live.
From healthcare to finance, AI is helping professionals make better decisions.
The future of AI looks bright with new breakthroughs happening every day.
`

console.log('Article:', article)

const analysis = await analysisChain.invoke({ text: article })

console.log('\nAnalysis Results:')
console.log('Summary:', analysis.summary)
console.log('Sentiment:', analysis.sentiment)
console.log('Keywords:', analysis.keywords)
