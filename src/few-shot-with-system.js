import { ChatDeepSeek } from '@langchain/deepseek'
import { PromptTemplate } from '@langchain/core/prompts'
import { FewShotPromptTemplate } from '@langchain/core/prompts'
import { SystemMessage, HumanMessage } from '@langchain/core/messages'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Language configuration - change this to 'french', 'chinese', 'spanish', etc.
const language = 'french'

const examples = [
  {
    question: `
      Write a JSDoc comment for the following JS function
      function add(a, b) {
        return a + b;
      }`,
    answer: `
      /**
      * Adds two numbers together
      * @param {number} a - The first number
      * @param {number} b - The second number
      * @returns {number} The sum of the two numbers
      */`,
  },
  {
    question: `
      Write a JSDoc comment for the following JS function
      function getUser(id) {
        return db.findUserById(id);
      }
    `,
    answer: `
      /**
      * Retrieves user information from the database by user ID
      * @param {string} id - The unique user ID
      * @returns {Object|null} Returns the user object, or null if not found
      */`,
  },
]

// Helper to escape single curly braces so the prompt templating (f-string)
// won't attempt to interpolate JS code blocks that contain `{` and `}`.
const escapeCurlyBraces = (s) => s.replace(/\{/g, '{{').replace(/\}/g, '}}')

// Use escaped examples to avoid Missing value for input errors when the
// example text contains curly braces (common in code snippets).
const escapedExamples = examples.map((ex) => ({
  ...ex,
  question: escapeCurlyBraces(ex.question),
  answer: escapeCurlyBraces(ex.answer),
}))

const examplePrompt = PromptTemplate.fromTemplate(
  'Question: {question}\nAnswer: {answer}'
)

const prompt = new FewShotPromptTemplate({
  examples: escapedExamples,
  examplePrompt,
  suffix: 'Question: {input}',
  inputVariables: ['input'],
})

const formatted = await prompt.format({
  input: `
  Write a JSDoc comment in ${language} for the following JS function
  function formatDate(date) {
    return date.toISOString().split('T')[0];
  }`,
})

console.log('Formatted prompt:\n')
console.log(formatted)
console.log('\n---\n')

// Define system prompt without language specification
const systemPrompt = `You are a senior Node.js engineer. Please write documentation comments for the given function.
Format requirements:
1. Use JSDoc style.
2. Each parameter must have a description.
3. Must include a return value description at the end.`

// Create messages with system prompt and formatted examples
const messages = [
  new SystemMessage(systemPrompt),
  new HumanMessage(formatted),
]

console.log('AI Response:\n')

// Invoke LLM with system prompt and formatted prompt
const response = await llm.invoke(messages)
console.log(response.content)
