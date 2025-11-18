import { ChatDeepSeek } from '@langchain/deepseek'
import { PromptTemplate } from '@langchain/core/prompts'
import { FewShotPromptTemplate } from '@langchain/core/prompts'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

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
  Write a JSDoc comment for the following JS function
  function formatDate(date) {
    return date.toISOString().split('T')[0];
  }`,
})

console.log('Formatted prompt:\n')
console.log(formatted)
console.log('\n---\n')
console.log('AI Response:\n')

// Invoke LLM with the formatted prompt
const response = await llm.invoke(formatted)
console.log(response.content)