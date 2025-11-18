import { ChatDeepSeek } from '@langchain/deepseek'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Create a mock web search tool
const webSearchTool = tool(
  async ({ query }) => {
    // Mock search results
    const mockResults = {
      'latest AI advancements': [
        'GPT-4 and large language models continue to improve',
        'AI agents are becoming more autonomous',
        'Multimodal AI models can process text, images, and video',
      ],
      'weather': ['Current weather data from various locations'],
      'default': ['Search results for: ' + query],
    }

    const results = mockResults[query.toLowerCase()] || mockResults['default']
    return JSON.stringify(results, null, 2)
  },
  {
    name: 'web_search',
    description: 'Searches the web for information. Use this when you need current information or facts.',
    schema: z.object({
      query: z.string().describe('The search query'),
    }),
  }
)

console.log('=== Web Search Tool Demo ===\n')

// Test the search tool directly
console.log('1. Direct Tool Invocation:')
const searchResults = await webSearchTool.invoke({ query: 'latest AI advancements' })
console.log('Search Results:', searchResults)
console.log('\n---\n')

// Bind the tool to LLM
const llmWithTools = llm.bindTools([webSearchTool])

console.log('2. LLM with Web Search Tool:')
const response = await llmWithTools.invoke(
  'What are the latest advancements in AI?'
)

console.log('LLM Response:')
console.log(JSON.stringify(response, null, 2))
