import { ChatDeepSeek } from '@langchain/deepseek'
import 'dotenv/config'
import { createAgent, tool } from 'langchain'
import * as z from 'zod'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Tool 1: Weather tool
const getWeather = tool((input) => `It's always sunny in ${input.city}!`, {
  name: 'get_weather',
  description: 'Get the weather for a given city',
  schema: z.object({
    city: z.string().describe('The city to get the weather for'),
  }),
})

// Tool 2: Tavily search tool
const tavilySearch = tool(
  async (input) => {
    // Use Tavily API to search
    const apiKey = process.env.TAVILY_API_KEY

    if (!apiKey) {
      return 'Tavily API key not found. Please set TAVILY_API_KEY in your .env file'
    }

    try {
      const response = await fetch('https://api.tavily.com/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          api_key: apiKey,
          query: input.query,
          max_results: 3,
        }),
      })

      const data = await response.json()

      if (data.results && data.results.length > 0) {
        // Format the results
        const formattedResults = data.results
          .map(
            (result, index) =>
              `${index + 1}. ${result.title}\n   ${result.content}\n   URL: ${result.url}`
          )
          .join('\n\n')
        return formattedResults
      } else {
        return 'No results found'
      }
    } catch (error) {
      return `Error searching: ${error.message}`
    }
  },
  {
    name: 'tavily_search',
    description: 'Search the web for current information using Tavily search engine. Use this when you need up-to-date information or facts.',
    schema: z.object({
      query: z.string().describe('The search query'),
    }),
  }
)

const agent = createAgent({
  model: llm,
  tools: [getWeather, tavilySearch],
})

// Test 1: Weather query
console.log('=== Test 1: Weather Query ===\n')
const res1 = await agent.invoke({
  messages: [{ role: 'user', content: "What's the weather in Tokyo?" }],
})
console.log(res1)

// Test 2: Web search query
console.log('\n\n=== Test 2: Web Search Query ===\n')
const res2 = await agent.invoke({
  messages: [{ role: 'user', content: 'What are the latest developments in artificial intelligence in 2025?' }],
})
console.log(res2)