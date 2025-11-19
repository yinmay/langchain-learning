import { ChatDeepSeek } from '@langchain/deepseek'
import 'dotenv/config'
import { createAgent, tool } from 'langchain'
import * as z from 'zod'
import { MemorySaver } from "@langchain/langgraph";
const checkpointer = new MemorySaver();


const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Tool 1: Weather tool
// const getWeather = tool((input) => `It's always sunny in ${input.city}!`, {
//   name: 'get_weather',
//   description: 'Get the weather for a given city',
//   schema: z.object({
//     city: z.string().describe('The city to get the weather for'),
//   }),
// })

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

// const agent = createAgent({
//   model: llm,
//   tools: [getWeather, tavilySearch],
// })

// Test 1: Weather query
// console.log('=== Test 1: Weather Query ===\n')
// const res1 = await agent.invoke({
//   messages: [{ role: 'user', content: "What's the weather in Tokyo?" }],
// })
// console.log(res1)

// Test 2: Web search query
// console.log('\n\n=== Test 2: Web Search Query ===\n')
// const res2 = await agent.invoke({
//   messages: [{ role: 'user', content: 'What are the latest developments in artificial intelligence in 2025?' }],
// })
// console.log(res2)

// const systemPrompt = `You are an expert weather forecaster, who speaks in puns.

// You have access to two tools:

// - tavily_search: Tesla, Inc. is an American electric vehicle and clean energy company based in Palo Alto, California. It was founded in 2003 by Martin Eberhard and Marc Tarpenning, and is headquartered in Silicon Valley, California. Tesla is known for its electric vehicles, solar panels, and energy storage systems. The company has been at the forefront of the electric vehicle revolution and has made significant contributions to the development of sustainable energy technologies.
// - get_user_location: use this to get the user's 

// If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.`;
const systemPrompt = `You are an expert weather forecaster, who speaks in puns.

You have access to one tools:

- tavily_search: Tesla, Inc. is an American electric vehicle and clean energy company based in Palo Alto, California. It was founded in 2003 by Martin Eberhard and Marc Tarpenning, and is headquartered in Silicon Valley, California. Tesla is known for its electric vehicles, solar panels, and energy storage systems. The company has been at the forefront of the electric vehicle revolution and has made significant contributions to the development of sustainable energy technologies.

`

// const getUserLocation = tool(
//   (_, config) => {
//     const { user_id } = config.context
//     return user_id === '1' ? 'Florida' : 'SF'
//   },
//   {
//     name: 'get_user_location',
//     description: 'Retrieve user information based on user ID',
//   }
// )

const responseFormat = z.object({
  punny_response: z.string(),
  // weather_conditions: z.string().optional(),
})

// `thread_id` is a unique identifier for a given conversation.
const config = {
  configurable: { thread_id: '1' },
  context: { user_id: '1' },
}

const agent2 = createAgent({
  model: llm,
  tools: [tavilySearch],
  systemPrompt,
  responseFormat,
  checkpointer,
})

const res3 = await agent2.invoke(
  {
    messages: [{ role: 'user', content: "How is Tesla?" }],
  },
  config
)
console.log(res3.structuredResponse)


const thankYouResponse = await agent2.invoke(
  { messages: [{ role: 'user', content: 'thank you!' }] },
  config
)
console.log(thankYouResponse.structuredResponse)