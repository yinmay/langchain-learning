import { ChatDeepSeek } from '@langchain/deepseek'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import 'dotenv/config'

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
})

// Create a weather tool
const getWeatherTool = tool(
  async ({ city }) => {
    // Mock weather data for different cities
    const weatherData = {
      'new york': {
        temperature: 22,
        condition: 'Sunny',
        humidity: 65,
        windSpeed: 15,
      },
      'london': {
        temperature: 15,
        condition: 'Rainy',
        humidity: 80,
        windSpeed: 20,
      },
      'tokyo': {
        temperature: 18,
        condition: 'Cloudy',
        humidity: 70,
        windSpeed: 10,
      },
      'beijing': {
        temperature: 20,
        condition: 'Clear',
        humidity: 55,
        windSpeed: 12,
      },
      'paris': {
        temperature: 17,
        condition: 'Partly Cloudy',
        humidity: 68,
        windSpeed: 18,
      },
      'sydney': {
        temperature: 25,
        condition: 'Sunny',
        humidity: 60,
        windSpeed: 22,
      },
    }

    const cityKey = city.toLowerCase()
    const weather = weatherData[cityKey]

    if (weather) {
      return `Weather in ${city}:
- Temperature: ${weather.temperature}°C
- Condition: ${weather.condition}
- Humidity: ${weather.humidity}%
- Wind Speed: ${weather.windSpeed} km/h`
    } else {
      return `Weather data not available for ${city}. Available cities: ${Object.keys(weatherData).join(', ')}`
    }
  },
  {
    name: 'get_weather',
    description: 'Gets the current weather information for a specified city. Returns temperature, condition, humidity, and wind speed.',
    schema: z.object({
      city: z.string().describe('The name of the city to get weather for'),
    }),
  }
)

console.log('=== Weather Tool Demo ===\n')

// Test 1: Direct tool invocation
console.log('1. Direct Tool Invocation:')
const weatherResult1 = await getWeatherTool.invoke({ city: 'Tokyo' })
console.log(weatherResult1)
console.log('\n---\n')

const weatherResult2 = await getWeatherTool.invoke({ city: 'Paris' })
console.log(weatherResult2)
console.log('\n---\n')

// Test 2: LLM with weather tool
const llmWithTools = llm.bindTools([getWeatherTool])

console.log('2. LLM with Weather Tool:')
const response = await llmWithTools.invoke(
  "What's the weather like in London today?"
)

console.log('LLM Response:')
console.log(JSON.stringify(response, null, 2))
