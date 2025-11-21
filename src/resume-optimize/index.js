import 'dotenv/config'
import { randomUUID } from 'crypto'
import { app } from './graph.js'

// Test case
const testResume = `
John Smith
Frontend Developer | 7 years experience
Email: john.smith@example.com

Technical Skills:
- Proficient in HTML, CSS, JavaScript
- Familiar with React.js framework
- Basic knowledge of Node.js
- Experience with Git

Project Experience:
1. Company Website Development
   - Responsible for page layout and styling
   - Implemented interactive effects using jQuery

2. Admin Dashboard System
   - Participated in partial page development
   - Developed form components using Vue.js
`

async function main() {
  console.log('=== Resume Optimization Agent ===\n')
  console.log('Input Resume:\n', testResume)
  console.log('\nAnalyzing...\n')

  const result = await app.invoke(
    { resumeText: testResume },
    { configurable: { thread_id: randomUUID() } }
  )

  console.log('=== Optimization Suggestions ===\n')
  console.log(result.suggestions)
}

main().catch(console.error)
