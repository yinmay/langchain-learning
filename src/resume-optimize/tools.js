import { tool } from '@langchain/core/tools'
import { z } from 'zod'

// Tool: Parse resume
export const parseResumeTool = tool(
  async ({ resumeText }) => {
    return JSON.stringify({ parsed: true, text: resumeText })
  },
  {
    name: 'parse_resume',
    description: 'Parse resume text and extract personal info, skills, and project experience',
    schema: z.object({
      resumeText: z.string().describe('Original resume text'),
    }),
  }
)

// Tool: Evaluate skills match
export const evaluateSkillsTool = tool(
  async ({ skills, yearsOfExperience }) => {
    return JSON.stringify({ skills, yearsOfExperience, evaluated: true })
  },
  {
    name: 'evaluate_skills',
    description: 'Evaluate whether skill depth and breadth match years of experience',
    schema: z.object({
      skills: z.string().describe('List of skills'),
      yearsOfExperience: z.number().describe('Years of experience'),
    }),
  }
)

// Tool: Evaluate project experience
export const evaluateProjectsTool = tool(
  async ({ projects, yearsOfExperience }) => {
    return JSON.stringify({ projects, yearsOfExperience, evaluated: true })
  },
  {
    name: 'evaluate_projects',
    description: 'Evaluate whether project content and difficulty match years of experience',
    schema: z.object({
      projects: z.string().describe('Project experience'),
      yearsOfExperience: z.number().describe('Years of experience'),
    }),
  }
)

export const tools = [parseResumeTool, evaluateSkillsTool, evaluateProjectsTool]
