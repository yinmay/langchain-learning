import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { llm, llmWithTools, toolNode } from './llm.js'

// Node 1: Parse resume
export async function parseResumeNode(state) {
  const response = await llmWithTools.invoke([
    new SystemMessage(`You are a professional resume parsing expert. Please extract the following information from the resume:
1. Personal information (name, years of experience, contact info, etc.)
2. Technical skills (tech stack, tools, languages, etc.)
3. Project experience (project name, responsibilities, technologies, achievements)

Please call the parse_resume tool to process the resume.`),
    new HumanMessage(`Please parse the following resume:\n\n${state.resumeText}`),
  ])

  if (response.tool_calls?.length > 0) {
    await toolNode.invoke({ messages: [response] })
  }

  const extractResponse = await llm.invoke([
    new SystemMessage(`Extract information from the resume and return directly in JSON format:
{
  "personalInfo": "personal info summary",
  "yearsOfExperience": number,
  "skills": "skills list",
  "projects": "project experience summary"
}`),
    new HumanMessage(state.resumeText),
  ])

  try {
    const content = extractResponse.content
    const jsonMatch = content.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0])
      return {
        personalInfo: parsed.personalInfo || '',
        yearsOfExperience: parsed.yearsOfExperience || 0,
        skills: parsed.skills || '',
        projects: parsed.projects || '',
      }
    }
  } catch (e) {
    console.error('Parse failed:', e)
  }
  return {}
}

// Node 2: Evaluate skills
export async function evaluateSkillsNode(state) {
  const response = await llmWithTools.invoke([
    new SystemMessage(`You are a technical interview expert. Please evaluate whether the programmer's skills match their years of experience.

Evaluation dimensions:
- Skill depth: Are there core technologies that are deeply mastered?
- Skill breadth: Is the tech stack coverage reasonable?
- Experience match: Does the skill level meet expectations for years of experience?

Please call the evaluate_skills tool, then provide your evaluation conclusion.`),
    new HumanMessage(
      `Years of experience: ${state.yearsOfExperience}\nSkills: ${state.skills}`
    ),
  ])

  if (response.tool_calls?.length > 0) {
    await toolNode.invoke({ messages: [response] })
  }

  const evalResponse = await llm.invoke([
    new SystemMessage(
      `As a technical interview expert, evaluate the match between skills and years of experience. Provide evaluation result (Match/Below Expected/Above Expected) and specific analysis.`
    ),
    new HumanMessage(
      `Years of experience: ${state.yearsOfExperience}\nSkills: ${state.skills}`
    ),
  ])

  return { skillsEvaluation: evalResponse.content }
}

// Node 3: Evaluate projects
export async function evaluateProjectsNode(state) {
  const response = await llmWithTools.invoke([
    new SystemMessage(`You are a project experience evaluation expert. Please evaluate whether project experience matches years of experience.

Evaluation dimensions:
- Project complexity: Has the candidate taken on challenging projects?
- Scope of responsibility: Are there core responsibilities or leadership experience?
- Technical depth: Is the technology application in projects thorough?
- Quantified results: Are there measurable achievements?

Please call the evaluate_projects tool, then provide your evaluation conclusion.`),
    new HumanMessage(
      `Years of experience: ${state.yearsOfExperience}\nProject experience: ${state.projects}`
    ),
  ])

  if (response.tool_calls?.length > 0) {
    await toolNode.invoke({ messages: [response] })
  }

  const evalResponse = await llm.invoke([
    new SystemMessage(
      `As a project experience evaluation expert, evaluate the match between project experience and years of experience. Provide evaluation result and specific analysis.`
    ),
    new HumanMessage(
      `Years of experience: ${state.yearsOfExperience}\nProject experience: ${state.projects}`
    ),
  ])

  return { projectsEvaluation: evalResponse.content }
}

// Node 4: Generate suggestions
export async function generateSuggestionsNode(state) {
  const response = await llm.invoke([
    new SystemMessage(`You are a senior resume consultant. Based on the resume evaluation results, provide specific and actionable optimization suggestions.

Suggestions should include:
1. Overall assessment
2. Skills section improvement suggestions
3. Project experience improvement suggestions
4. Resume presentation suggestions`),
    new HumanMessage(`
Personal info: ${state.personalInfo}
Years of experience: ${state.yearsOfExperience}

Skills evaluation result:
${state.skillsEvaluation}

Projects evaluation result:
${state.projectsEvaluation}

Please provide comprehensive optimization suggestions.`),
  ])

  return { suggestions: response.content }
}
