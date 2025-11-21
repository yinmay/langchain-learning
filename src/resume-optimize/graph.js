import { StateGraph, START, END, MemorySaver } from '@langchain/langgraph'
import { ResumeState } from './state.js'
import {
  parseResumeNode,
  evaluateSkillsNode,
  evaluateProjectsNode,
  generateSuggestionsNode,
} from './nodes.js'

// Memory saver for conversation persistence
const checkpointer = new MemorySaver()

// Build the workflow
const workflow = new StateGraph(ResumeState)
  .addNode('parseResume', parseResumeNode)
  .addNode('evaluateSkills', evaluateSkillsNode)
  .addNode('evaluateProjects', evaluateProjectsNode)
  .addNode('generateSuggestions', generateSuggestionsNode)
  .addEdge(START, 'parseResume')
  .addEdge('parseResume', 'evaluateSkills')
  .addEdge('evaluateSkills', 'evaluateProjects')
  .addEdge('evaluateProjects', 'generateSuggestions')
  .addEdge('generateSuggestions', END)

export const app = workflow.compile({ checkpointer })
