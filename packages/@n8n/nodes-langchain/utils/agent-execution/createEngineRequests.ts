import type { DynamicStructuredTool, Tool } from '@langchain/classic/tools';
import isObject from 'lodash/isObject';
import omit from 'lodash/omit';
import type { EngineRequest, IDataObject } from 'n8n-workflow';
import { NodeConnectionTypes } from 'n8n-workflow';

import { readGeminiReplay } from './geminiThoughtSignature';
import type {
	HitlMetadata,
	RequestResponseMetadata,
	ThinkingMetadata,
	ToolCallRequest,
	ToolMetadata,
} from './types';
import { isGeminiThoughtSignatureBlock, isRedactedThinkingBlock, isThinkingBlock } from './types';

export function hasGatedToolNodeName(
	metadata: unknown,
): metadata is ToolMetadata & { gatedToolNodeName: string } {
	return (
		isObject(metadata) &&
		typeof (metadata as Record<string, unknown>).gatedToolNodeName === 'string'
	);
}

export function extractHitlMetadata(
	metadata: ToolMetadata,
	toolName: string,
	toolInput: IDataObject,
): HitlMetadata | undefined {
	if (!hasGatedToolNodeName(metadata)) return undefined;

	return {
		gatedToolNodeName: metadata.gatedToolNodeName,
		toolName,
		originalInput: toolInput.toolParameters as IDataObject,
	};
}

/**
 * Extracts thinking metadata from tool call, with fallback to shared batch data.
 * Handles both Gemini thought signatures and Anthropic thinking blocks.
 */
function extractThinkingMetadata(
	toolCall: ToolCallRequest,
	sharedMessageLog: unknown[] | undefined,
	sharedAdditionalKwargs: Record<string, unknown> | undefined,
): ThinkingMetadata {
	const result: ThinkingMetadata = {};

	// Use toolCall's additionalKwargs or fall back to shared one from batch
	const effectiveAdditionalKwargs =
		(toolCall.additionalKwargs as Record<string, unknown> | undefined) ?? sharedAdditionalKwargs;
	// Use toolCall's messageLog or fall back to shared one from batch
	const effectiveMessageLog =
		toolCall.messageLog && toolCall.messageLog.length > 0 ? toolCall.messageLog : sharedMessageLog;

	// Keep the signature array and the content parts. A later turn has to send
	// each value back on the same part. The first non-empty value can belong
	// to a thought part, so it is not the function-call signature.
	const geminiReplay = readGeminiReplay({
		toolCallId: toolCall.toolCallId,
		additionalKwargs: effectiveAdditionalKwargs,
		messages: effectiveMessageLog,
	});
	let thoughtSignature = geminiReplay.thoughtSignature;

	// Extract DeepSeek reasoning_content (must be echoed back on tool-calling turns)
	let reasoningContent: string | undefined;
	if (effectiveAdditionalKwargs) {
		const rc = effectiveAdditionalKwargs.reasoning_content;
		if (typeof rc === 'string' && rc.length > 0) reasoningContent = rc;
	}

	// Extract thinking content and additional thought signatures from messageLog
	let thinkingContent: string | undefined;
	let thinkingType: 'thinking' | 'redacted_thinking' | undefined;
	let thinkingSignature: string | undefined;

	if (effectiveMessageLog && Array.isArray(effectiveMessageLog)) {
		for (const message of effectiveMessageLog) {
			// Check if message has content that could contain thought_signature or thinking blocks
			if (message && typeof message === 'object' && 'content' in message) {
				const content = message.content;
				// Content can be string or array of content blocks
				if (Array.isArray(content)) {
					// Look for thought_signature in content blocks (Gemini)
					// and thinking/redacted_thinking blocks (Anthropic)
					for (const block of content) {
						// Gemini thought_signature as content block (only if not already found)
						if (!thoughtSignature && isGeminiThoughtSignatureBlock(block)) {
							thoughtSignature = block.thoughtSignature;
						}

						// Anthropic thinking blocks
						if (isThinkingBlock(block)) {
							thinkingContent = block.thinking;
							thinkingType = 'thinking';
							thinkingSignature = block.signature;
						} else if (isRedactedThinkingBlock(block)) {
							thinkingContent = block.data;
							thinkingType = 'redacted_thinking';
						}
					}
				}

				// DeepSeek reasoning_content (fallback for the non-streaming path, where
				// ToolCallRequest.additionalKwargs isn't populated from the raw AIMessage).
				// Independent of the thoughtSignature gate above - must run regardless of
				// whether a Gemini signature was found.
				if (!reasoningContent && 'additional_kwargs' in message) {
					const msgAdditionalKwargs = message.additional_kwargs as
						| Record<string, unknown>
						| undefined;
					const rc = msgAdditionalKwargs?.reasoning_content;
					if (typeof rc === 'string' && rc.length > 0) reasoningContent = rc;
				}

				if (thoughtSignature || thinkingContent !== undefined || reasoningContent) break;
			}
		}
	}

	// Build result object
	if (thoughtSignature || geminiReplay.signatures) {
		result.google = {
			...(thoughtSignature ? { thoughtSignature } : {}),
			...(geminiReplay.signatures ? { signatures: geminiReplay.signatures } : {}),
			...(geminiReplay.signatures && geminiReplay.content !== undefined
				? { content: geminiReplay.content }
				: {}),
		};
	}

	// Empty thinking text is valid because the signature carries the reasoning.
	if (thinkingContent !== undefined && thinkingType) {
		result.anthropic = {
			thinkingContent,
			thinkingType,
			...(thinkingSignature ? { thinkingSignature } : {}),
		};
	}

	if (reasoningContent) {
		result.deepseek = { reasoningContent };
	}

	return result;
}

/**
 * Creates engine requests from tool calls.
 * Maps tool call information to the format expected by the n8n engine
 * for executing tool nodes.
 *
 * This is a generalized version that can be used across different agent types
 * (Tools Agent, OpenAI Functions Agent, etc.).
 *
 * @param toolCalls - Array of tool call requests to convert
 * @param itemIndex - The current item index
 * @param tools - Array of available tools
 * @returns Array of engine request objects (filtered to remove undefined entries)
 */
export function createEngineRequests(
	toolCalls: ToolCallRequest[],
	itemIndex: number,
	tools: Array<DynamicStructuredTool | Tool>,
): EngineRequest<RequestResponseMetadata>['actions'] {
	// For parallel tool calls, LangChain may only populate messageLog on the first action.
	// Find a shared messageLog to use for all tool calls in this batch.
	const sharedMessageLog = toolCalls.find(
		(tc) => tc.messageLog && tc.messageLog.length > 0,
	)?.messageLog;
	// Similarly for additionalKwargs (contains Gemini thought signatures)
	const sharedAdditionalKwargs = toolCalls.find((tc) => tc.additionalKwargs)?.additionalKwargs as
		| Record<string, unknown>
		| undefined;

	return toolCalls
		.map((toolCall) => {
			// First try to get from metadata (for toolkit tools)
			const foundTool = tools.find((tool) => tool.name === toolCall.tool);

			if (!foundTool) return undefined;

			const nodeName = foundTool.metadata?.sourceNodeName;

			if (typeof nodeName !== 'string') return undefined;

			const metadata = (foundTool.metadata ?? {}) as ToolMetadata;
			const toolInput = toolCall.toolInput as IDataObject;
			const hitlMetadata = extractHitlMetadata(metadata, toolCall.tool, toolInput);

			let input: IDataObject = toolInput;
			if (metadata.isFromToolkit) {
				input = { ...input, tool: toolCall.tool };
			}
			if (hitlMetadata) {
				// This input will be used as HITL node input
				input = {
					// omit hitlParameters, because they are destructured into the input instead
					...omit(input, 'hitlParameters'),
					...(input.hitlParameters as IDataObject),
					toolParameters: input.toolParameters,
				};
			}

			return {
				actionType: 'ExecutionNodeAction' as const,
				nodeName,
				input,
				type: NodeConnectionTypes.AiTool,
				id: toolCall.toolCallId,
				metadata: {
					itemIndex,
					hitl: hitlMetadata,
					toolName: toolCall.tool,
					...extractThinkingMetadata(toolCall, sharedMessageLog, sharedAdditionalKwargs),
				},
			};
		})
		.filter((item): item is NonNullable<typeof item> => item !== undefined);
}
