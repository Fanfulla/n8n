import { isRecord } from '@n8n/utils/is-record';

import type { GeminiContentBlock, GeminiReplayContent } from './types';

type GeminiReplay = {
	thoughtSignature?: string;
	signatures?: string[];
	content?: GeminiReplayContent;
};

const REPLAY_BLOCK_TYPES = new Set([
	'text',
	'reasoning',
	'image_url',
	'media',
	'image',
	'audio',
	'file',
	'input_audio',
]);

function readStringArray(value: unknown): string[] | undefined {
	if (!Array.isArray(value) || value.length === 0) return undefined;
	if (!value.every((entry) => typeof entry === 'string')) return undefined;
	return value;
}

function readSignatureMap(value: unknown): Record<string, string> | undefined {
	if (!isRecord(value)) return undefined;
	const signatures: Record<string, string> = {};
	for (const [key, entry] of Object.entries(value)) {
		if (typeof entry === 'string' && entry.length > 0) signatures[key] = entry;
	}
	return Object.keys(signatures).length > 0 ? signatures : undefined;
}

/**
 * Counts content parts the way @langchain/google-common does.
 * Returns undefined when a block would not survive that conversion.
 */
function geminiContentPartCount(content: unknown): number | undefined {
	if (content === undefined || content === null || content === '') return 0;
	if (typeof content === 'string') return 1;
	if (!Array.isArray(content)) return undefined;

	let count = 0;
	for (const block of content) {
		if (!isRecord(block)) return undefined;
		const type = block.type;
		if (typeof type !== 'string' || !REPLAY_BLOCK_TYPES.has(type)) return undefined;
		if (type === 'text') {
			if (typeof block.text === 'string' && block.text.length > 0) count += 1;
			continue;
		}
		if (type === 'reasoning') {
			if (typeof block.reasoning === 'string' && block.reasoning.length > 0) count += 1;
			continue;
		}
		count += 1;
	}
	return count;
}

function signaturesMatchGeminiParts(
	content: unknown,
	toolCallCount: number,
	signatures: readonly string[],
): boolean {
	const contentParts = geminiContentPartCount(content);
	if (contentParts === undefined || toolCallCount < 1) return false;
	return signatures.length === contentParts + toolCallCount;
}

/**
 * Returns the signature on the first function call.
 * A leading thought or text part can carry a different signature. Gemini 3
 * rejects the step when the function call does not keep its own signature.
 * A streamed chunk can append one empty slot. The signature then sits one
 * index earlier.
 */
function signatureForFirstFunctionCall(
	signatures: readonly string[],
	toolCallCount: number,
): string | undefined {
	if (signatures.length === 0) return undefined;
	const count = toolCallCount > 0 ? toolCallCount : 1;
	const alignedIndex = signatures.length - count;
	const aligned = alignedIndex >= 0 ? signatures[alignedIndex] : undefined;
	if (aligned) return aligned;
	const previous = alignedIndex > 0 ? signatures[alignedIndex - 1] : undefined;
	if (previous) return previous;
	return signatures.find((signature) => signature.length > 0);
}

function readReplayContent(value: unknown): GeminiReplayContent | undefined {
	if (typeof value === 'string') return value.length > 0 ? value : undefined;
	if (!Array.isArray(value) || value.length === 0) return undefined;
	if (geminiContentPartCount(value) === undefined) return undefined;

	const blocks: GeminiContentBlock[] = [];
	for (const entry of value) {
		if (!isRecord(entry) || typeof entry.type !== 'string') return undefined;
		blocks.push({ ...entry, type: entry.type });
	}
	return blocks;
}

/**
 * Returns the original parts when the signature array lines up with them.
 * A mismatch means the array would be dropped, so the caller keeps the
 * single-signature replay instead.
 */
export function geminiReplayForToolCount(
	google: { signatures?: string[]; content?: unknown } | undefined,
	toolCallCount: number,
): { content: GeminiReplayContent; signatures: string[] } | undefined {
	const signatures = google?.signatures;
	if (!signatures || toolCallCount < 1) return undefined;
	const content = readReplayContent(google.content) ?? [];
	if (!signaturesMatchGeminiParts(content, toolCallCount, signatures)) return undefined;
	return { content, signatures: [...signatures] };
}

export function readGeminiReplay(input: {
	toolCallId: string;
	additionalKwargs?: Record<string, unknown>;
	messages?: unknown[];
}): GeminiReplay {
	let signatures = input.additionalKwargs
		? readStringArray(input.additionalKwargs.signatures)
		: undefined;
	let signatureMap = input.additionalKwargs
		? readSignatureMap(input.additionalKwargs.__gemini_function_call_thought_signatures__)
		: undefined;
	let toolCallCount = 1;
	let content: GeminiReplayContent | undefined;

	for (const message of input.messages ?? []) {
		if (!isRecord(message)) continue;
		const kwargs = isRecord(message.additional_kwargs) ? message.additional_kwargs : undefined;
		const messageSignatures = kwargs ? readStringArray(kwargs.signatures) : undefined;
		if (messageSignatures) signatures = messageSignatures;
		const messageMap = kwargs
			? readSignatureMap(kwargs.__gemini_function_call_thought_signatures__)
			: undefined;
		if (messageMap) signatureMap = messageMap;
		if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
			toolCallCount = message.tool_calls.length;
		}
		if (content === undefined) {
			const messageContent = readReplayContent(message.content);
			if (messageContent !== undefined) content = messageContent;
		}
	}

	const thoughtSignature = signatures
		? signatureForFirstFunctionCall(signatures, toolCallCount)
		: signatureMap
			? (signatureMap[input.toolCallId] ?? Object.values(signatureMap)[0])
			: undefined;

	return {
		...(thoughtSignature ? { thoughtSignature } : {}),
		...(signatures ? { signatures } : {}),
		...(content !== undefined ? { content } : {}),
	};
}
