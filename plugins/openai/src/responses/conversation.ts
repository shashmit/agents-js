// SPDX-FileCopyrightText: 2026 LiveKit, Inc.
//
// SPDX-License-Identifier: Apache-2.0
import { llm } from '@livekit/agents';
import type OpenAI from 'openai';

type InputItem = OpenAI.Responses.ResponseInputItem;

/** Output sent for a function call the server holds but that never produced a result. */
export const INTERRUPTED_TOOL_OUTPUT =
  'The tool call was interrupted before a result was returned.';

/** Upper bound on server-reported missing tool outputs patched within a single turn. */
export const MAX_TOOL_OUTPUT_RECOVERIES = 3;

const MISSING_TOOL_OUTPUT_RE = /No tool output found for function call ([A-Za-z0-9_-]+)/;

/** Extracts the call id from OpenAI's "No tool output found for function call ..." error. */
export function parseMissingToolOutputCallId(message: string | undefined): string | undefined {
  return message?.match(MISSING_TOOL_OUTPUT_RE)?.[1];
}

function normalizeText(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

/**
 * Client-side mirror of what an OpenAI Conversation already holds.
 *
 * With the Conversations API the server persists every completed response's input and output
 * items and prepends them to the next request. Re-sending them would duplicate history (and
 * tokens) inside the conversation, so each turn only sends items the server does not have:
 *
 * - System/developer messages are sent through the request's `instructions` field instead of
 *   as input items. They apply to every turn but are never written into the conversation, so
 *   instructions that change between turns or sessions do not pile up server-side.
 * - Function calls and assistant text the model produced are already stored, so their local
 *   copies (created by the agent from the streamed output) are skipped.
 * - A function call the server holds without an output (for example, a tool interrupted by the
 *   user) would make OpenAI reject the request, so a placeholder output is sent for it.
 *
 * State only advances when a response completes, which is when OpenAI commits the items.
 */
export class ConversationState {
  readonly conversationId: string;

  /** Local chat item ids that the server conversation already reflects. */
  #syncedItemIds = new Set<string>();
  /** Call ids of function calls stored in the conversation. */
  #serverCallIds = new Set<string>();
  /** Call ids whose function_call_output is stored in the conversation. */
  #resolvedCallIds = new Set<string>();
  /** Assistant text the model produced that is not yet matched to a local chat item. */
  #serverAssistantTexts: string[] = [];

  constructor(conversationId: string) {
    this.conversationId = conversationId;
  }

  beginTurn(chatCtx: llm.ChatContext): ConversationTurn {
    const instructions: string[] = [];
    const items: llm.ChatItem[] = [];
    const seenItemIds: string[] = [];
    const matchedTextIndices = new Set<number>();

    for (const item of chatCtx.items) {
      if (item.type === 'message' && (item.role === 'system' || item.role === 'developer')) {
        const text = item.rawTextContent;
        if (text) instructions.push(text);
        continue;
      }

      seenItemIds.push(item.id);
      if (this.#syncedItemIds.has(item.id)) continue;

      if (item.type === 'function_call' && this.#serverCallIds.has(item.callId)) continue;
      if (item.type === 'function_call_output' && this.#resolvedCallIds.has(item.callId)) {
        continue;
      }
      if (item.type === 'message' && item.role === 'assistant') {
        const textIndex = this.#matchServerText(item, matchedTextIndices);
        if (textIndex !== undefined) {
          matchedTextIndices.add(textIndex);
          continue;
        }
      }

      items.push(item);
    }

    return new ConversationTurn(this, {
      instructions: instructions.length > 0 ? instructions.join('\n\n') : undefined,
      items,
      seenItemIds,
      matchedTextIndices,
    });
  }

  unresolvedServerCallIds(): string[] {
    return [...this.#serverCallIds].filter((callId) => !this.#resolvedCallIds.has(callId));
  }

  /** @internal */
  _commit(update: {
    seenItemIds: string[];
    matchedTextIndices: Set<number>;
    callIds: string[];
    resolvedCallIds: string[];
    outputText: string;
  }): void {
    for (const id of update.seenItemIds) this.#syncedItemIds.add(id);
    for (const callId of update.callIds) this.#serverCallIds.add(callId);
    for (const callId of update.resolvedCallIds) this.#resolvedCallIds.add(callId);

    this.#serverAssistantTexts = this.#serverAssistantTexts.filter(
      (_, index) => !update.matchedTextIndices.has(index),
    );
    const outputText = normalizeText(update.outputText);
    if (outputText) this.#serverAssistantTexts.push(outputText);
  }

  /**
   * Finds stored model output that a local assistant message was created from. The local copy
   * can be shorter than the stored one when playback was interrupted and the agent truncated the
   * message to what was actually spoken.
   */
  #matchServerText(item: llm.ChatMessage, exclude: Set<number>): number | undefined {
    const text = normalizeText(item.rawTextContent ?? '');
    if (!text) return undefined;

    const index = this.#serverAssistantTexts.findIndex(
      (serverText, i) => !exclude.has(i) && serverText.startsWith(text),
    );
    return index === -1 ? undefined : index;
  }
}

/** The items one request sends to a conversation, plus what it learned from the response. */
export class ConversationTurn {
  /** System/developer messages, to be sent as the request's `instructions`. */
  readonly instructions: string | undefined;

  #state: ConversationState;
  #items: llm.ChatItem[];
  #seenItemIds: string[];
  #matchedTextIndices: Set<number>;
  #recoveredCallIds: string[] = [];
  #outputCallIds: string[] = [];
  #outputText = '';
  #committed = false;

  constructor(
    state: ConversationState,
    {
      instructions,
      items,
      seenItemIds,
      matchedTextIndices,
    }: {
      instructions: string | undefined;
      items: llm.ChatItem[];
      seenItemIds: string[];
      matchedTextIndices: Set<number>;
    },
  ) {
    this.#state = state;
    this.instructions = instructions;
    this.#items = items;
    this.#seenItemIds = seenItemIds;
    this.#matchedTextIndices = matchedTextIndices;
  }

  get conversationId(): string {
    return this.#state.conversationId;
  }

  /**
   * Builds the request input. Unlike `ChatContext.toProviderFormat`, a function call output is
   * kept even when its call is not part of this input, because the call lives server-side.
   */
  async toInput(): Promise<InputItem[]> {
    const outputCallIds = new Set(
      this.#items
        .filter((item): item is llm.FunctionCallOutput => item.type === 'function_call_output')
        .map((item) => item.callId),
    );
    const needsOutput = (callId: string) =>
      !outputCallIds.has(callId) && !this.#recoveredCallIds.includes(callId);

    const input: InputItem[] = [];

    // Calls stored by a previous response that never got an output must be answered before
    // anything else, or the server rejects the request.
    for (const callId of this.#state.unresolvedServerCallIds()) {
      if (needsOutput(callId)) input.push(interruptedToolOutput(callId));
    }
    for (const callId of this.#recoveredCallIds) {
      input.push(interruptedToolOutput(callId));
    }

    for (const item of this.#items) {
      if (item.type === 'message') {
        const converted = (await new llm.ChatContext([item]).toProviderFormat(
          'openai.responses',
        )) as InputItem[];
        input.push(...converted);
      } else if (item.type === 'function_call') {
        input.push({
          type: 'function_call',
          call_id: item.callId,
          name: item.name,
          arguments: item.args,
        });
        if (needsOutput(item.callId)) input.push(interruptedToolOutput(item.callId));
      } else if (item.type === 'function_call_output') {
        input.push({ type: 'function_call_output', call_id: item.callId, output: item.output });
      }
      // Other item types (e.g. agent handoffs) are local bookkeeping with no Responses form.
    }

    return input;
  }

  /**
   * Answers a call the server reported as missing an output (e.g. one left by a previous agent
   * process). Returns false when the call was already patched or the recovery budget is spent,
   * meaning the request should fail instead of retrying.
   */
  recoverMissingToolOutput(callId: string): boolean {
    if (this.#recoveredCallIds.includes(callId)) return false;
    if (this.#recoveredCallIds.length >= MAX_TOOL_OUTPUT_RECOVERIES) return false;
    this.#recoveredCallIds.push(callId);
    return true;
  }

  /** Clears output recorded by a previous attempt of this turn. */
  resetOutput(): void {
    this.#outputCallIds = [];
    this.#outputText = '';
  }

  recordFunctionCall(callId: string): void {
    this.#outputCallIds.push(callId);
  }

  recordText(delta: string): void {
    this.#outputText += delta;
  }

  /** Marks the turn's input and output as stored by the server. Call on `response.completed`. */
  commit(): void {
    if (this.#committed) return;
    this.#committed = true;

    const sentCallIds = this.#items
      .filter((item): item is llm.FunctionCall => item.type === 'function_call')
      .map((item) => item.callId);
    const resolvedCallIds = [
      ...this.#items
        .filter((item): item is llm.FunctionCallOutput => item.type === 'function_call_output')
        .map((item) => item.callId),
      ...this.#recoveredCallIds,
      // every call without an output got a placeholder in toInput()
      ...sentCallIds,
      ...this.#state.unresolvedServerCallIds(),
    ];

    this.#state._commit({
      seenItemIds: this.#seenItemIds,
      matchedTextIndices: this.#matchedTextIndices,
      callIds: [...sentCallIds, ...this.#outputCallIds],
      resolvedCallIds,
      outputText: this.#outputText,
    });
  }
}

function interruptedToolOutput(callId: string): InputItem {
  return { type: 'function_call_output', call_id: callId, output: INTERRUPTED_TOOL_OUTPUT };
}
